import os
import sys
import contextlib
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ==========================================
# 0. 断网防御与最强离线拦截锁
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

LOCAL_MODEL_DIR = "/home/star/zkx/iknow-audio/data/model"
CLAP_WEIGHTS_PATH = os.path.join(LOCAL_MODEL_DIR, "CLAP_weights_2023.pth")
GPT2_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "gpt2")
ROBERTA_LOCAL_PATH = "/home/star/zkx/CLAP/model/roberta-base"

import msclap.CLAPWrapper
def offline_hf_hub_download(*args, **kwargs):
    if not os.path.exists(CLAP_WEIGHTS_PATH):
        raise FileNotFoundError(f"未找到权重文件: {CLAP_WEIGHTS_PATH}")
    return CLAP_WEIGHTS_PATH
msclap.CLAPWrapper.hf_hub_download = offline_hf_hub_download

import transformers
def patch_transformers_offline(cls_name):
    cls = getattr(transformers, cls_name)
    orig_func = cls.from_pretrained
    @classmethod
    def my_func(cls_inner, pretrained_model_name_or_path, *args, **kwargs):
        target_path = GPT2_LOCAL_PATH if "gpt2" in str(pretrained_model_name_or_path).lower() else ROBERTA_LOCAL_PATH
        kwargs["local_files_only"] = True
        return orig_func.__func__(cls_inner, target_path, *args, **kwargs)
    setattr(cls, "from_pretrained", my_func)

for cls_name in ["AutoModel", "AutoConfig", "AutoTokenizer", "GPT2Tokenizer", "RobertaTokenizer"]:
    try: patch_transformers_offline(cls_name)
    except AttributeError: continue

from msclap import CLAP
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

# ==========================================
# 1. 核心参数定义
# ==========================================
ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv"
ESC50_AUDIO_DIR = "/home/star/zkx/CLAP/data/ESC-50/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 多跳与剪枝核心参数
TOP_K = 5   
TOP_M = 3              # 每跳检索的候选数
TOP_P = 5              # 🌟 剪枝阈值：多跳检索出众多词后，利用音频 Grounding 仅保留得分最高的 5 个词
LOGIT_SCALE = 100.0    # LME 公式的温度缩放

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

def get_kg_entity(esc_class):
    clean_name = esc_class.replace('_', ' ').strip()
    if '(' in clean_name:
        clean_name = clean_name.split('(')[0].strip()
    clean_name = clean_name.replace(' - ', ' ')
    return clean_name

# ==========================================
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 iKnow-audio ESC-50 (M³KG-RAG 多跳扩展 + 音频接地剪枝 双轨对比版)...")
    
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    TARGET_RELATIONS = ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children']
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in training_factory.relation_to_id]

    # 🌟 极速图谱缓存器：防止多跳带来巨大的运算开销
    kge_cache = {}
    def get_top_m_tails(head, relation, m=TOP_M):
        cache_key = (head, relation)
        if cache_key in kge_cache:
            return kge_cache[cache_key]
        
        if head not in training_factory.entity_to_id:
            fallback = head.split(' ')[-1]
            if fallback not in training_factory.entity_to_id:
                kge_cache[cache_key] = []
                return []
            head = fallback
            
        try:
            pred = predict_target(model=kge_model, head=head, relation=relation, triples_factory=training_factory)
            tails = pred.df.sort_values(by="score", ascending=False).head(m)['tail_label'].tolist()
            kge_cache[cache_key] = tails
            return tails
        except:
            kge_cache[cache_key] = []
            return []

    df = pd.read_csv(ESC50_CSV)
    unique_categories = sorted(df['category'].unique()) 
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    print(f"🎵 推理开始 (总计 {len(df)} 样本)...")
    baseline_ranks, kg_lme_ranks, kg_max_ranks = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        audio_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['category']]
        
        # 1. 音频特征提取
        audio_embed = clap_model.get_audio_embeddings([audio_path])
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 2. 计算基准得分
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        baseline_ranks.append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        final_scores_lme = cos_sim_orig.clone() 
        final_scores_max = cos_sim_orig.clone() 
        top_k_indices = sorted_indices_baseline[:TOP_K]

        # 3. M³KG-RAG 多跳查询与音频剪枝
        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            
            raw_multi_hop_tails = set()
            
            # --- 🚀 模块 A：2-Hop 多跳扩展 ---
            for r1 in VALID_RELATIONS:
                hop1_tails = get_top_m_tails(kg_entity_name, r1, TOP_M)
                for t1 in hop1_tails:
                    t1_clean = t1.lower().strip()
                    if t1_clean != orig_class_name.lower() and t1_clean not in class_labels_set:
                        raw_multi_hop_tails.add(t1)
                    
                    # 第 2 跳扩展
                    for r2 in VALID_RELATIONS:
                        hop2_tails = get_top_m_tails(t1, r2, TOP_M)
                        for t2 in hop2_tails:
                            t2_clean = t2.lower().strip()
                            if t2_clean != orig_class_name.lower() and t2_clean not in class_labels_set:
                                raw_multi_hop_tails.add(t2)
            
            multi_hop_tails_list = list(raw_multi_hop_tails)
            pruned_tails = []
            
            # --- ✂️ 模块 B：CLAP 音频接地剪枝 (GRASP) ---
            if len(multi_hop_tails_list) > 0:
                # 提取这些尾实体的文本特征
                tail_embs = clap_model.get_text_embeddings(multi_hop_tails_list)
                tail_embs = F.normalize(tail_embs, dim=-1)
                
                # 计算它们在当前音频中的 "存在分数 (Presence Score)"
                presence_scores = torch.matmul(audio_embed, tail_embs.T).squeeze()
                if presence_scores.dim() == 0:
                    presence_scores = presence_scores.unsqueeze(0)
                
                # 保留最匹配的 Top-P 个词，剪枝掉多跳带来的噪声
                keep_num = min(TOP_P, len(multi_hop_tails_list))
                _, top_indices = torch.topk(presence_scores, keep_num)
                
                pruned_tails = [multi_hop_tails_list[idx] for idx in top_indices.cpu().numpy()]
            
            # --- 🧩 模块 C：标签拼接与双公式对比验证 ---
            # 严格按照原文拼接要求，无自然语言废话
            enriched_prompts = [f"{orig_class_name}, {t}" for t in pruned_tails]
            
            if len(enriched_prompts) > 0:
                prompt_embs = clap_model.get_text_embeddings(enriched_prompts)
                prompt_embs = F.normalize(prompt_embs, dim=-1)
                cos_sim_prompts = torch.matmul(audio_embed, prompt_embs.T).squeeze()
                
                if cos_sim_prompts.dim() == 0:
                    cos_sim_prompts = cos_sim_prompts.unsqueeze(0)
                
                # ==== 公式 1：带温度缩放的 Log-Mean-Exp (LME) ====
                s_c_lme = cos_sim_orig[c_idx] * LOGIT_SCALE
                s_p_lme = cos_sim_prompts * LOGIT_SCALE
                all_s_lme = torch.cat([s_c_lme.unsqueeze(0), s_p_lme]) 
                final_scores_lme[c_idx] = (torch.logsumexp(all_s_lme, dim=0) - np.log(len(all_s_lme))) / LOGIT_SCALE
                
                # ==== 公式 2：Max Pooling + Alpha=0.85 ====
                best_prompt_score = torch.max(cos_sim_prompts)
                ALPHA = 0.85
                final_scores_max[c_idx] = (ALPHA * cos_sim_orig[c_idx]) + ((1.0 - ALPHA) * best_prompt_score)

        # 分别对两个公式的得分进行重排
        ranks_lme = torch.argsort(final_scores_lme, descending=True).cpu().numpy()
        kg_lme_ranks.append(np.where(ranks_lme == true_idx)[0][0] + 1)
        
        ranks_max = torch.argsort(final_scores_max, descending=True).cpu().numpy()
        kg_max_ranks.append(np.where(ranks_max == true_idx)[0][0] + 1)

    b_h1, b_h3, b_h5, b_mrr = compute_metrics(baseline_ranks)
    lme_h1, lme_h3, lme_h5, lme_mrr = compute_metrics(kg_lme_ranks)
    max_h1, max_h3, max_h5, max_mrr = compute_metrics(kg_max_ranks)

    print("\n" + "="*85)
    print(f"{'Metric':<10} | {'Baseline':<12} | {'+KG (多跳+剪枝+LME)':<22} | {'+KG (多跳+剪枝+Max加权)':<22}")
    print("-" * 85)
    print(f"{'Hit@1':<10} | {b_h1:<12.2f} | {lme_h1:<22.2f} | {max_h1:<22.2f}")
    print(f"{'Hit@3':<10} | {b_h3:<12.2f} | {lme_h3:<22.2f} | {max_h3:<22.2f}")
    print(f"{'Hit@5':<10} | {b_h5:<12.2f} | {lme_h5:<22.2f} | {max_h5:<22.2f}")
    print(f"{'MRR':<10} | {b_mrr:<12.2f} | {lme_mrr:<22.2f} | {max_mrr:<22.2f}")
    print("="*85)

if __name__ == "__main__":
    main()