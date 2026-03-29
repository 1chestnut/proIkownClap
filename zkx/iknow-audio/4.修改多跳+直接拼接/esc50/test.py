import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ==========================================
# 0. 环境与离线配置
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

LOCAL_MODEL_DIR = "/home/star/zkx/iknow-audio/data/model"
CLAP_WEIGHTS_PATH = os.path.join(LOCAL_MODEL_DIR, "CLAP_weights_2023.pth")
GPT2_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "gpt2")
ROBERTA_LOCAL_PATH = "/home/star/zkx/CLAP/model/roberta-base"

import msclap.CLAPWrapper
def offline_hf_hub_download(*args, **kwargs):
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
# 1. 核心参数与多模态 RAG 设置
# ==========================================
ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv"
ESC50_AUDIO_DIR = "/home/star/zkx/CLAP/data/ESC-50/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 M³KG-RAG 核心超参数
TOP_K = 5        # 基准截断
TOP_M = 3        # 每跳选取数量
TOP_P = 5        # GRASP 剪枝保留数
ALPHA = 0.6      # 类锚定权重
DECAY_GAMMA = 0.85 # 二跳距离衰减系数
LOGIT_SCALE = 100.0 

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

def get_kg_entity(esc_class):
    clean_name = esc_class.replace('_', ' ').strip()
    if '(' in clean_name: clean_name = clean_name.split('(')[0].strip()
    return clean_name.replace(' - ', ' ')

# ==========================================
# 3. 主程序：多聚合公式对比版本
# ==========================================
@torch.no_grad()
def main():
    print(f"🚀 启动 M³KG-RAG 机制消融实验: 验证多跳关系衰减与聚合函数...")
    
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    # 💡 结构层修复：严格区分第一跳和第二跳的允许关系
    ALL_RELS = ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children']
    HOP1_RELATIONS = [r for r in ALL_RELS if r in training_factory.relation_to_id]
    # 二跳禁止使用 perceived as 等容易导致语义发散的关系
    HOP2_RELATIONS = [r for r in ['belongs to class', 'has parent', 'event composed of', 'has children'] if r in training_factory.relation_to_id]

    kge_cache = {}
    def get_top_m_tails(head, relation, m=TOP_M):
        cache_key = (head, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        if head not in training_factory.entity_to_id:
            fallback = head.split(' ')[-1]
            if fallback not in training_factory.entity_to_id: return []
            head = fallback
        try:
            pred = predict_target(model=kge_model, head=head, relation=relation, triples_factory=training_factory)
            tails = pred.df.sort_values(by="score", ascending=False).head(m)['tail_label'].tolist()
            kge_cache[cache_key] = tails
            return tails
        except: return []

    df = pd.read_csv(ESC50_CSV)
    unique_categories = sorted(df['category'].unique()) 
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # 容器：记录各轨道的排名
    ranks = {
        'Baseline': [],
        'M1_MaxAlpha': [],
        'M2_LME_iKnow': [],
        'M3_SoftAlpha_Decay': []
    }

    print(f"🎵 开始推理 ({len(df)} 样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="M³KG-RAG Testing"):
        audio_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['category']]
        
        audio_embed = clap_model.get_audio_embeddings([audio_path])
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # [0] Baseline 计算
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        ranks['Baseline'].append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        # 初始化得分板
        score_m1 = cos_sim_orig.clone()
        score_m2 = cos_sim_orig.clone()
        score_m3 = cos_sim_orig.clone()

        top_k_indices = sorted_indices_baseline[:TOP_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            
            # 收集带 Hop 标签的实体: dict[tail_name, is_hop2_boolean]
            tail_hop_map = {} 
            
            # --- 严格受限的多跳检索 ---
            for r1 in HOP1_RELATIONS:
                hop1_tails = get_top_m_tails(kg_entity_name, r1, TOP_M)
                for t1 in hop1_tails:
                    if t1.lower() != orig_class_name.lower() and t1.lower() not in class_labels_set:
                        # 记录为 Hop 1
                        tail_hop_map[t1] = False 
                    
                    # 第二跳严格使用受限关系
                    for r2 in HOP2_RELATIONS:
                        hop2_tails = get_top_m_tails(t1, r2, TOP_M)
                        for t2 in hop2_tails:
                            if t2.lower() != orig_class_name.lower() and t2.lower() not in class_labels_set:
                                # 如果实体已作为一跳被发现，保留一跳的优先级；否则记为 Hop 2
                                if t2 not in tail_hop_map:
                                    tail_hop_map[t2] = True 
            
            candidate_tails = list(tail_hop_map.keys())
            
            if len(candidate_tails) > 0:
                # 生成拼接文本，并获取对应的 $\gamma$ 衰减数组
                prompts = [f"{orig_class_name}, {t}" for t in candidate_tails]
                gamma_tensor = torch.tensor([DECAY_GAMMA if tail_hop_map[t] else 1.0 for t in candidate_tails]).to(DEVICE)
                
                p_embs = clap_model.get_text_embeddings(prompts)
                p_embs = F.normalize(p_embs, dim=-1)
                
                # 计算 GRASP 音频-文本基准得分
                raw_p_scores = torch.matmul(audio_embed, p_embs.T).squeeze()
                if raw_p_scores.dim() == 0: raw_p_scores = raw_p_scores.unsqueeze(0)
                
                # 💡 剪枝层：Hop-Aware 的 GRASP
                # 应用衰减系数后再做 top-k 过滤，使得二跳实体更难被选中
                decayed_p_scores = raw_p_scores * gamma_tensor
                _, top_p_idx = torch.topk(decayed_p_scores, min(TOP_P, len(prompts)))
                
                # 提取最终存活的得分
                final_p_scores = raw_p_scores[top_p_idx]       # 未衰减的分数 (用于M1/M2)
                final_decayed_scores = decayed_p_scores[top_p_idx] # 衰减后的分数 (用于M3)
                
                # ==========================================
                # 💡 聚合层计算
                # ==========================================
                
                # M1: Max-Alpha (容易被多跳噪音带偏)
                best_p_val = torch.max(final_p_scores)
                score_m1[c_idx] = (ALPHA * cos_sim_orig[c_idx]) + ((1.0 - ALPHA) * best_p_val)
                
                # M2: 原版 iKnow LME (包含原类 + 未衰减的增强文本共同软池化)
                c_logit = cos_sim_orig[c_idx] * LOGIT_SCALE
                p_logits = final_p_scores * LOGIT_SCALE
                all_logits_m2 = torch.cat([c_logit.unsqueeze(0), p_logits])
                score_m2[c_idx] = (torch.logsumexp(all_logits_m2, dim=0) - np.log(len(all_logits_m2))) / LOGIT_SCALE
                
                # M3: Soft-Alpha Decay (M³KG-RAG 推荐形态)
                # 先对“衰减后的”提示词做 LogSumExp 软投票，求出上下文综合得分
                decayed_p_logits = final_decayed_scores * LOGIT_SCALE
                soft_prompt_score = (torch.logsumexp(decayed_p_logits, dim=0) - np.log(len(decayed_p_logits))) / LOGIT_SCALE
                # 再通过 Alpha 与原始类得分锚定
                score_m3[c_idx] = (ALPHA * cos_sim_orig[c_idx]) + ((1.0 - ALPHA) * soft_prompt_score)

        # 结算排名
        ranks['M1_MaxAlpha'].append(np.where(torch.argsort(score_m1, descending=True).cpu().numpy() == true_idx)[0][0] + 1)
        ranks['M2_LME_iKnow'].append(np.where(torch.argsort(score_m2, descending=True).cpu().numpy() == true_idx)[0][0] + 1)
        ranks['M3_SoftAlpha_Decay'].append(np.where(torch.argsort(score_m3, descending=True).cpu().numpy() == true_idx)[0][0] + 1)

    # ==========================================
    # 4. 展示横向对比表格
    # ==========================================
    print("\n" + "=" * 80)
    print(f"{'Metric':<8} | {'Baseline':<12} | {'M1(Max+Alpha)':<15} | {'M2(LME_iKnow)':<15} | {'M3(Soft+Decay)':<15}")
    print("-" * 80)
    
    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    b_res = compute_metrics(ranks['Baseline'])
    m1_res = compute_metrics(ranks['M1_MaxAlpha'])
    m2_res = compute_metrics(ranks['M2_LME_iKnow'])
    m3_res = compute_metrics(ranks['M3_SoftAlpha_Decay'])

    for i, m_name in enumerate(metrics):
        row = f"{m_name:<8} | {b_res[i]:<12.2f} | {m1_res[i]:<15.2f} | {m2_res[i]:<15.2f} | {m3_res[i]:<15.2f}"
        print(row)
    print("=" * 80)
    print("\n💡 公式解析:")
    print("M1 (Max+Alpha): 传统的 0.6 * Orig + 0.4 * Max(Prompts)。多跳容易翻车。")
    print("M2 (LME_iKnow): 原文 LogSumExp 池化。平滑，但不区分一跳二跳。")
    print("M3 (Soft+Decay): 对二跳衰减 0.85 惩罚，然后对 Prompts 求软均值(LogSumExp)，最后 0.6 比例融合原始分数。最稳健。")

if __name__ == "__main__":
    main()