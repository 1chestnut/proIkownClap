import os
import sys
import contextlib
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json

# ==========================================
# 0. 断网防御、离线拦截锁与 GPU 绑定
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 绑定 GPU

LOCAL_MODEL_DIR = "/home/star/zkx/iknow-audio/data/model"
CLAP_WEIGHTS_PATH = os.path.join(LOCAL_MODEL_DIR, "CLAP_weights_2023.pth")
GPT2_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "gpt2")
ROBERTA_LOCAL_PATH = "/home/star/zkx/CLAP/model/roberta-base"

import msclap.CLAPWrapper
def offline_hf_hub_download(*args, **kwargs):
    if not os.path.exists(CLAP_WEIGHTS_PATH):
        raise FileNotFoundError(f"未找到权重: {CLAP_WEIGHTS_PATH}")
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
# 1. 核心参数与多模态 RAG 设置 (ESC-50)
# ==========================================
ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv"
ESC50_AUDIO_DIR = "/home/star/zkx/CLAP/data/ESC-50/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# 使用你原始效果最好的提示词字典
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/esc50/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 超参数
TOP_K = 5
TOP_M = 3
TOP_P = 5
DECAY_GAMMA = 0.85 # 二跳距离衰减系数
LOGIT_SCALE = 100.0

# 🌟 自适应早退的相对边距阈值
RELATIVE_MARGIN = -0.02

# 🔥 新增：负向知识惩罚系数
BETA_NEG = 0.3  

# 动态 α 参数范围
ALPHA_MIN = 0.4
ALPHA_MAX = 0.8

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return (np.mean(ranks <= 1) * 100,
            np.mean(ranks <= 3) * 100,
            np.mean(ranks <= 5) * 100,
            np.mean(1.0 / ranks) * 100)

def get_kg_entity(esc_class):
    clean_name = esc_class.replace('_', ' ').strip()
    if '(' in clean_name: clean_name = clean_name.split('(')[0].strip()
    return clean_name.replace(' - ', ' ')

def to_tensor(emb):
    if isinstance(emb, np.ndarray): return torch.from_numpy(emb)
    return emb

def load_llm_prompts(file_path):
    prompt_map = {}
    if not os.path.exists(file_path): return prompt_map
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, text in data.items():
            parts = key.split("||")
            if len(parts) == 3:
                sub, rel, obj = [p.strip().lower() for p in parts]
                prompt_map[f"{sub}||{rel}||{obj}"] = text
    return prompt_map

# ==========================================
# 2. 主程序
# ==========================================
@torch.no_grad()
def main():
    print(f"🚀 启动 ESC-50: [负向剥离 BETA={BETA_NEG}] + [静态早退 {RELATIVE_MARGIN}] + [动态α融合]...")

    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    ALL_RELS = ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children']
    HOP1_RELATIONS = [r for r in ALL_RELS if r in training_factory.relation_to_id]
    HOP2_RELATIONS = [r for r in ['belongs to class', 'has parent', 'event composed of', 'has children']
                      if r in training_factory.relation_to_id]

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
    text_embeds = to_tensor(text_embeds).to(DEVICE).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

    ranks = {'Baseline': [], 'Ours_AdaptiveM3': []}
    
    total_candidates_processed = 0
    hop2_triggered_count = 0

    print(f"🎵 推理开始 ({len(df)} 样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adaptive M3 Pipeline"):
        audio_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['category']]

        audio_embed_raw = clap_model.get_audio_embeddings([audio_path])
        audio_embed = to_tensor(audio_embed_raw).to(DEVICE).float()
        audio_embed = F.normalize(audio_embed, dim=-1)

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        ranks['Baseline'].append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = max(ALPHA_MIN, min(ALPHA_MAX, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim))

        score_m3 = cos_sim_orig.clone()
        top_k_indices = sorted_indices_baseline[:TOP_K]

        for c_idx in top_k_indices:
            total_candidates_processed += 1
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx])
            baseline_score = cos_sim_orig[c_idx].item()
            
            tau_dynamic = baseline_score + RELATIVE_MARGIN

            # ==========================================
            # 🌟 新增：提取并构建竞争者均值 (Hard Negative Center)
            # ==========================================
            competitor_indices = [j for j in top_k_indices if j != c_idx]
            if len(competitor_indices) > 0:
                mean_competitor_emb = torch.mean(text_embeds[competitor_indices], dim=0, keepdim=True)
            else:
                mean_competitor_emb = torch.zeros_like(text_embeds[0:1])
            # ==========================================

            candidate_info_hop1 = {}
            for r1 in HOP1_RELATIONS:
                for t1 in get_top_m_tails(kg_entity_name, r1, TOP_M):
                    t1_clean = t1.lower().strip()
                    if t1_clean != orig_class_name.lower() and t1_clean not in class_labels_set:
                        lk1 = f"{kg_entity_name.lower()}||{r1.lower()}||{t1_clean}"
                        candidate_info_hop1[t1_clean] = prompt_map.get(lk1, f"{orig_class_name}, {t1}")

            hop1_prompts = list(candidate_info_hop1.values())
            
            if len(hop1_prompts) == 0:
                max_hop1_score = -999.0
                hop1_scores = torch.tensor([]).to(DEVICE)
            else:
                p1_embs_list = [to_tensor(clap_model.get_text_embeddings([p])).to(DEVICE).float() for p in hop1_prompts]
                p1_embs_raw = torch.cat(p1_embs_list, dim=0)
                
                # 🔥 对 Hop-1 知识进行负向剥离
                p1_embs_contrastive = p1_embs_raw - BETA_NEG * mean_competitor_emb
                p1_embs = F.normalize(p1_embs_contrastive, dim=-1)
                
                hop1_scores = torch.matmul(audio_embed, p1_embs.T).squeeze()
                if hop1_scores.dim() == 0: hop1_scores = hop1_scores.unsqueeze(0)
                max_hop1_score = torch.max(hop1_scores).item()

            if max_hop1_score >= tau_dynamic:
                final_decayed_scores = hop1_scores * 1.0
                prompts_to_pool = hop1_prompts
            else:
                hop2_triggered_count += 1
                candidate_info_hop2 = {}
                for t1_clean in candidate_info_hop1.keys():
                    for r2 in HOP2_RELATIONS:
                        for t2 in get_top_m_tails(t1_clean, r2, TOP_M):
                            t2_clean = t2.lower().strip()
                            if t2_clean != orig_class_name.lower() and t2_clean not in class_labels_set:
                                lk2 = f"{t1_clean}||{r2.lower()}||{t2_clean}"
                                candidate_info_hop2[t2_clean] = prompt_map.get(lk2, f"{orig_class_name}, {t2}")
                
                hop2_prompts = list(candidate_info_hop2.values())
                if len(hop2_prompts) > 0:
                    p2_embs_list = [to_tensor(clap_model.get_text_embeddings([p])).to(DEVICE).float() for p in hop2_prompts]
                    p2_embs_raw = torch.cat(p2_embs_list, dim=0)
                    
                    # 🔥 对 Hop-2 知识也进行负向剥离 (给个0.5的衰减)
                    p2_embs_contrastive = p2_embs_raw - (BETA_NEG * 0.5) * mean_competitor_emb
                    p2_embs = F.normalize(p2_embs_contrastive, dim=-1)
                    
                    hop2_scores = torch.matmul(audio_embed, p2_embs.T).squeeze()
                    if hop2_scores.dim() == 0: hop2_scores = hop2_scores.unsqueeze(0)
                    
                    final_decayed_scores = torch.cat([hop1_scores * 1.0, hop2_scores * DECAY_GAMMA])
                    prompts_to_pool = hop1_prompts + hop2_prompts
                else:
                    final_decayed_scores = hop1_scores * 1.0
                    prompts_to_pool = hop1_prompts

            if len(prompts_to_pool) > 0:
                _, top_p_idx = torch.topk(final_decayed_scores, min(TOP_P, len(prompts_to_pool)))
                best_p_scores = final_decayed_scores[top_p_idx]
                
                decayed_p_logits = best_p_scores * LOGIT_SCALE
                soft_prompt_score = (torch.logsumexp(decayed_p_logits, dim=0) - np.log(len(decayed_p_logits))) / LOGIT_SCALE
                score_m3[c_idx] = (alpha_dynamic * baseline_score) + ((1.0 - alpha_dynamic) * soft_prompt_score)

        ranks['Ours_AdaptiveM3'].append(np.where(torch.argsort(score_m3, descending=True).cpu().numpy() == true_idx)[0][0] + 1)

    # ========== 结果输出 ==========
    b_res = compute_metrics(ranks['Baseline'])
    m3_res = compute_metrics(ranks['Ours_AdaptiveM3'])

    col_w = 26
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'Ours (Adaptive M3)':<{col_w}}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    for i, m_name in enumerate(metrics):
        print(f"{m_name:<8} | {b_res[i]:<{col_w}.2f} | {m3_res[i]:<{col_w}.2f}")
    print("=" * len(header))

    # ========== 效率指标统计 ==========
    hop2_rate = (hop2_triggered_count / total_candidates_processed) * 100
    print("\n⚡ [自适应检索效率统计]")
    print(f"动态阈值参数 (Relative Margin) : {RELATIVE_MARGIN}")
    print(f"负向惩罚系数 (Beta Neg) : {BETA_NEG}")
    print(f"二跳检索触发率 (Hop-2 Trigger Rate) : {hop2_rate:.1f}%")
    print(f"计算节省量 (Computational Savings) : 成功拦截了 {100 - hop2_rate:.1f}% 的冗余深层检索！")

if __name__ == "__main__":
    main()