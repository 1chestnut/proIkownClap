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
# 0. 断网防御与最强离线拦截锁
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 绑定 GPU

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
# 1. 路径与核心参数定义 (TUT2017)
# ==========================================
TUT_DIR = "/home/star/zkx/iknow-audio/data/TUT2017/development/TUT-acoustic-scenes-2017-development"
TUT_META = os.path.join(TUT_DIR, "meta.txt")

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# LLM 自然语言提示词文件路径
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/tut2017/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 核心超参数
TOP_K = 5
TOP_M = 3
TOP_P = 5
DECAY_GAMMA = 0.85   # 多跳衰减系数
LOGIT_SCALE = 100.0

# 🌟 [自适应深度] 早退的相对边距阈值
RELATIVE_MARGIN = -0.02

# 动态 α 参数范围
ALPHA_MIN = 0.4
ALPHA_MAX = 0.8

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return (np.mean(ranks <= 1) * 100,
            np.mean(ranks <= 3) * 100,
            np.mean(ranks <= 5) * 100,
            np.mean(1.0 / ranks) * 100)

def get_kg_entity(tut_class):
    mapping = {
        'cafe/restaurant': 'cafe or restaurant',
        'city_center': 'city center',
        'forest_path': 'forest path',
        'grocery_store': 'grocery store',
        'metro_station': 'metro station',
        'residential_area': 'residential area'
    }
    return mapping.get(tut_class, tut_class.replace('_', ' ').replace('/', ' or '))

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
# 3. 主程序：多跳 + 早退 + 动态α + 🔥分层级联池化
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 TUT2017 分层级联版: [早退] + [动态α] + [🔥深度隔离池化]...")
    
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())
    
    # 关系配置：TUT是场景识别，严防漂移
    HOP1_RELATIONS = [r for r in ['is variant of', 'has parent', 'scene contains', 'event composed of', 'described by'] 
                      if r in AVAILABLE_RELATIONS]
    HOP2_RELATIONS = [r for r in ['is variant of', 'has parent', 'described by', 'event composed of'] 
                      if r in AVAILABLE_RELATIONS]
    
    kge_cache = {}
    def get_top_m_tails(head_entity, relation, m=TOP_M):
        cache_key = (head_entity, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        query_entity = head_entity
        if query_entity not in training_factory.entity_to_id:
            fallback = query_entity.split(' ')[0] 
            if fallback in training_factory.entity_to_id: 
                query_entity = fallback
            else: return []
        try:
            pred = predict_target(model=kge_model, head=query_entity, relation=relation, triples_factory=training_factory)
            tails = pred.df.sort_values(by="score", ascending=False).head(m)['tail_label'].tolist()
            kge_cache[cache_key] = tails
            return tails
        except: return []

    # 数据准备
    data_records = []
    with open(TUT_META, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2: data_records.append({'audio_rel_path': parts[0], 'class': parts[1]})
    
    df = pd.DataFrame(data_records)
    unique_categories = sorted(df['class'].unique()) 
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ').replace('/', ' or ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    text_embeds = to_tensor(clap_model.get_text_embeddings(clean_classes)).to(DEVICE).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

    ranks = {'Baseline': [], 'Ours_Hierarchical': []}
    
    # 效率统计 (移除了广度截断，固定为 TOP_K)
    total_candidates_processed = 0
    hop2_triggered_count = 0

    print(f"🎵 推理开始 (总计 {len(df)} 样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Hierarchical Pipeline"):
        audio_path = os.path.join(TUT_DIR, row['audio_rel_path'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['class']]
        
        audio_embed_raw = clap_model.get_audio_embeddings([audio_path])
        audio_embed = to_tensor(audio_embed_raw).to(DEVICE).float()
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 1. Baseline 相似度
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        ranks['Baseline'].append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        # 2. 动态 α 计算
        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = max(ALPHA_MIN, min(ALPHA_MAX, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim))
        final_scores = cos_sim_orig.clone() 

        top_k_indices = sorted_indices_baseline[:TOP_K]

        for c_idx in top_k_indices:
            total_candidates_processed += 1
            orig_class_name = clean_classes[c_idx]
            kg_scene_name = get_kg_entity(unique_categories[c_idx]) 
            baseline_score = cos_sim_orig[c_idx].item()
            
            # 早退动态阈值
            tau_dynamic = baseline_score + RELATIVE_MARGIN
            
            # ==========================================
            # STAGE 1: 获取 Hop-1 并独立池化
            # ==========================================
            candidate_info_hop1 = {}
            for r1 in HOP1_RELATIONS:
                for t1 in get_top_m_tails(kg_scene_name, r1, TOP_M):
                    t1_c = t1.lower().strip()
                    if (t1_c != orig_class_name.lower() and t1_c not in class_labels_set and t1_c != kg_scene_name.lower()):
                        lk1 = f"{kg_scene_name.lower()}||{r1.lower()}||{t1_c}"
                        candidate_info_hop1[t1_c] = prompt_map.get(lk1, f"{orig_class_name}, {t1}")
            
            hop1_prompts = list(candidate_info_hop1.values())
            
            if len(hop1_prompts) == 0:
                max_hop1_score = -999.0
                pool_hop1 = torch.tensor(-999.0).to(DEVICE)
            else:
                p1_embs_list = [to_tensor(clap_model.get_text_embeddings([p])).to(DEVICE).float() for p in hop1_prompts]
                p1_embs = F.normalize(torch.cat(p1_embs_list, dim=0), dim=-1)
                
                hop1_scores = torch.matmul(audio_embed, p1_embs.T).squeeze()
                if hop1_scores.dim() == 0: hop1_scores = hop1_scores.unsqueeze(0)
                max_hop1_score = torch.max(hop1_scores).item()
                
                # 🔥 Hop-1 独立 GRASP 剪枝与池化
                hop1_top_p = torch.topk(hop1_scores, min(TOP_P, len(hop1_scores)))[0]
                hop1_logits = hop1_top_p * LOGIT_SCALE
                pool_hop1 = (torch.logsumexp(hop1_logits, dim=0) - np.log(len(hop1_logits))) / LOGIT_SCALE

            # ==========================================
            # STAGE 2: 阈值裁决与 Hop-2 独立池化
            # ==========================================
            if max_hop1_score >= tau_dynamic:
                # 🌟 第一跳及格，早退！直接使用 Hop-1 的池化分数
                soft_prompt_score = pool_hop1
            else:
                # ❌ 不及格，触发二跳
                hop2_triggered_count += 1
                candidate_info_hop2 = {}
                for t1_c in candidate_info_hop1.keys():
                    for r2 in HOP2_RELATIONS:
                        for t2 in get_top_m_tails(t1_c, r2, TOP_M):
                            t2_c = t2.lower().strip()
                            if (t2_c != orig_class_name.lower() and t2_c not in class_labels_set and t2_c != kg_scene_name.lower()):
                                if t2_c not in candidate_info_hop2:
                                    lk2 = f"{t1_c}||{r2.lower()}||{t2_c}"
                                    candidate_info_hop2[t2_c] = prompt_map.get(lk2, f"{orig_class_name}, {t2}")
                
                hop2_prompts = list(candidate_info_hop2.values())
                if len(hop2_prompts) > 0:
                    p2_embs_list = [to_tensor(clap_model.get_text_embeddings([p])).to(DEVICE).float() for p in hop2_prompts]
                    p2_embs = F.normalize(torch.cat(p2_embs_list, dim=0), dim=-1)
                    
                    hop2_scores = torch.matmul(audio_embed, p2_embs.T).squeeze()
                    if hop2_scores.dim() == 0: hop2_scores = hop2_scores.unsqueeze(0)
                    
                    # 🔥 Hop-2 独立 GRASP 剪枝与池化
                    hop2_top_p = torch.topk(hop2_scores, min(TOP_P, len(hop2_scores)))[0]
                    hop2_logits = hop2_top_p * LOGIT_SCALE
                    pool_hop2 = (torch.logsumexp(hop2_logits, dim=0) - np.log(len(hop2_logits))) / LOGIT_SCALE
                    
                    # 对二跳池化分数应用衰减
                    pool_hop2 = pool_hop2 * DECAY_GAMMA
                    
                    # 🌟 核心：分层级联隔离 (Max Pooling)！彻底杜绝稀释效应
                    soft_prompt_score = torch.max(pool_hop1, pool_hop2)
                else:
                    soft_prompt_score = pool_hop1

            # 最终的 M3 动态加权融合
            final_scores[c_idx] = (alpha_dynamic * baseline_score) + ((1.0 - alpha_dynamic) * soft_prompt_score)

        # 结算预测排名
        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        ranks['Ours_Hierarchical'].append(np.where(sorted_indices_kg == true_idx)[0][0] + 1)

    # ==========================================
    # 4. 打印格式化对比表
    # ==========================================
    b_res = compute_metrics(ranks['Baseline'])
    m3_res = compute_metrics(ranks['Ours_Hierarchical'])
    
    col_w = 28
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'Ours (Hierarchical M3)':<{col_w}}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    for i, m_name in enumerate(metrics):
        print(f"{m_name:<8} | {b_res[i]:<{col_w}.2f} | {m3_res[i]:<{col_w}.2f}")
    print("=" * len(header))

    # ========== 效率指标统计 ==========
    hop2_rate = (hop2_triggered_count / total_candidates_processed) * 100 if total_candidates_processed > 0 else 0
    
    print("\n⚡ [自适应检索效率统计]")
    print(f"动态阈值参数 (Relative Margin) : {RELATIVE_MARGIN}")
    print(f"二跳检索触发率 (Hop-2 Trigger Rate) : {hop2_rate:.1f}%")
    print(f"计算节省量 (Computational Savings) : 成功拦截了 {100 - hop2_rate:.1f}% 的冗余深层检索！")
    print("💡 结论: 引入深度隔离的分层级联池化，确保了一跳高保真提示词绝对不会被二跳发散所稀释。")

if __name__ == "__main__":
    main()