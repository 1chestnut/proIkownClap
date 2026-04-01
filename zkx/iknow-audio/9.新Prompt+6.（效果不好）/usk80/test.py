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
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

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
# 1. 路径与核心参数定义 (US8K)
# ==========================================
US8K_CSV = "/home/star/zkx/iknow-audio/data/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
US8K_AUDIO_DIR = "/home/star/zkx/iknow-audio/data/UrbanSound8K/UrbanSound8K/audio"

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# 🌟 LLM 自然语言提示词文件路径
LLM_PROMPTS_PATH = "/data/zkx/zkx/iknow-audio/9.新Prompt+6./promps/us8k_llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 黄金胜利参数组合 (M3 终极机制)
TOP_K = 5       # 候选类别截断
TOP_M = 3       # 知识图谱每跳分支
TOP_P = 5       # GRASP 剪枝保留数
DECAY_GAMMA = 0.85 # 🌟 多跳衰减系数 (打压二跳噪音)
LOGIT_SCALE = 100.0

# 动态 α 参数范围
ALPHA_MIN = 0.4
ALPHA_MAX = 0.8

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

# 🌟 加载并解析 LLM 提示词
def load_llm_prompts(file_path):
    prompt_map = {}
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 找不到提示词文件 {file_path}，将回退到基础拼接格式。")
        return prompt_map
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, text in data.items():
            parts = key.split("||")
            if len(parts) == 3:
                sub, rel, obj = [p.strip().lower() for p in parts]
                prompt_map[f"{sub}||{rel}||{obj}"] = text
    return prompt_map

def get_kg_entity(us8k_class):
    mapping = {
        'air_conditioner': 'air conditioner', 'car_horn': 'car horn',
        'children_playing': 'children playing', 'dog_bark': 'dog barking', 
        'drilling': 'drilling', 'engine_idling': 'engine idling',
        'gun_shot': 'gunshot', 'jackhammer': 'jackhammer',
        'siren': 'siren', 'street_music': 'street music'
    }
    return mapping.get(us8k_class, us8k_class.replace('_', ' '))

# ==========================================
# 3. 主程序：多跳 + LLM自然语义 + 动态α融合
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 US8K 终极版: [多跳图谱] + [LLM自然语义] + [动态α融合]...")
    
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    print(f"📚 成功加载 {len(prompt_map)} 条 LLM 增强提示词。")
    
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())
    
    # 💡 严格约束：防止空间漂移
    HOP1_RELATIONS = [r for r in ['overlaps with', 'occurs in', 'associated with environment', 'localized in', 'used for', 'part of scene'] if r in AVAILABLE_RELATIONS]
    HOP2_RELATIONS = [r for r in ['overlaps with', 'part of scene', 'used for'] if r in AVAILABLE_RELATIONS]

    kge_cache = {}
    def get_top_m_tails(head, relation, m=TOP_M):
        cache_key = (head, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        query_entity = head
        if query_entity not in training_factory.entity_to_id:
            fallback = query_entity.split(' ')[0] 
            if fallback in training_factory.entity_to_id: query_entity = fallback
            else: return []
        try:
            pred = predict_target(model=kge_model, head=query_entity, relation=relation, triples_factory=training_factory)
            tails = pred.df.sort_values(by="score", ascending=False).head(m)['tail_label'].tolist()
            kge_cache[cache_key] = tails
            return tails
        except: return []

    df = pd.read_csv(US8K_CSV)
    unique_categories = sorted(df['class'].unique()) 
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    baseline_ranks, m3_llm_ranks = [], []
    alpha_values = []  # 记录每个样本的动态 α

    print(f"🎵 推理开始 (总计 {len(df)} 样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="M3+LLM Pipeline"):
        audio_path = os.path.join(US8K_AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['class']]
        
        audio_embed = clap_model.get_audio_embeddings([audio_path])
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 1. Baseline 计算
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        baseline_ranks.append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        # ========== 动态 α 计算 ==========
        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim
        alpha_dynamic = max(ALPHA_MIN, min(ALPHA_MAX, alpha_dynamic))
        alpha_values.append(alpha_dynamic)
        # ================================

        final_scores = cos_sim_orig.clone()
        top_k_indices = sorted_indices_baseline[:TOP_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            
            # 记录合并信息: dict[tail_name, {'prompt': str, 'is_hop2': bool}]
            candidate_info = {}
            
            # --- Hop 1 检索 ---
            for r1 in HOP1_RELATIONS:
                for t1 in get_top_m_tails(kg_entity_name, r1, TOP_M):
                    t1_c = t1.lower().strip()
                    if t1_c != orig_class_name.lower() and t1_c not in class_labels_set:
                        # 查表匹配 LLM 描述
                        lk1 = f"{kg_entity_name.lower()}||{r1.lower()}||{t1_c}"
                        p1 = prompt_map.get(lk1, f"{orig_class_name}, {t1}")
                        candidate_info[t1_c] = {'prompt': p1, 'is_hop2': False}
                        
                        # --- Hop 2 检索 ---
                        for r2 in HOP2_RELATIONS:
                            for t2 in get_top_m_tails(t1, r2, TOP_M):
                                t2_c = t2.lower().strip()
                                if t2_c != orig_class_name.lower() and t2_c not in class_labels_set:
                                    # 查表匹配 LLM 描述
                                    lk2 = f"{t1_c}||{r2.lower()}||{t2_c}"
                                    p2 = prompt_map.get(lk2, f"{orig_class_name}, {t2}")
                                    # 保留最高优先级 (Hop1 优先)
                                    if t2_c not in candidate_info:
                                        candidate_info[t2_c] = {'prompt': p2, 'is_hop2': True}
            
            if len(candidate_info) > 0:
                # 提取去重后的 prompts 和对应的衰减权重
                prompts = []
                gammas = []
                for info in candidate_info.values():
                    prompts.append(info['prompt'])
                    gammas.append(DECAY_GAMMA if info['is_hop2'] else 1.0)
                
                gamma_tensor = torch.tensor(gammas).to(DEVICE)
                
                p_embs = clap_model.get_text_embeddings(prompts)
                p_embs = F.normalize(p_embs, dim=-1)
                
                raw_p_scores = torch.matmul(audio_embed, p_embs.T).squeeze()
                if raw_p_scores.dim() == 0: raw_p_scores = raw_p_scores.unsqueeze(0)
                
                # 💡 GRASP 剪枝：应用衰减系数进行惩罚过滤
                decayed_p_scores = raw_p_scores * gamma_tensor
                _, top_p_idx = torch.topk(decayed_p_scores, min(TOP_P, len(prompts)))
                
                # 提取存活的衰减分数
                final_decayed_scores = decayed_p_scores[top_p_idx]
                
                # 💡 M3 终极聚合机制 (Soft+Decay + 动态α)
                decayed_p_logits = final_decayed_scores * LOGIT_SCALE
                # 软池化
                soft_prompt_score = (torch.logsumexp(decayed_p_logits, dim=0) - np.log(len(decayed_p_logits))) / LOGIT_SCALE
                # 动态α加权锚定
                final_scores[c_idx] = (alpha_dynamic * cos_sim_orig[c_idx]) + ((1.0 - alpha_dynamic) * soft_prompt_score)

        # 结算预测排名
        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        m3_llm_ranks.append(np.where(sorted_indices_kg == true_idx)[0][0] + 1)

    # ==========================================
    # 4. 打印格式化对比表
    # ==========================================
    b_res = compute_metrics(baseline_ranks)
    m3_res = compute_metrics(m3_llm_ranks)
    
    col_w = 22
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'M3-RAG + LLM 增强 (动态α)':<{col_w}}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    for i, m_name in enumerate(metrics):
        row_str = f"{m_name:<8} | {b_res[i]:<{col_w}.2f} | {m3_res[i]:<{col_w}.2f}"
        print(row_str)
    print("=" * len(header))


if __name__ == "__main__":
    main()