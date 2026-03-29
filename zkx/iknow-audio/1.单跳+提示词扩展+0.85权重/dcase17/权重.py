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
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 🌟 绑定 GPU

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
# 1. 路径与核心参数定义 (DCASE17-T4)
# ==========================================
DCASE_CSV = "/home/star/zkx/iknow-audio/data/DCASE17-T4/my_evaluation_dataset.csv"
DCASE_AUDIO_DIR = "/home/star/zkx/iknow-audio/data/DCASE17-T4/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/单跳+提示词扩展+0.85权重/dcase17/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 实验超参数
TOP_K = 5   
TOP_M = 3 
LOGIT_SCALE = 100.0 
ALPHA_LIST = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9] # 🌟 待对齐权重梯度

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

def load_llm_prompts(file_path):
    prompt_map = {}
    if not os.path.exists(file_path): return prompt_map
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, text in data.items():
            parts = key.split("||")
            if len(parts) == 3:
                prompt_map["||".join([p.strip().lower() for p in parts])] = text
    return prompt_map

# ==========================================
# 2. DCASE 专属逻辑
# ==========================================
DCASE_17_CLASSES = [
    'ambulance (siren)', 'bicycle', 'bus', 'car', 'car alarm', 'car passing by', 
    'civil defense siren', 'fire engine, fire truck (siren)', 'motorcycle', 
    'police car (siren)', 'reversing beeps', 'screaming', 'skateboard', 
    'train', 'train horn', 'truck', 'air horn, truck horn'
]

def get_kg_entity(class_name):
    mapping = {
        'ambulance (siren)': 'ambulance',
        'fire engine, fire truck (siren)': 'fire engine',
        'police car (siren)': 'police car',
        'air horn, truck horn': 'horn',
        'civil defense siren': 'siren',
        'reversing beeps': 'beep',
        'car passing by': 'car'
    }
    return mapping.get(class_name, class_name)

# ==========================================
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 DCASE17 单跳+LLM文本扩充 全权重对齐实验...")
    
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    VALID_RELATIONS = [r for r in ['indicates', 'described by', 'used for', 'associated with environment', 'has parent', 'is instance of'] if r in training_factory.relation_to_id]

    clean_classes = DCASE_17_CLASSES
    class_to_idx = {cat: i for i, cat in enumerate(clean_classes)}
    class_labels_set = set(clean_classes)
    
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    df = pd.read_csv(DCASE_CSV)
    
    # 🌟 初始化全轨道容器
    results_ranks = {
        'Baseline': [],
        'LME': [],
        **{f'Alpha_{a}': [] for a in ALPHA_LIST}
    }

    print(f"🎵 推理开始 (DCASE17)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Weights Search"):
        audio_path = os.path.join(DCASE_AUDIO_DIR, row['audio_filename'])
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0: continue
        
        # 多标签解析逻辑
        raw_label_str = str(row['paper_formatted_labels']).lower().strip()
        raw_label_str = raw_label_str.replace('fire engine, fire truck (siren)', 'C_FIRE').replace('air horn, truck horn', 'C_AIR')
        
        true_indices = []
        for part in raw_label_str.split(','):
            part = part.strip()
            if part == 'C_FIRE': true_indices.append(class_to_idx['fire engine, fire truck (siren)'])
            elif part == 'C_AIR': true_indices.append(class_to_idx['air horn, truck horn'])
            elif part in class_to_idx: true_indices.append(class_to_idx[part])
        
        true_indices = list(set(true_indices))
        if not true_indices: continue
        
        try:
            audio_embed = clap_model.get_audio_embeddings([audio_path])
        except: continue
        
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 1. Baseline
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        results_ranks['Baseline'].append(min([np.where(sorted_indices == t_idx)[0][0] + 1 for t_idx in true_indices]))

        # 初始化得分板
        score_lme = cos_sim_orig.clone()
        score_alpha_map = {a: cos_sim_orig.clone() for a in ALPHA_LIST}
        
        top_k_indices = sorted_indices[:TOP_K]

        # 2. 对 Top-K 候选类进行单跳+LLM增强
        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(orig_class_name) 
            unique_prompts = []
            
            for r in VALID_RELATIONS:
                try:
                    if kg_entity_name not in training_factory.entity_to_id: continue
                    pred = predict_target(model=kge_model, head=kg_entity_name, relation=r, triples_factory=training_factory)
                    tails = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
                    
                    for t in tails:
                        t_clean = t.lower().strip()
                        if t_clean != orig_class_name.lower() and t_clean not in class_labels_set:
                            # 查表匹配 LLM 自然语义
                            lookup_key = f"{kg_entity_name.lower()}||{r.lower()}||{t_clean}"
                            unique_prompts.append(prompt_map.get(lookup_key, f"{orig_class_name}, {t}"))
                except: continue
            
            unique_prompts = list(set(unique_prompts))

            if len(unique_prompts) > 0:
                p_embs = clap_model.get_text_embeddings(unique_prompts)
                p_embs = F.normalize(p_embs, dim=-1)
                p_scores = torch.matmul(audio_embed, p_embs.T).squeeze()
                if p_scores.dim() == 0: p_scores = p_scores.unsqueeze(0)
                
                # ==== 轨道 A: LME 对齐 ====
                s_c_logit = cos_sim_orig[c_idx] * LOGIT_SCALE
                s_p_logits = p_scores * LOGIT_SCALE
                all_logits = torch.cat([s_c_logit.unsqueeze(0), s_p_logits])
                score_lme[c_idx] = (torch.logsumexp(all_logits, dim=0) - np.log(len(all_logits))) / LOGIT_SCALE
                
                # ==== 轨道 B: Max(Alpha) 梯度计算 ====
                best_p_score = torch.max(p_scores)
                orig_s = cos_sim_orig[c_idx]
                for a in ALPHA_LIST:
                    score_alpha_map[a][c_idx] = (a * orig_s) + ((1.0 - a) * best_p_score)

    # 3. 结果排名结算 (多标签取最优排名)
        idx_lme = torch.argsort(score_lme, descending=True).cpu().numpy()
        results_ranks['LME'].append(min([np.where(idx_lme == t_idx)[0][0] + 1 for t_idx in true_indices]))
        for a in ALPHA_LIST:
            idx_alpha = torch.argsort(score_alpha_map[a], descending=True).cpu().numpy()
            results_ranks[f'Alpha_{a}'].append(min([np.where(idx_alpha == t_idx)[0][0] + 1 for t_idx in true_indices]))

    # ==========================================
    # 4. 打印对齐结果表
    # ==========================================
    metrics_list = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    col_w = 11
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'LME':<{col_w}}"
    for a in ALPHA_LIST: header += f" | {str(a):<{col_w-4}}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for i, m_name in enumerate(metrics_list):
        row = f"{m_name:<8} | "
        b_val = compute_metrics(results_ranks['Baseline'])[i]
        l_val = compute_metrics(results_ranks['LME'])[i]
        row += f"{b_val:<{col_w}.2f} | {l_val:<{col_w}.2f}"
        for a in ALPHA_LIST:
            val = compute_metrics(results_ranks[f'Alpha_{a}'])[i]
            row += f" | {val:<{col_w-4}.2f}"
        print(row)
    print("=" * len(header))

if __name__ == "__main__":
    main()