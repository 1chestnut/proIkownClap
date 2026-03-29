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
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 🌟 绑定 2 号 GPU

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
# 1. 核心参数定义 (ESC-50)
# ==========================================
ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv"
ESC50_AUDIO_DIR = "/home/star/zkx/CLAP/data/ESC-50/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/提示词扩展/esc50/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 实验超参数
TOP_K = 5   
TOP_M = 3 
LOGIT_SCALE = 100.0 
ALPHA_LIST = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9] # 🌟 待对齐的权重梯度

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

def get_kg_entity(esc_class):
    clean_name = esc_class.replace('_', ' ').strip()
    if '(' in clean_name: clean_name = clean_name.split('(')[0].strip()
    return clean_name.replace(' - ', ' ')

def load_llm_prompts(file_path):
    prompt_map = {}
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 找不到提示词文件 {file_path}")
        return prompt_map
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, text in data.items():
            parts = key.split("||")
            if len(parts) == 3:
                # 统一转为小写存储，确保查找成功
                prompt_map["||".join([p.strip().lower() for p in parts])] = text
    return prompt_map

# ==========================================
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 ESC-50 单跳+LLM文本扩充 全权重对比实验...")
    
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    print(f"📚 成功加载 {len(prompt_map)} 条 LLM 扩充语义。")
    
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    VALID_RELATIONS = [r for r in ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children'] if r in training_factory.relation_to_id]

    df = pd.read_csv(ESC50_CSV)
    unique_categories = sorted(df['category'].unique()) 
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # 🌟 初始化全轨道排名存储器
    results_ranks = {
        'Baseline': [],
        'LME': [],
        **{f'Alpha_{a}': [] for a in ALPHA_LIST}
    }

    print(f"🎵 推理开始...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ablation"):
        audio_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['category']]
        
        audio_embed = clap_model.get_audio_embeddings([audio_path])
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 1. Baseline
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        results_ranks['Baseline'].append(np.where(sorted_indices == true_idx)[0][0] + 1)

        # 准备得分板
        score_lme = cos_sim_orig.clone()
        score_alpha_map = {a: cos_sim_orig.clone() for a in ALPHA_LIST}
        
        top_k_indices = sorted_indices[:TOP_K]

        # 2. 对 Top-K 候选类进行单跳知识与 LLM 语义增强
        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            unique_prompts = []
            
            for r in VALID_RELATIONS:
                try:
                    # 🌟 核心：单跳查询 (predict_target)
                    pred = predict_target(model=kge_model, head=kg_entity_name, relation=r, triples_factory=training_factory)
                    tails = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
                    
                    for t in tails:
                        if t.lower() != orig_class_name.lower() and t.lower() not in class_labels_set:
                            # 🌟 核心：匹配 LLM 自然语义扩充
                            lookup_key = f"{kg_entity_name.lower()}||{r.lower()}||{t.lower()}"
                            if lookup_key in prompt_map:
                                unique_prompts.append(prompt_map[lookup_key])
                            else:
                                unique_prompts.append(f"{orig_class_name}, {t}")
                except: continue
            
            unique_prompts = list(set(unique_prompts)) # 去重

            if len(unique_prompts) > 0:
                p_embs = clap_model.get_text_embeddings(unique_prompts)
                p_embs = F.normalize(p_embs, dim=-1)
                p_scores = torch.matmul(audio_embed, p_embs.T).squeeze()
                if p_scores.dim() == 0: p_scores = p_scores.unsqueeze(0)
                
                # ==== 轨道 A: LME 对齐 (Log-Mean-Exp) ====
                s_c_logit = cos_sim_orig[c_idx] * LOGIT_SCALE
                s_p_logits = p_scores * LOGIT_SCALE
                all_logits = torch.cat([s_c_logit.unsqueeze(0), s_p_logits])
                score_lme[c_idx] = (torch.logsumexp(all_logits, dim=0) - np.log(len(all_logits))) / LOGIT_SCALE
                
                # ==== 轨道 B: Max(Alpha) 梯度计算 ====
                best_p_score = torch.max(p_scores) # 🌟 统一使用 Max Pooling
                orig_s = cos_sim_orig[c_idx]
                for a in ALPHA_LIST:
                    score_alpha_map[a][c_idx] = (a * orig_s) + ((1.0 - a) * best_p_score)

        # 3. 统计本样本在各权重下的最终名次
        results_ranks['LME'].append(np.where(torch.argsort(score_lme, descending=True).cpu().numpy() == true_idx)[0][0] + 1)
        for a in ALPHA_LIST:
            r_alpha = torch.argsort(score_alpha_map[a], descending=True).cpu().numpy()
            results_ranks[f'Alpha_{a}'].append(np.where(r_alpha == true_idx)[0][0] + 1)

    # ==========================================
    # 4. 打印对齐结果表
    # ==========================================
    metrics_list = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    col_w = 10
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'LME':<{col_w}}"
    for a in ALPHA_LIST: header += f" | {str(a):<{col_w-3}}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for i, m_name in enumerate(metrics_list):
        row = f"{m_name:<8} | "
        b_m = compute_metrics(results_ranks['Baseline'])[i]
        l_m = compute_metrics(results_ranks['LME'])[i]
        row += f"{b_m:<{col_w}.2f} | {l_m:<{col_w}.2f}"
        
        for a in ALPHA_LIST:
            val = compute_metrics(results_ranks[f'Alpha_{a}'])[i]
            row += f" | {val:<{col_w-3}.2f}"
        print(row)
    print("=" * len(header))

if __name__ == "__main__":
    main()