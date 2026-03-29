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
# 0. 环境与 GPU 绑定
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

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
# 1. 核心路径与参数定义
# ==========================================
ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv"
ESC50_AUDIO_DIR = "/home/star/zkx/CLAP/data/ESC-50/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/提示词扩展/esc50/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 核心胜利参数 (基于您的总结)
ALPHA = 0.6       
TOP_K = 5         
TOP_M = 3         
TOP_P = 5         

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

def get_kg_entity(esc_class):
    clean_name = esc_class.replace('_', ' ').strip()
    if '(' in clean_name: clean_name = clean_name.split('(')[0].strip()
    return clean_name.replace(' - ', ' ')

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
# 2. 核心推理函数
# ==========================================
@torch.no_grad()
def main():
    print(f"🚀 启动 ESC50 终极组合方案: 多跳 + 语义扩充 + 剪枝 (Alpha={ALPHA})")
    
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
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
    
    text_embeds = F.normalize(clap_model.get_text_embeddings(clean_classes), dim=-1)

    baseline_ranks, combined_ranks = [], []
    kge_cache = {}

    def cached_predict(head, rel):
        key = (head, rel)
        if key in kge_cache: return kge_cache[key]
        try:
            pred = predict_target(model=kge_model, head=head, relation=rel, triples_factory=training_factory)
            tails = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
            kge_cache[key] = tails
            return tails
        except: return []

    print(f"🎵 推理处理中...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ESC50 Combination"):
        audio_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['category']]
        
        audio_embed = F.normalize(clap_model.get_audio_embeddings([audio_path]), dim=-1)
        
        # 1. Baseline 得分
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        baseline_ranks.append(np.where(sorted_indices == true_idx)[0][0] + 1)

        # 2. 核心增强得分计算
        final_scores = cos_sim_orig.clone()
        top_k_indices = sorted_indices[:TOP_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_head = get_kg_entity(unique_categories[c_idx])
            
            # --- 步骤 A: 2-Hop 多跳扩展 ---
            multi_hop_candidates = [] # 记录关联，用于后续匹配语义
            raw_tails = set()

            for r1 in VALID_RELATIONS:
                tails1 = cached_predict(kg_head, r1)
                for t1 in tails1:
                    t1_l = t1.lower()
                    if t1_l != orig_class_name.lower() and t1_l not in class_labels_set:
                        multi_hop_candidates.append((kg_head, r1, t1))
                        raw_tails.add(t1)
                    # 第二跳
                    for r2 in VALID_RELATIONS:
                        tails2 = cached_predict(t1, r2)
                        for t2 in tails2:
                            t2_l = t2.lower()
                            if t2_l != orig_class_name.lower() and t2_l not in class_labels_set:
                                multi_hop_candidates.append((t1, r2, t2))
                                raw_tails.add(t2)

            # --- 步骤 B: 音频接地剪枝 (GRASP) ---
            if raw_tails:
                tail_list = list(raw_tails)
                tail_embs = F.normalize(clap_model.get_text_embeddings(tail_list), dim=-1)
                grasp_scores = torch.matmul(audio_embed, tail_embs.T).squeeze()
                if grasp_scores.dim() == 0: grasp_scores = grasp_scores.unsqueeze(0)
                
                # 仅保留在当前音频中得分最高的 Top-P 个概念词，过滤干扰
                _, top_p_idx = torch.topk(grasp_scores, min(TOP_P, len(tail_list)))
                pruned_tails_set = set([tail_list[i].lower() for i in top_p_idx.cpu().numpy()])

                # --- 步骤 C: 匹配 LLM 语义扩充字典 ---
                final_prompts = []
                for sub, rel, obj in multi_hop_candidates:
                    if obj.lower() in pruned_tails_set:
                        # 查找 LLM 扩充句子
                        lk = f"{sub.lower()}||{rel.lower()}||{obj.lower()}"
                        if lk in prompt_map:
                            final_prompts.append(prompt_map[lk])
                        else:
                            # 兜底：直接拼接
                            final_prompts.append(f"{orig_class_name}, {obj}")
                
                # --- 步骤 D: 计算融合得分 (Alpha=0.6) ---
                if final_prompts:
                    final_prompts = list(set(final_prompts))
                    p_embs = F.normalize(clap_model.get_text_embeddings(final_prompts), dim=-1)
                    # Max Pooling: 取最契合音频的那一条语义
                    best_p_score = torch.max(torch.matmul(audio_embed, p_embs.T))
                    
                    # Late Fusion
                    final_scores[c_idx] = (ALPHA * cos_sim_orig[c_idx]) + ((1.0 - ALPHA) * best_p_score)

        idx_combined = torch.argsort(final_scores, descending=True).cpu().numpy()
        combined_ranks.append(np.where(idx_combined == true_idx)[0][0] + 1)

    # 4. 打印最终结果
    b_metrics = compute_metrics(baseline_ranks)
    c_metrics = compute_metrics(combined_ranks)
    
    print("\n" + "="*65)
    print(f"{'Metric':<10} | {'Baseline':<15} | {'Final Combined (0.6)':<20}")
    print("-" * 65)
    for i, m in enumerate(['Hit@1', 'Hit@3', 'Hit@5', 'MRR']):
        print(f"{m:<10} | {b_metrics[i]:<15.2f} | {c_metrics[i]:<20.2f}")
    print("="*65)

if __name__ == "__main__":
    main()