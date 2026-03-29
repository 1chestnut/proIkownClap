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

# LLM 自然语言提示词文件路径
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/提示词扩展/esc50/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TOP_K = 5   
TOP_M = 3   

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

def get_kg_entity(esc_class):
    clean_name = esc_class.replace('_', ' ').strip()
    if '(' in clean_name:
        clean_name = clean_name.split('(')[0].strip()
    clean_name = clean_name.replace(' - ', ' ')
    return clean_name

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

# ==========================================
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 iKnow-audio ESC-50 离线复现 (Mean Pooling 破局版)...")
    
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    print(f"📚 成功加载 {len(prompt_map)} 条自然语言提示词。")
    
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    TARGET_RELATIONS = ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children']
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in training_factory.relation_to_id]

    df = pd.read_csv(ESC50_CSV)
    unique_categories = sorted(df['category'].unique()) 
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    # 预计算基准文本特征 (只包含单个类名)
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    print(f"🎵 推理开始 (总计 {len(df)} 样本)...")
    baseline_ranks, kg_ranks = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['category']]
        
        audio_embed = clap_model.get_audio_embeddings([audio_path])
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 1. 计算 Baseline 得分和排序
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        baseline_ranks.append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        final_scores = cos_sim_orig.clone() 
        top_k_indices = sorted_indices_baseline[:TOP_K]

        # 2. 对 Top-K 候选类进行 KG 和自然语言增强
        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            
            unique_prompts = set()
            
            # 搜集 KG 知识
            for r in VALID_RELATIONS:
                try:
                    pred = predict_target(model=kge_model, head=kg_entity_name, relation=r, triples_factory=training_factory)
                    tails = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
                    
                    for t in tails:
                        if t.lower() != orig_class_name.lower() and t.lower() not in class_labels_set:
                            lookup_key = f"{kg_entity_name.lower()}||{r.lower()}||{t.lower()}"
                            if lookup_key in prompt_map:
                                unique_prompts.add(prompt_map[lookup_key])
                            else:
                                unique_prompts.add(f"{orig_class_name}, {t}")
                except Exception: 
                    continue
            
            enriched_prompts = list(unique_prompts)
            
            if len(enriched_prompts) > 0:
                # 1. 提取自然语言提示词的特征
                embs_list = []
                for prompt in enriched_prompts:
                    single_emb = clap_model.get_text_embeddings([prompt])
                    embs_list.append(single_emb)
                
                prompt_embs = torch.cat(embs_list, dim=0) 
                prompt_embs = F.normalize(prompt_embs, dim=-1)
                
                # 2. 计算音频与所有提示词的相似度得分
                prompt_scores = torch.matmul(audio_embed, prompt_embs.T).squeeze()
                
                # 🌟 统一策略 1：坚决使用 Max Pooling（最高分机制），防止特征稀释！
                if prompt_scores.dim() == 0:
                    best_prompt_score = prompt_scores
                else:
                    best_prompt_score = torch.max(prompt_scores)
                
                # 🌟 统一策略 2：坚决锁定 ALPHA = 0.85（原特征占主导，自然语言打辅助）
                ALPHA = 0.85
                original_score = cos_sim_orig[c_idx]
                
                # Late Fusion 加权融合
                new_score = (ALPHA * original_score) + ((1.0 - ALPHA) * best_prompt_score)
                final_scores[c_idx] = new_score

        # 根据更新后的分数重新排序
        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        kg_ranks.append(np.where(sorted_indices_kg == true_idx)[0][0] + 1)

    b_h1, b_h3, b_h5, b_mrr = compute_metrics(baseline_ranks)
    kg_h1, kg_h3, kg_h5, kg_mrr = compute_metrics(kg_ranks)

    print("\n" + "="*60)
    print(f"{'Metric':<12} | {'Baseline':<15} | {'iKnow (+KG & NL Mean)':<20}")
    print("-" * 60)
    print(f"{'Hit@1':<12} | {b_h1:<15.2f} | {kg_h1:<15.2f}")
    print(f"{'Hit@3':<12} | {b_h3:<15.2f} | {kg_h3:<15.2f}")
    print(f"{'Hit@5':<12} | {b_h5:<15.2f} | {kg_h5:<15.2f}")
    print(f"{'MRR':<12} | {b_mrr:<15.2f} | {kg_mrr:<15.2f}")
    print("="*60)

if __name__ == "__main__":
    main()