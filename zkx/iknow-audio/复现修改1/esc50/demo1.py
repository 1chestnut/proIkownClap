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

# 🌟 路径定义
LOCAL_MODEL_DIR = "/home/star/zkx/iknow-audio/data/model"
CLAP_WEIGHTS_PATH = os.path.join(LOCAL_MODEL_DIR, "CLAP_weights_2023.pth")
GPT2_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "gpt2")
ROBERTA_LOCAL_PATH = "/home/star/zkx/CLAP/model/roberta-base"

# 🌟 离线锁 1：拦截权重下载，重定向到本地 .pth 文件
import msclap.CLAPWrapper
def offline_hf_hub_download(*args, **kwargs):
    if not os.path.exists(CLAP_WEIGHTS_PATH):
        raise FileNotFoundError(f"未找到权重文件: {CLAP_WEIGHTS_PATH}")
    print(f"✅ 已拦截联网请求，载入本地权重: {CLAP_WEIGHTS_PATH}")
    return CLAP_WEIGHTS_PATH
msclap.CLAPWrapper.hf_hub_download = offline_hf_hub_download

# 🌟 离线锁 2：拦截所有文本模型/分词器请求，定向到本地 GPT2 或 Roberta
import transformers

def patch_transformers_offline(cls_name):
    cls = getattr(transformers, cls_name)
    orig_func = cls.from_pretrained
    @classmethod
    def my_func(cls_inner, pretrained_model_name_or_path, *args, **kwargs):
        # 自动识别 msclap 调用的 gpt2 还是原本的 roberta
        target_path = GPT2_LOCAL_PATH if "gpt2" in pretrained_model_name_or_path.lower() else ROBERTA_LOCAL_PATH
        kwargs["local_files_only"] = True
        return orig_func.__func__(cls_inner, target_path, *args, **kwargs)
    setattr(cls, "from_pretrained", my_func)

# 覆盖所有可能的入口
for cls_name in ["AutoModel", "AutoConfig", "AutoTokenizer", "GPT2Tokenizer", "RobertaTokenizer"]:
    try:
        patch_transformers_offline(cls_name)
    except AttributeError:
        continue

print("✅ 离线环境拦截器已就绪。")

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

# 论文 Algorithm 1 逻辑
TOP_K = 5   
TOP_M = 3   
LOGIT_SCALE = 100.0  # 恢复对比学习温度，确保 LogSumExp 的 Soft-Max 效应

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
    print("🚀 启动 iKnow-audio ESC-50 离线复现 (MS-CLAP 基准)...")
    
    # 初始化微软 CLAP
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

    # 加载知识图谱模型
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    # 定义关系
    TARGET_RELATIONS = ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children']
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in training_factory.relation_to_id]

    df = pd.read_csv(ESC50_CSV)
    unique_categories = sorted(df['category'].unique()) 
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    # 预计算基准文本特征
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    print(f"🎵 推理开始 (总计 {len(df)} 样本)...")
    baseline_ranks, kg_ranks = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
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

        final_scores = cos_sim_orig.clone() 
        top_k_indices = sorted_indices_baseline[:TOP_K]

        # 3. 知识聚合 (修正版 Eq 4)
        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            
            unique_tails = set()
            for r in VALID_RELATIONS:
                try:
                    pred = predict_target(model=kge_model, head=kg_entity_name, relation=r, triples_factory=training_factory)
                    tails = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
                    for t in tails:
                        if t.lower() != orig_class_name.lower() and t.lower() not in class_labels_set:
                            unique_tails.add(t)
                except: continue
            
            enriched_prompts = [f"{orig_class_name}, {t}" for t in unique_tails]
            
            if len(enriched_prompts) > 0:
                prompt_embs = clap_model.get_text_embeddings(enriched_prompts)
                prompt_embs = F.normalize(prompt_embs, dim=-1)
                cos_sim_prompts = torch.matmul(audio_embed, prompt_embs.T).squeeze(0)
                
                # 严格按照带温度的 LogSumExp
                s_c = cos_sim_orig[c_idx] * LOGIT_SCALE
                s_p_j = cos_sim_prompts * LOGIT_SCALE
                
                all_s = torch.cat([s_c.unsqueeze(0), s_p_j]) 
                # 加入 Log-Mean-Exp 修正，防止路径数量干扰
                s_tilde_c = (torch.logsumexp(all_s, dim=0) - np.log(len(all_s))) / LOGIT_SCALE
                
                final_scores[c_idx] = s_tilde_c

        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        kg_ranks.append(np.where(sorted_indices_kg == true_idx)[0][0] + 1)

    b_h1, b_h3, b_h5, b_mrr = compute_metrics(baseline_ranks)
    kg_h1, kg_h3, kg_h5, kg_mrr = compute_metrics(kg_ranks)

    print("\n" + "="*60)
    print(f"{'Metric':<12} | {'Baseline':<15} | {'iKnow-audio (+KG)':<15}")
    print("-" * 60)
    print(f"{'Hit@1':<12} | {b_h1:<15.2f} | {kg_h1:<15.2f}")
    print(f"{'Hit@3':<12} | {b_h3:<15.2f} | {kg_h3:<15.2f}")
    print(f"{'Hit@5':<12} | {b_h5:<15.2f} | {kg_h5:<15.2f}")
    print(f"{'MRR':<12} | {b_mrr:<15.2f} | {kg_mrr:<15.2f}")
    print("="*60)

if __name__ == "__main__":
    main()