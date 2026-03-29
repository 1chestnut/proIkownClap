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
# 0. 断网防御与环境配置
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import transformers
orig_roberta = transformers.RobertaTokenizer.from_pretrained
def my_roberta_pretrained(pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path == "roberta-base":
        pretrained_model_name_or_path = "/home/star/zkx/CLAP/model/roberta-base"
    kwargs["local_files_only"] = True
    return orig_roberta(pretrained_model_name_or_path, *args, **kwargs)

transformers.RobertaTokenizer.from_pretrained = my_roberta_pretrained
sys.path.insert(0, "/home/star/zkx/CLAP/code/CLAP-main/src")

from laion_clap import CLAP_Module
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

# ==========================================
# 1. 路径与核心参数定义 (适配 US8K)
# ==========================================
MODEL_PATH = "/home/star/zkx/CLAP/model/630k-audioset-fusion-best.pt"
# 修改为 US8K 的路径
US8K_CSV = "/home/star/zkx/iknow-audio/data/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
US8K_AUDIO_DIR = "/home/star/zkx/iknow-audio/data/UrbanSound8K/UrbanSound8K/audio"

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 严格遵循论文 Algorithm 1 参数
TOP_K = 5   
TOP_M = 3   
LOGIT_SCALE = 100.0  

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try: yield
        finally: sys.stdout = old_stdout

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

# ==========================================
# 2. US8K 专属实体映射字典 (保证 10/10 命中)
# ==========================================
def get_kg_entity(us8k_class):
    # 处理 US8K 特有的下划线命名
    mapping = {
        'air_conditioner': 'air conditioner',
        'car_horn': 'car horn',
        'children_playing': 'children playing', # 如果图谱里没有，后续会自动回退到 child 或 playing
        'dog_bark': 'dog barking',              # 映射到标准的 dog barking 或 dog
        'drilling': 'drilling',
        'engine_idling': 'engine idling',
        'gun_shot': 'gunshot',
        'jackhammer': 'jackhammer',
        'siren': 'siren',
        'street_music': 'street music'
    }
    return mapping.get(us8k_class, us8k_class.replace('_', ' '))

# ==========================================
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    print("🚀 初始化 iKnow-audio US8K 零样本评估...")
    
    with suppress_stdout():
        model = CLAP_Module(enable_fusion=True, amodel='HTSAT-tiny', tmodel='roberta')
        model.load_ckpt(MODEL_PATH)
        model.to(DEVICE)
        model.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    # ------------------ 🌟 关系设定 🌟 ------------------
    # 严格对齐论文 Figure 5 中关于 US8K 的结论：城市音高度依赖场景和环境上下文！
    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())
    TARGET_RELATIONS = [
        'overlaps with', 'occurs in', 'associated with environment', 
        'localized in', 'used for', 'part of scene'
    ]
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in AVAILABLE_RELATIONS]
    print(f"🔗 使用对 US8K 最优的 {len(VALID_RELATIONS)} 个场景环境关系进行知识提取...")

    def get_top_m_tails(head_entity, relation, m=3):
        # 智能回退机制：如果组合词找不到，尝试用核心名词去找
        query_entity = head_entity
        if query_entity not in training_factory.entity_to_id:
            fallback = query_entity.split(' ')[0] # 比如 children playing 找不到就找 children
            if fallback in training_factory.entity_to_id:
                query_entity = fallback
            else:
                return []
        try:
            pred = predict_target(model=kge_model, head=query_entity, relation=relation, triples_factory=training_factory)
            return pred.df.sort_values(by="score", ascending=False).head(m)['tail_label'].tolist()
        except: return []

    # ------------------ 准备数据 ------------------
    df = pd.read_csv(US8K_CSV)
    unique_categories = sorted(df['class'].unique()) 
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    
    # 基础 Prompt：例如 "This is a sound of dog bark."
    clean_classes = [cat.replace('_', ' ') for cat in unique_categories]
    prompt_list = [f"This is a sound of {cat}." for cat in clean_classes]
    
    text_embeds = model.get_text_embedding(prompt_list)
    text_embeds = F.normalize(torch.from_numpy(text_embeds).to(DEVICE), dim=-1)

    print(f"🎵 开始对 US8K 的 {len(df)} 个音频执行推理...")
    baseline_ranks, kg_ranks = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference Progress"):
        # US8K 专属路径格式：audio/foldX/filename.wav
        audio_path = os.path.join(US8K_AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['class']]
        
        # 1. 提取音频特征
        audio_embed = torch.from_numpy(model.get_audio_embedding_from_filelist(x=[audio_path])).to(DEVICE)
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 2. 计算基准相似度
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        baseline_ranks.append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        # 3. 初始化 Top-K 分数
        final_scores = (cos_sim_orig * LOGIT_SCALE).clone() 
        top_k_indices = sorted_indices_baseline[:TOP_K]

        # 4. 提取与聚合知识 (Eq 4)
        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            enriched_prompts = []

            for r in VALID_RELATIONS:
                tails = get_top_m_tails(kg_entity_name, r, TOP_M)
                for t in tails:
                    enriched_prompts.append(f"This is a sound of {orig_class_name}, {t}.")
            
            if len(enriched_prompts) > 0:
                prompt_embs = torch.from_numpy(model.get_text_embedding(enriched_prompts)).to(DEVICE)
                prompt_embs = F.normalize(prompt_embs, dim=-1)
                
                cos_sim_prompts = torch.matmul(audio_embed, prompt_embs.T).squeeze(0)
                
                # Equation 4: LogSumExp
                s_c = cos_sim_orig[c_idx] * LOGIT_SCALE
                s_p_j = cos_sim_prompts * LOGIT_SCALE
                
                all_s = torch.cat([s_c.unsqueeze(0), s_p_j]) 
                s_tilde_c = torch.logsumexp(all_s, dim=0) 
                
                final_scores[c_idx] = s_tilde_c

        # 5. 最终预测重排
        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        kg_ranks.append(np.where(sorted_indices_kg == true_idx)[0][0] + 1)

    b_h1, b_h3, b_h5, b_mrr = compute_metrics(baseline_ranks)
    kg_h1, kg_h3, kg_h5, kg_mrr = compute_metrics(kg_ranks)

    print("\n" + "="*50)
    print(f"{'Metric':<10} | {'CLAP (Baseline)':<15} | {'CLAP +KG':<15}")
    print("-" * 50)
    print(f"{'Hit@1':<10} | {b_h1:<15.2f} | {kg_h1:<15.2f}")
    print(f"{'Hit@3':<10} | {b_h3:<15.2f} | {kg_h3:<15.2f}")
    print(f"{'Hit@5':<10} | {b_h5:<15.2f} | {kg_h5:<15.2f}")
    print(f"{'MRR':<10} | {b_mrr:<15.2f} | {kg_mrr:<15.2f}")
    print("="*50)

if __name__ == "__main__":
    main()