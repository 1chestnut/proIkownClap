import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ==========================================
# 0. 环境与离线配置 (保持你的路径不变)
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

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
# 1. 核心参数定义 (新增 ALPHA_LIST)
# ==========================================
ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv"
ESC50_AUDIO_DIR = "/home/star/zkx/CLAP/data/ESC-50/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 实验超参数
TOP_K = 5 
TOP_M = 3 
TOP_P = 5 
ALPHA_LIST = [0.5, 0.6, 0.7] # 💡 重点测试这三个权重

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

def get_kg_entity(esc_class):
    clean_name = esc_class.replace('_', ' ').strip()
    if '(' in clean_name: clean_name = clean_name.split('(')[0].strip()
    return clean_name.replace(' - ', ' ')

# ==========================================
# 3. 主程序：权重消融实验版
# ==========================================
@torch.no_grad()
def main():
    print(f"🚀 启动权重敏感性分析: 验证 Alpha {ALPHA_LIST} ...")
    
    # 加载模型
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    TARGET_RELATIONS = ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children']
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in training_factory.relation_to_id]

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

    # 初始化各权重的排名记录器
    baseline_ranks = []
    alpha_ranks_dict = {a: [] for a in ALPHA_LIST}

    print(f"🎵 开始推理 ({len(df)} 样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Weights Search"):
        audio_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['category']]
        
        # 1. 音频特征
        audio_embed = clap_model.get_audio_embeddings([audio_path])
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 2. 基准得分 (Alpha=1.0)
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        baseline_ranks.append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        # 为每个 Alpha 复制一份原始得分向量
        score_vectors = {a: cos_sim_orig.clone() for a in ALPHA_LIST}
        top_k_indices = sorted_indices_baseline[:TOP_K]

        # 3. 多跳检索与剪枝
        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            
            raw_multi_hop_tails = set()
            for r1 in VALID_RELATIONS:
                hop1 = get_top_m_tails(kg_entity_name, r1, TOP_M)
                for t1 in hop1:
                    if t1.lower() != orig_class_name.lower() and t1.lower() not in class_labels_set:
                        raw_multi_hop_tails.add(t1)
                    for r2 in VALID_RELATIONS:
                        hop2 = get_top_m_tails(t1, r2, TOP_M)
                        for t2 in hop2:
                            if t2.lower() != orig_class_name.lower() and t2.lower() not in class_labels_set:
                                raw_multi_hop_tails.add(t2)
            
            multi_hop_list = list(raw_multi_hop_tails)
            
            # GRASP 剪枝与增强
            if len(multi_hop_list) > 0:
                t_embs = clap_model.get_text_embeddings(multi_hop_list)
                t_embs = F.normalize(t_embs, dim=-1)
                p_scores = torch.matmul(audio_embed, t_embs.T).squeeze()
                if p_scores.dim() == 0: p_scores = p_scores.unsqueeze(0)
                
                _, top_p_idx = torch.topk(p_scores, min(TOP_P, len(multi_hop_list)))
                pruned_tails = [multi_hop_list[i] for i in top_p_idx.cpu().numpy()]
                
                # 计算增强后的最佳分数
                prompts = [f"{orig_class_name}, {t}" for t in pruned_tails]
                p_embs = clap_model.get_text_embeddings(prompts)
                p_embs = F.normalize(p_embs, dim=-1)
                best_p_score = torch.max(torch.matmul(audio_embed, p_embs.T))

                # ==== 核心逻辑：应用不同的 Alpha 权重 ====
                for alpha in ALPHA_LIST:
                    score_vectors[alpha][c_idx] = (alpha * cos_sim_orig[c_idx]) + ((1.0 - alpha) * best_p_score)

        # 记录各个 Alpha 下的排名
        for alpha in ALPHA_LIST:
            r = torch.argsort(score_vectors[alpha], descending=True).cpu().numpy()
            alpha_ranks_dict[alpha].append(np.where(r == true_idx)[0][0] + 1)

    # ==========================================
    # 4. 结果展示
    # ==========================================
    b_res = compute_metrics(baseline_ranks)
    print("\n" + "="*100)
    header = f"{'Metric':<10} | {'Baseline':<12}"
    for a in ALPHA_LIST: header += f" | {'Alpha=' + str(a):<12}"
    print(header)
    print("-" * 100)

    for i, m_name in enumerate(['Hit@1', 'Hit@3', 'Hit@5', 'MRR']):
        row_str = f"{m_name:<10} | {b_res[i]:<12.2f}"
        for a in ALPHA_LIST:
            a_res = compute_metrics(alpha_ranks_dict[a])
            row_str += f" | {a_res[i]:<12.2f}"
        print(row_str)
    print("="*100)

if __name__ == "__main__":
    main()