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
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

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
# 1. 路径与核心参数定义
# ==========================================
TUT_DIR = "/home/star/zkx/iknow-audio/data/TUT2017/development/TUT-acoustic-scenes-2017-development"
TUT_META = os.path.join(TUT_DIR, "meta.txt")

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 多跳与剪枝核心参数
TOP_K = 5   
TOP_M = 3 
TOP_P = 5 
LOGIT_SCALE = 100.0 

# 🌟 待验证的权重梯度 (0.4 到 0.9)
ALPHA_LIST = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

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

# ==========================================
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    print(f"🚀 启动 iKnow-audio TUT2017 权重消融实验...")
    print(f"📊 正在验证范围: Baseline, LME, 以及 Alpha={ALPHA_LIST}")
    
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    VALID_RELATIONS = [r for r in ['is variant of', 'has parent', 'scene contains', 'event composed of', 'described by'] if r in training_factory.relation_to_id]
    
    kge_cache = {}
    def get_top_m_tails(head_entity, relation, m=TOP_M):
        cache_key = (head_entity, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        query_entity = head_entity
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
    
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # 容器初始化
    baseline_ranks = []
    lme_ranks = []
    alpha_ranks_dict = {a: [] for a in ALPHA_LIST}

    print(f"🎵 推理开始...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Weights Testing"):
        audio_path = os.path.join(TUT_DIR, row['audio_rel_path'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['class']]
        
        audio_embed = clap_model.get_audio_embeddings([audio_path])
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        baseline_ranks.append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        # 准备得分板
        score_lme = cos_sim_orig.clone() 
        score_alpha_map = {a: cos_sim_orig.clone() for a in ALPHA_LIST}
        
        top_k_indices = sorted_indices_baseline[:TOP_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_scene_name = get_kg_entity(unique_categories[c_idx]) 
            
            raw_multi_hop = set()
            for r1 in VALID_RELATIONS:
                hop1 = get_top_m_tails(kg_scene_name, r1, TOP_M)
                for t1 in hop1:
                    t1_c = t1.lower().strip()
                    if t1_c != orig_class_name.lower() and t1_c not in class_labels_set and t1_c != kg_scene_name.lower():
                        raw_multi_hop.add(t1)
                    for r2 in VALID_RELATIONS:
                        hop2 = get_top_m_tails(t1, r2, TOP_M)
                        for t2 in hop2:
                            t2_c = t2.lower().strip()
                            if t2_c != orig_class_name.lower() and t2_c not in class_labels_set and t2_c != kg_scene_name.lower():
                                raw_multi_hop.add(t2)
            
            multi_hop_list = list(raw_multi_hop)
            if len(multi_hop_list) > 0:
                # GRASP 剪枝
                t_embs = clap_model.get_text_embeddings(multi_hop_list)
                t_embs = F.normalize(t_embs, dim=-1)
                p_scores = torch.matmul(audio_embed, t_embs.T).squeeze()
                if p_scores.dim() == 0: p_scores = p_scores.unsqueeze(0)
                
                _, top_p_idx = torch.topk(p_scores, min(TOP_P, len(multi_hop_list)))
                pruned_tails = [multi_hop_list[i] for i in top_p_idx.cpu().numpy()]
            
                # 拼接增强描述
                enriched_prompts = [f"{orig_class_name}, {t}" for t in pruned_tails]
                p_embs = clap_model.get_text_embeddings(enriched_prompts)
                p_embs = F.normalize(p_embs, dim=-1)
                sim_prompts = torch.matmul(audio_embed, p_embs.T).squeeze()
                if sim_prompts.dim() == 0: sim_prompts = sim_prompts.unsqueeze(0)
                
                # 1. LME 计算
                s_c_lme = cos_sim_orig[c_idx] * LOGIT_SCALE
                s_p_lme = sim_prompts * LOGIT_SCALE
                all_s = torch.cat([s_c_lme.unsqueeze(0), s_p_lme]) 
                score_lme[c_idx] = (torch.logsumexp(all_s, dim=0) - np.log(len(all_s))) / LOGIT_SCALE
                
                # 2. 多轨道 Max 融合
                best_p = torch.max(sim_prompts)
                for a in ALPHA_LIST:
                    score_alpha_map[a][c_idx] = (a * cos_sim_orig[c_idx]) + ((1.0 - a) * best_p)

        # 结算本样本在各轨道下的名次
        lme_ranks.append(np.where(torch.argsort(score_lme, descending=True).cpu().numpy() == true_idx)[0][0] + 1)
        for a in ALPHA_LIST:
            r_alpha = torch.argsort(score_alpha_map[a], descending=True).cpu().numpy()
            alpha_ranks_dict[a].append(np.where(r_alpha == true_idx)[0][0] + 1)

    # ==========================================
    # 4. 打印最终对比表
    # ==========================================
    b_res = compute_metrics(baseline_ranks)
    l_res = compute_metrics(lme_ranks)
    
    col_w = 11
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'LME':<{col_w}}"
    for a in ALPHA_LIST:
        header += f" | {'M(' + str(a) + ')':<{col_w}}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    for i, m_name in enumerate(metrics):
        row = f"{m_name:<8} | {b_res[i]:<{col_w}.2f} | {l_res[i]:<{col_w}.2f}"
        for a in ALPHA_LIST:
            a_res = compute_metrics(alpha_ranks_dict[a])
            row += f" | {a_res[i]:<{col_w}.2f}"
        print(row)
    print("=" * len(header))

if __name__ == "__main__":
    main()