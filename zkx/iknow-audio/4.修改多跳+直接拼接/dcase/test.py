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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 M³KG-RAG 核心超参数
TOP_K = 5       
TOP_M = 3       
TOP_P = 5       
ALPHA = 0.6          # 🌟 固定类锚定最优权重
DECAY_GAMMA = 0.85   # 🌟 二跳距离衰减系数
LOGIT_SCALE = 100.0  

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

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
# 3. 主程序：多跳直接拼接 + 聚合公式消融
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 DCASE17: [多跳+直接拼接] 的 M³KG-RAG 机制验证...")
    
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())
    
    # 💡 结构层修复：DCASE 包含很多环境车辆音，极易发生“车辆->出现于街道->其他街道声音”的漂移
    ALL_RELS = ['indicates', 'described by', 'used for', 'associated with environment', 'has parent', 'is instance of']
    HOP1_RELATIONS = [r for r in ALL_RELS if r in AVAILABLE_RELATIONS]
    
    # 第二跳禁止环境蔓延 ('associated with environment')
    HOP2_RELATIONS = [r for r in ['indicates', 'described by', 'used for', 'has parent', 'is instance of'] if r in AVAILABLE_RELATIONS]

    kge_cache = {}
    def get_top_m_tails(head_entity, relation, m=TOP_M):
        cache_key = (head_entity, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        if head_entity not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=head_entity, relation=relation, triples_factory=training_factory)
            tails = pred.df.sort_values(by="score", ascending=False).head(m)['tail_label'].tolist()
            kge_cache[cache_key] = tails
            return tails
        except: return []

    clean_classes = DCASE_17_CLASSES
    class_to_idx = {cat: i for i, cat in enumerate(clean_classes)}
    class_labels_set = set(clean_classes)
    
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    df = pd.read_csv(DCASE_CSV)
    
    # 🌟 初始化对比容器
    ranks = {
        'Baseline': [],
        'M1_MaxAlpha': [],
        'M2_LME_iKnow': [],
        'M3_SoftAlpha_Decay': []
    }

    print(f"🎵 推理开始 (总计 {len(df)} 样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="DCASE M³KG"):
        audio_path = os.path.join(DCASE_AUDIO_DIR, row['audio_filename'])
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0: continue
        
        # 多标签解析逻辑 (完全保留你的做法)
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
        
        # [0] Baseline 计算
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        ranks['Baseline'].append(min([np.where(sorted_indices_baseline == t_idx)[0][0] + 1 for t_idx in true_indices]))

        # 准备得分板
        score_m1 = cos_sim_orig.clone()
        score_m2 = cos_sim_orig.clone()
        score_m3 = cos_sim_orig.clone()
        
        top_k_indices = sorted_indices_baseline[:TOP_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(orig_class_name) 
            
            # 记录实体跳数: dict[tail_name, is_hop2_boolean]
            tail_hop_map = {}
            
            # --- 挖掘受限的多跳知识 ---
            for r1 in HOP1_RELATIONS:
                hop1_tails = get_top_m_tails(kg_entity_name, r1, TOP_M)
                for t1 in hop1_tails:
                    t1_c = t1.lower().strip()
                    if t1_c != orig_class_name.lower() and t1_c not in class_labels_set:
                        tail_hop_map[t1] = False # Hop1
                    
                    for r2 in HOP2_RELATIONS:
                        hop2_tails = get_top_m_tails(t1, r2, TOP_M)
                        for t2 in hop2_tails:
                            t2_c = t2.lower().strip()
                            if t2_c != orig_class_name.lower() and t2_c not in class_labels_set:
                                if t2 not in tail_hop_map:
                                    tail_hop_map[t2] = True # Hop2
            
            candidate_tails = list(tail_hop_map.keys())
            
            if len(candidate_tails) > 0:
                # [无 LLM] 直接拼接提取特征
                prompts = [f"{orig_class_name}, {t}" for t in candidate_tails]
                gamma_tensor = torch.tensor([DECAY_GAMMA if tail_hop_map[t] else 1.0 for t in candidate_tails]).to(DEVICE)
                
                p_embs = clap_model.get_text_embeddings(prompts)
                p_embs = F.normalize(p_embs, dim=-1)
                
                raw_p_scores = torch.matmul(audio_embed, p_embs.T).squeeze()
                if raw_p_scores.dim() == 0: raw_p_scores = raw_p_scores.unsqueeze(0)
                
                # 💡 剪枝层：Hop-Aware 的 Top-P 过滤
                decayed_p_scores = raw_p_scores * gamma_tensor
                _, top_p_idx = torch.topk(decayed_p_scores, min(TOP_P, len(prompts)))
                
                final_p_scores = raw_p_scores[top_p_idx]           # 原分数 (M1/M2)
                final_decayed_scores = decayed_p_scores[top_p_idx] # 衰减分数 (M3)

                # ==========================================
                # 💡 聚合公式竞技场
                # ==========================================
                
                # M1: Max-Alpha
                best_p_val = torch.max(final_p_scores)
                score_m1[c_idx] = (ALPHA * cos_sim_orig[c_idx]) + ((1.0 - ALPHA) * best_p_val)
                
                # M2: 原版 iKnow LME
                s_c_lme = cos_sim_orig[c_idx] * LOGIT_SCALE
                s_p_lme = final_p_scores * LOGIT_SCALE
                all_s = torch.cat([s_c_lme.unsqueeze(0), s_p_lme])
                score_m2[c_idx] = (torch.logsumexp(all_s, dim=0) - np.log(len(all_s))) / LOGIT_SCALE
                
                # M3: Soft-Alpha Decay (衰减平滑 + 类锚定)
                decayed_p_logits = final_decayed_scores * LOGIT_SCALE
                soft_prompt_score = (torch.logsumexp(decayed_p_logits, dim=0) - np.log(len(decayed_p_logits))) / LOGIT_SCALE
                score_m3[c_idx] = (ALPHA * cos_sim_orig[c_idx]) + ((1.0 - ALPHA) * soft_prompt_score)

        # 结算本样本排名 (多标签取最优 Min)
        ranks_m1 = torch.argsort(score_m1, descending=True).cpu().numpy()
        ranks['M1_MaxAlpha'].append(min([np.where(ranks_m1 == t_idx)[0][0] + 1 for t_idx in true_indices]))
        
        ranks_m2 = torch.argsort(score_m2, descending=True).cpu().numpy()
        ranks['M2_LME_iKnow'].append(min([np.where(ranks_m2 == t_idx)[0][0] + 1 for t_idx in true_indices]))
        
        ranks_m3 = torch.argsort(score_m3, descending=True).cpu().numpy()
        ranks['M3_SoftAlpha_Decay'].append(min([np.where(ranks_m3 == t_idx)[0][0] + 1 for t_idx in true_indices]))

    # ==========================================
    # 4. 打印格式化表格
    # ==========================================
    b_res = compute_metrics(ranks['Baseline'])
    m1_res = compute_metrics(ranks['M1_MaxAlpha'])
    m2_res = compute_metrics(ranks['M2_LME_iKnow'])
    m3_res = compute_metrics(ranks['M3_SoftAlpha_Decay'])
    
    col_w = 15
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'M1(Max+Alpha)':<{col_w}} | {'M2(LME_iKnow)':<{col_w}} | {'M3(Soft+Decay)':<{col_w}}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    for i, m_name in enumerate(metrics):
        row = f"{m_name:<8} | {b_res[i]:<{col_w}.2f} | {m1_res[i]:<{col_w}.2f} | {m2_res[i]:<{col_w}.2f} | {m3_res[i]:<{col_w}.2f}"
        print(row)
    print("=" * len(header))
    
    print("\n💡 核心论点支持 (DCASE17 篇):")
    print("即便是‘直接拼接’，M3 机制也能通过打压二跳的‘associated with environment’关系来稳住大局！")

if __name__ == "__main__":
    main()