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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 绑定 GPU

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

# 使用你原始效果最好的 LLM 提示词路径
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/dcase17/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 核心超参数
TOP_K = 5
TOP_M = 3
TOP_P = 5
DECAY_GAMMA = 0.85   # 二跳距离物理衰减系数
LOGIT_SCALE = 100.0

# 🌟 自适应早退的相对边距阈值
RELATIVE_MARGIN = -0.02

# 🌟 KGE 软注入的温度系数 (控制置信度差距，1.0为原始比例)
KGE_TEMP = 1.0

# 动态 α 参数范围
ALPHA_MIN = 0.4
ALPHA_MAX = 0.8

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return (np.mean(ranks <= 1) * 100,
            np.mean(ranks <= 3) * 100,
            np.mean(ranks <= 5) * 100,
            np.mean(1.0 / ranks) * 100)

def to_tensor(emb):
    if isinstance(emb, np.ndarray): return torch.from_numpy(emb)
    return emb

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
# 3. 主程序：多跳 + 早退 + 动态α + 🔥KGE软注入
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 DCASE17 终极完全体: [多跳] + [早退] + [动态α] + [🔥KGE软注入]...")

    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())

    ALL_RELS = ['indicates', 'described by', 'used for', 'associated with environment', 'has parent', 'is instance of']
    HOP1_RELATIONS = [r for r in ALL_RELS if r in AVAILABLE_RELATIONS]
    HOP2_RELATIONS = [r for r in ['indicates', 'described by', 'used for', 'has parent', 'is instance of']
                      if r in AVAILABLE_RELATIONS]

    kge_cache = {}
    # 🌟 修改：同时返回 tail_label 和 图谱的预测得分 score
    def get_top_m_tails_with_scores(head_entity, relation, m=TOP_M):
        cache_key = (head_entity, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        if head_entity not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=head_entity, relation=relation, triples_factory=training_factory)
            top_df = pred.df.sort_values(by="score", ascending=False).head(m)
            tails = top_df['tail_label'].tolist()
            scores = top_df['score'].tolist()
            result = list(zip(tails, scores))
            kge_cache[cache_key] = result
            return result
        except: return []

    clean_classes = DCASE_17_CLASSES
    class_to_idx = {cat: i for i, cat in enumerate(clean_classes)}
    class_labels_set = set(clean_classes)

    text_embeds = to_tensor(clap_model.get_text_embeddings(clean_classes)).to(DEVICE).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

    df = pd.read_csv(DCASE_CSV)

    ranks = {'Baseline': [], 'Ours_UltimateM3': []}
    
    # 效率统计
    total_candidates_processed = 0
    hop2_triggered_count = 0

    print(f"🎵 推理开始 (总计 {len(df)} 个多标签样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ultimate M3 Pipeline"):
        audio_path = os.path.join(DCASE_AUDIO_DIR, row['audio_filename'])
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0: continue

        # 多标签解析
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

        try: audio_embed_raw = clap_model.get_audio_embeddings([audio_path])
        except: continue

        audio_embed = to_tensor(audio_embed_raw).to(DEVICE).float()
        audio_embed = F.normalize(audio_embed, dim=-1)

        # Baseline 相似度
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        ranks['Baseline'].append(min([np.where(sorted_indices_baseline == t_idx)[0][0] + 1 for t_idx in true_indices]))

        # 动态 α 计算
        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = max(ALPHA_MIN, min(ALPHA_MAX, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim))

        final_scores = cos_sim_orig.clone()
        top_k_indices = sorted_indices_baseline[:TOP_K]

        for c_idx in top_k_indices:
            total_candidates_processed += 1
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(orig_class_name)
            baseline_score = cos_sim_orig[c_idx].item()
            
            # 早退动态阈值
            tau_dynamic = baseline_score + RELATIVE_MARGIN

            # ==========================================
            # STAGE 1: 获取并评估第一跳 (🔥带KGE软注入)
            # ==========================================
            candidate_info_hop1 = {}
            for r1 in HOP1_RELATIONS:
                # 🌟 接收 KGE 置信度
                for t1, kge_score in get_top_m_tails_with_scores(kg_entity_name, r1, TOP_M):
                    t1_c = t1.lower().strip()
                    if t1_c != orig_class_name.lower() and t1_c not in class_labels_set:
                        lk1 = f"{kg_entity_name.lower()}||{r1.lower()}||{t1_c}"
                        candidate_info_hop1[t1_c] = {
                            'prompt': prompt_map.get(lk1, f"{orig_class_name}, {t1}"),
                            'kge_score': kge_score # 记录分数
                        }

            if len(candidate_info_hop1) == 0:
                max_hop1_score = -999.0
                hop1_scores = torch.tensor([]).to(DEVICE)
                hop1_prompts = []
            else:
                hop1_prompts = [info['prompt'] for info in candidate_info_hop1.values()]
                raw_kge_scores_1 = torch.tensor([info['kge_score'] for info in candidate_info_hop1.values()]).to(DEVICE).float()
                
                # 🔥 核心：计算 KGE 软注入权重 (Softmax 归一化)
                kge_weights_1 = F.softmax(raw_kge_scores_1 / KGE_TEMP, dim=0) * len(raw_kge_scores_1)

                p1_embs_list = [to_tensor(clap_model.get_text_embeddings([p])).to(DEVICE).float() for p in hop1_prompts]
                p1_embs = F.normalize(torch.cat(p1_embs_list, dim=0), dim=-1)
                
                raw_hop1_scores = torch.matmul(audio_embed, p1_embs.T).squeeze()
                if raw_hop1_scores.dim() == 0: raw_hop1_scores = raw_hop1_scores.unsqueeze(0)
                
                # 🔥 将 CLAP 相似度与图谱逻辑置信度融合！
                hop1_scores = raw_hop1_scores * kge_weights_1
                max_hop1_score = torch.max(hop1_scores).item()

            # ==========================================
            # STAGE 2: 阈值裁决 (Early Stopping)
            # ==========================================
            if max_hop1_score >= tau_dynamic:
                # 🌟 第一跳及格，早退
                final_decayed_scores = hop1_scores * 1.0
                prompts_to_pool = hop1_prompts
            else:
                # ❌ 不及格，触发深层探索
                hop2_triggered_count += 1
                candidate_info_hop2 = {}
                for t1_c in candidate_info_hop1.keys():
                    for r2 in HOP2_RELATIONS:
                        # 🌟 接收二跳的 KGE 置信度
                        for t2, kge_score in get_top_m_tails_with_scores(t1_c, r2, TOP_M):
                            t2_c = t2.lower().strip()
                            if t2_c != orig_class_name.lower() and t2_c not in class_labels_set:
                                if t2_c not in candidate_info_hop2:
                                    lk2 = f"{t1_c}||{r2.lower()}||{t2_c}"
                                    candidate_info_hop2[t2_c] = {
                                        'prompt': prompt_map.get(lk2, f"{orig_class_name}, {t2}"),
                                        'kge_score': kge_score
                                    }

                if len(candidate_info_hop2) > 0:
                    hop2_prompts = [info['prompt'] for info in candidate_info_hop2.values()]
                    raw_kge_scores_2 = torch.tensor([info['kge_score'] for info in candidate_info_hop2.values()]).to(DEVICE).float()
                    
                    # 🔥 核心：计算二跳的 KGE 软注入权重
                    kge_weights_2 = F.softmax(raw_kge_scores_2 / KGE_TEMP, dim=0) * len(raw_kge_scores_2)
                    
                    p2_embs_list = [to_tensor(clap_model.get_text_embeddings([p])).to(DEVICE).float() for p in hop2_prompts]
                    p2_embs = F.normalize(torch.cat(p2_embs_list, dim=0), dim=-1)
                    
                    raw_hop2_scores = torch.matmul(audio_embed, p2_embs.T).squeeze()
                    if raw_hop2_scores.dim() == 0: raw_hop2_scores = raw_hop2_scores.unsqueeze(0)
                    
                    # 🔥 将 CLAP 相似度与图谱逻辑置信度融合！
                    hop2_scores = raw_hop2_scores * kge_weights_2
                    
                    # 拼接：一跳乘 1.0，二跳乘 物理衰减 DECAY_GAMMA (0.85)
                    final_decayed_scores = torch.cat([hop1_scores * 1.0, hop2_scores * DECAY_GAMMA])
                    prompts_to_pool = hop1_prompts + hop2_prompts
                else:
                    final_decayed_scores = hop1_scores * 1.0
                    prompts_to_pool = hop1_prompts

            # ==========================================
            # STAGE 3: GRASP 剪枝与双动态池化
            # ==========================================
            if len(prompts_to_pool) > 0:
                _, top_p_idx = torch.topk(final_decayed_scores, min(TOP_P, len(prompts_to_pool)))
                best_p_scores = final_decayed_scores[top_p_idx]
                
                decayed_p_logits = best_p_scores * LOGIT_SCALE
                soft_prompt_score = (torch.logsumexp(decayed_p_logits, dim=0) - np.log(len(decayed_p_logits))) / LOGIT_SCALE
                final_scores[c_idx] = (alpha_dynamic * baseline_score) + ((1.0 - alpha_dynamic) * soft_prompt_score)

        # 结算多标签排名
        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        ranks['Ours_UltimateM3'].append(min([np.where(sorted_indices_kg == t_idx)[0][0] + 1 for t_idx in true_indices]))

    # ========== 结果输出 ==========
    b_res = compute_metrics(ranks['Baseline'])
    m3_res = compute_metrics(ranks['Ours_UltimateM3'])

    col_w = 26
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'Ours (Ultimate M3)':<{col_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    for i, m_name in enumerate(metrics):
        print(f"{m_name:<8} | {b_res[i]:<{col_w}.2f} | {m3_res[i]:<{col_w}.2f}")
    print("=" * len(header))

    # ========== 效率指标统计 ==========
    hop2_rate = (hop2_triggered_count / total_candidates_processed) * 100 if total_candidates_processed > 0 else 0
    print("\n⚡ [全架构护航系统统计]")
    print(f"动态阈值参数 (Margin)      : {RELATIVE_MARGIN}")
    print(f"KGE软注入平滑度 (Temp)     : {KGE_TEMP}")
    print(f"二跳检索触发率 (Trigger %) : {hop2_rate:.1f}%")
    print(f"深层计算力节省 (Saved %)   : {100 - hop2_rate:.1f}%")

if __name__ == "__main__":
    main()