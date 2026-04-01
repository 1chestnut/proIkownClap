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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 🌟 绑定 GPU

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

# LLM 提示词路径
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/dcase17/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 超参数
TOP_K = 5
TOP_M = 3
TOP_P = 5
LOGIT_SCALE = 100.0

# 🌟 双动态驱动超参数 🌟
# 动态 Alpha (音频锚定置信度)
ALPHA_MIN = 0.4
ALPHA_MAX = 0.8

# 动态 Gamma (图谱预测置信度)
GAMMA_MIN = 0.70
GAMMA_MAX = 1.00

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
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 找不到提示词文件 {file_path}")
        return prompt_map
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
# 3. 主程序：双动态机制 M3 融合
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 DCASE17 终极版: [双动态驱动: 动态α + 动态γ] + [多跳图谱+LLM] + [M3软池化]...")

    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    print(f"📚 成功加载 {len(prompt_map)} 条 LLM 增强提示词。")

    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())

    # 关系配置
    ALL_RELS = ['indicates', 'described by', 'used for', 'associated with environment', 'has parent', 'is instance of']
    HOP1_RELATIONS = [r for r in ALL_RELS if r in AVAILABLE_RELATIONS]
    HOP2_RELATIONS = [r for r in ['indicates', 'described by', 'used for', 'has parent', 'is instance of']
                      if r in AVAILABLE_RELATIONS]

    kge_cache = {}
    # 🌟 改进：现在返回图谱尾实体及其对应的预测分数 score
    def get_top_m_tails_with_scores(head_entity, relation, m=TOP_M):
        cache_key = (head_entity, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        if head_entity not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=head_entity, relation=relation,
                                  triples_factory=training_factory)
            df_sorted = pred.df.sort_values(by="score", ascending=False).head(m)
            if len(df_sorted) == 0: return []
            
            # 返回 [(tail_label, score), ...]
            tails_with_scores = list(zip(df_sorted['tail_label'], df_sorted['score']))
            kge_cache[cache_key] = tails_with_scores
            return tails_with_scores
        except:
            return []

    clean_classes = DCASE_17_CLASSES
    class_to_idx = {cat: i for i, cat in enumerate(clean_classes)}
    class_labels_set = set(clean_classes)

    text_embeds = to_tensor(clap_model.get_text_embeddings(clean_classes)).to(DEVICE).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

    df = pd.read_csv(DCASE_CSV)

    # 容器
    ranks = {
        'Baseline': [],
        'M1_MaxAlpha': [],
        'M2_LME_iKnow': [],
        'M3_SoftAlpha_Decay': []
    }
    alpha_values = []  # 记录每个样本的动态 α
    gamma_values = []  # 记录每个样本计算出的动态 γ

    print(f"🎵 推理开始 (总计 {len(df)} 个多标签样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Dual-Dynamic DCASE"):
        audio_path = os.path.join(DCASE_AUDIO_DIR, row['audio_filename'])
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            continue

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
        if not true_indices:
            continue

        try:
            audio_embed = clap_model.get_audio_embeddings([audio_path])
        except:
            continue

        audio_embed = to_tensor(audio_embed).to(DEVICE).float()
        audio_embed = F.normalize(audio_embed, dim=-1)

        # Baseline 相似度
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        ranks['Baseline'].append(min([np.where(sorted_indices_baseline == t_idx)[0][0] + 1 for t_idx in true_indices]))

        # ========== 🌟 动态 1: 动态 Alpha 计算 (依据 CLAP 置信度) ==========
        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim
        alpha_dynamic = max(ALPHA_MIN, min(ALPHA_MAX, alpha_dynamic))
        alpha_values.append(alpha_dynamic)
        # ====================================================================

        # 得分板
        score_m1 = cos_sim_orig.clone()
        score_m2 = cos_sim_orig.clone()
        score_m3 = cos_sim_orig.clone()

        top_k_indices = sorted_indices_baseline[:TOP_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(orig_class_name)

            candidate_info = {}

            # 多跳检索 (提取 tails 和对应的 KGE scores)
            for r1 in HOP1_RELATIONS:
                for t1, s1 in get_top_m_tails_with_scores(kg_entity_name, r1, TOP_M):
                    t1_c = t1.lower().strip()
                    if t1_c != orig_class_name.lower() and t1_c not in class_labels_set:
                        lk1 = f"{kg_entity_name.lower()}||{r1.lower()}||{t1_c}"
                        p1 = prompt_map.get(lk1, f"{orig_class_name}, {t1}")
                        candidate_info[t1_c] = {'prompt': p1, 'is_hop2': False, 'kge_score': s1}

                    for r2 in HOP2_RELATIONS:
                        for t2, s2 in get_top_m_tails_with_scores(t1, r2, TOP_M):
                            t2_c = t2.lower().strip()
                            if t2_c != orig_class_name.lower() and t2_c not in class_labels_set:
                                # 对于二跳，置信度取两条边的平均值
                                avg_path_score = (s1 + s2) / 2.0
                                # 若实体有多种到达路径，保留图谱预测更靠谱(高分)的那条
                                if t2_c not in candidate_info or candidate_info[t2_c]['kge_score'] < avg_path_score:
                                    lk2 = f"{t1_c}||{r2.lower()}||{t2_c}"
                                    p2 = prompt_map.get(lk2, f"{orig_class_name}, {t2}")
                                    candidate_info[t2_c] = {'prompt': p2, 'is_hop2': True, 'kge_score': avg_path_score}

            if len(candidate_info) > 0:
                prompts = []
                raw_kge_scores = []
                for info in candidate_info.values():
                    prompts.append(info['prompt'])
                    raw_kge_scores.append(info['kge_score'])

                # ========== 🌟 动态 2: 动态 Gamma 计算 (依据图谱置信度) ==========
                raw_kge_scores = np.array(raw_kge_scores)
                min_s, max_s = raw_kge_scores.min(), raw_kge_scores.max()

                # 将图谱得分映射到 [GAMMA_MIN, GAMMA_MAX] 区间
                if max_s - min_s < 1e-5:
                    gammas = [1.0 for _ in raw_kge_scores] # 无法拉开差距时给满信任度
                else:
                    gammas = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * (raw_kge_scores - min_s) / (max_s - min_s)
                
                gamma_values.extend(gammas)
                gamma_tensor = torch.tensor(gammas).to(DEVICE).float()
                # ====================================================================

                # 逐个编码提示词
                p_embs_list = []
                for p in prompts:
                    p_emb_raw = clap_model.get_text_embeddings([p])
                    p_embs_list.append(to_tensor(p_emb_raw).to(DEVICE).float())
                p_embs = torch.cat(p_embs_list, dim=0)
                p_embs = F.normalize(p_embs, dim=-1)

                raw_p_scores = torch.matmul(audio_embed, p_embs.T).squeeze()
                if raw_p_scores.dim() == 0:
                    raw_p_scores = raw_p_scores.unsqueeze(0)

                # 剪枝层：乘上动态映射出的 Gamma
                decayed_p_scores = raw_p_scores * gamma_tensor
                _, top_p_idx = torch.topk(decayed_p_scores, min(TOP_P, len(prompts)))
                final_p_scores = raw_p_scores[top_p_idx]
                final_decayed_scores = decayed_p_scores[top_p_idx]

                # M1: Max + 动态 α
                best_p_val = torch.max(final_p_scores)
                score_m1[c_idx] = (alpha_dynamic * cos_sim_orig[c_idx]) + ((1.0 - alpha_dynamic) * best_p_val)

                # M2: 原始 LME (无 α)
                s_c_lme = cos_sim_orig[c_idx] * LOGIT_SCALE
                s_p_lme = final_p_scores * LOGIT_SCALE
                all_s = torch.cat([s_c_lme.unsqueeze(0), s_p_lme])
                score_m2[c_idx] = (torch.logsumexp(all_s, dim=0) - np.log(len(all_s))) / LOGIT_SCALE

                # M3: Soft + 动态 Decay + 动态 α
                decayed_p_logits = final_decayed_scores * LOGIT_SCALE
                soft_prompt_score = (torch.logsumexp(decayed_p_logits, dim=0) - np.log(len(decayed_p_logits))) / LOGIT_SCALE
                score_m3[c_idx] = (alpha_dynamic * cos_sim_orig[c_idx]) + ((1.0 - alpha_dynamic) * soft_prompt_score)

        # 记录排名（多标签取最优）
        

        ranks_m3 = torch.argsort(score_m3, descending=True).cpu().numpy()
        ranks['M3_SoftAlpha_Decay'].append(min([np.where(ranks_m3 == t_idx)[0][0] + 1 for t_idx in true_indices]))

    # ========== 输出结果 ==========
    b_res = compute_metrics(ranks['Baseline'])
    m3_res = compute_metrics(ranks['M3_SoftAlpha_Decay'])

    col_w = 15
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'M1(Max+Alpha)':<{col_w}} | {'M2(LME_iKnow)':<{col_w}} | {'双动态M3(Ours)':<{col_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    for i, m_name in enumerate(metrics):
        row = f"{m_name:<8} | {b_res[i]:<{col_w}.2f} |{m3_res[i]:<{col_w}.2f}"
        print(row)
    print("=" * len(header))

    # 统计双动态
    avg_alpha = np.mean(alpha_values)
    avg_gamma = np.mean(gamma_values) if len(gamma_values) > 0 else 1.0
    print(f"\n📊 [动态 Alpha (跨模态置信度)] 统计: 平均 = {avg_alpha:.4f}, 最小 = {np.min(alpha_values):.4f}, 最大 = {np.max(alpha_values):.4f}")
    print(f"📊 [动态 Gamma (知识内置信度)] 统计: 平均 = {avg_gamma:.4f}, 最小 = {np.min(gamma_values):.4f}, 最大 = {np.max(gamma_values):.4f}")

    print("\n💡 终极机制解析 (Dual-Dynamic Driven - DCASE17):")
    print("1. 动态 Alpha 解决『跨模态置信度』：音频听得越清晰，Alpha 越大，越相信原声；听得越模糊，越借重图谱推测。")
    print("2. 动态 Gamma 解决『知识内置信度』：通过 KGE 预测分数的归一化，赋予高确信度多跳知识更高的通行权。")
    print("3. 这是结合了『符号图谱结构』与『大模型自由语义』的最优解！")

if __name__ == "__main__":
    main()