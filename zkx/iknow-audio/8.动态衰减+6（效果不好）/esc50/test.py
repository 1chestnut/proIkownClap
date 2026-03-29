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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 🌟 绑定 GPU

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
    try:
        patch_transformers_offline(cls_name)
    except AttributeError:
        continue

from msclap import CLAP
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

# ==========================================
# 1. 核心参数与多模态 RAG 设置 (ESC-50)
# ==========================================
ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv"
ESC50_AUDIO_DIR = "/home/star/zkx/CLAP/data/ESC-50/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/esc50/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 超参数
TOP_K = 5          # 基准类别截断
TOP_M = 3          # 每跳选取数量
TOP_P = 5          # GRASP 剪枝保留数
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
    if isinstance(emb, np.ndarray): 
        return torch.from_numpy(emb)
    return emb

def get_kg_entity(esc_class):
    clean_name = esc_class.replace('_', ' ').strip()
    if '(' in clean_name:
        clean_name = clean_name.split('(')[0].strip()
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
                sub, rel, obj = [p.strip().lower() for p in parts]
                prompt_map[f"{sub}||{rel}||{obj}"] = text
    return prompt_map

# ==========================================
# 2. 主程序：双动态机制 M3 融合
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 ESC-50 终极版: [双动态驱动: 动态α + 动态γ] + [多跳图谱+LLM] + [M3软池化]...")

    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    print(f"📚 成功加载 {len(prompt_map)} 条 LLM 增强提示词。")

    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    # 关系配置
    ALL_RELS = ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children']
    HOP1_RELATIONS = [r for r in ALL_RELS if r in training_factory.relation_to_id]
    HOP2_RELATIONS = [r for r in ['belongs to class', 'has parent', 'event composed of', 'has children']
                      if r in training_factory.relation_to_id]

    kge_cache = {}
    # 🌟 改进：现在返回图谱尾实体及其对应的预测分数 score
    def get_top_m_tails_with_scores(head, relation, m=TOP_M):
        cache_key = (head, relation)
        if cache_key in kge_cache:
            return kge_cache[cache_key]
        if head not in training_factory.entity_to_id:
            fallback = head.split(' ')[-1]
            if fallback not in training_factory.entity_to_id:
                return []
            head = fallback
        try:
            pred = predict_target(model=kge_model, head=head, relation=relation,
                                  triples_factory=training_factory)
            df_sorted = pred.df.sort_values(by="score", ascending=False).head(m)
            if len(df_sorted) == 0: return []
            
            # 返回 [(tail_label, score), ...]
            tails_with_scores = list(zip(df_sorted['tail_label'], df_sorted['score']))
            kge_cache[cache_key] = tails_with_scores
            return tails_with_scores
        except:
            return []

    # 加载数据
    df = pd.read_csv(ESC50_CSV)
    unique_categories = sorted(df['category'].unique())
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])

    text_embeds = to_tensor(clap_model.get_text_embeddings(clean_classes)).to(DEVICE).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

    # 排名记录
    ranks = {
        'Baseline': [],
        'M1_MaxAlpha': [],
        'M2_LME_iKnow': [],
        'M3_SoftAlpha_Decay': []
    }
    alpha_values = []  # 记录动态 α
    gamma_values = []  # 记录动态 γ

    print(f"🎵 推理开始 ({len(df)} 样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Dual-Dynamic M3+LLM"):
        audio_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
        if not os.path.exists(audio_path):
            continue
        true_idx = class_to_idx[row['category']]

        audio_embed = clap_model.get_audio_embeddings([audio_path])
        audio_embed = to_tensor(audio_embed).to(DEVICE).float()
        audio_embed = F.normalize(audio_embed, dim=-1)

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        ranks['Baseline'].append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        # ========== 🌟 动态 1: 动态 Alpha 计算 (依据 CLAP 相似度) ==========
        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim
        alpha_dynamic = max(ALPHA_MIN, min(ALPHA_MAX, alpha_dynamic))
        alpha_values.append(alpha_dynamic)
        # ====================================================================

        # 初始化得分
        score_m1 = cos_sim_orig.clone()
        score_m2 = cos_sim_orig.clone()
        score_m3 = cos_sim_orig.clone()

        top_k_indices = sorted_indices_baseline[:TOP_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx])

            # 记录合并信息: dict[tail_name, {'prompt': str, 'kge_score': float}]
            candidate_info = {}

            # 多跳检索 (提取 tails 和 KGE scores)
            for r1 in HOP1_RELATIONS:
                for t1, s1 in get_top_m_tails_with_scores(kg_entity_name, r1, TOP_M):
                    t1_clean = t1.lower().strip()
                    if t1_clean != orig_class_name.lower() and t1_clean not in class_labels_set:
                        lk1 = f"{kg_entity_name.lower()}||{r1.lower()}||{t1_clean}"
                        p1 = prompt_map.get(lk1, f"{orig_class_name}, {t1}")
                        candidate_info[t1_clean] = {'prompt': p1, 'is_hop2': False, 'kge_score': s1}

                    for r2 in HOP2_RELATIONS:
                        for t2, s2 in get_top_m_tails_with_scores(t1, r2, TOP_M):
                            t2_clean = t2.lower().strip()
                            if t2_clean != orig_class_name.lower() and t2_clean not in class_labels_set:
                                # 对于二跳，置信度取两条边的平均值
                                avg_path_score = (s1 + s2) / 2.0
                                # 竞争机制：多路径到达时，保留图谱预测分数更高的
                                if t2_clean not in candidate_info or candidate_info[t2_clean]['kge_score'] < avg_path_score:
                                    lk2 = f"{t1_clean}||{r2.lower()}||{t2_clean}"
                                    p2 = prompt_map.get(lk2, f"{orig_class_name}, {t2}")
                                    candidate_info[t2_clean] = {'prompt': p2, 'is_hop2': True, 'kge_score': avg_path_score}

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

                # 逐个编码提示词，避免长度不一致
                p_embs_list = []
                for p in prompts:
                    p_emb_raw = clap_model.get_text_embeddings([p])
                    p_embs_list.append(to_tensor(p_emb_raw).to(DEVICE).float())
                p_embs = torch.cat(p_embs_list, dim=0)
                p_embs = F.normalize(p_embs, dim=-1)

                raw_p_scores = torch.matmul(audio_embed, p_embs.T).squeeze()
                if raw_p_scores.dim() == 0:
                    raw_p_scores = raw_p_scores.unsqueeze(0)

                # 💡 剪枝层：应用动态映射的 Gamma 衰减系数进行惩罚过滤
                decayed_p_scores = raw_p_scores * gamma_tensor
                _, top_p_idx = torch.topk(decayed_p_scores, min(TOP_P, len(prompts)))
                
                final_p_scores = raw_p_scores[top_p_idx]
                final_decayed_scores = decayed_p_scores[top_p_idx]

                # M1: Max + 动态 Alpha
                best_p_val = torch.max(final_p_scores)
                score_m1[c_idx] = (alpha_dynamic * cos_sim_orig[c_idx]) + ((1.0 - alpha_dynamic) * best_p_val)

                # M2: 原始 LME (无 Alpha 锚定)
                c_logit = cos_sim_orig[c_idx] * LOGIT_SCALE
                p_logits = final_p_scores * LOGIT_SCALE
                all_logits_m2 = torch.cat([c_logit.unsqueeze(0), p_logits])
                score_m2[c_idx] = (torch.logsumexp(all_logits_m2, dim=0) - np.log(len(all_logits_m2))) / LOGIT_SCALE

                # M3: Soft + 动态 Decay + 动态 Alpha
                decayed_p_logits = final_decayed_scores * LOGIT_SCALE
                soft_prompt_score = (torch.logsumexp(decayed_p_logits, dim=0) - np.log(len(decayed_p_logits))) / LOGIT_SCALE
                score_m3[c_idx] = (alpha_dynamic * cos_sim_orig[c_idx]) + ((1.0 - alpha_dynamic) * soft_prompt_score)

        # 记录排名
        ranks['M1_MaxAlpha'].append(np.where(torch.argsort(score_m1, descending=True).cpu().numpy() == true_idx)[0][0] + 1)
        ranks['M2_LME_iKnow'].append(np.where(torch.argsort(score_m2, descending=True).cpu().numpy() == true_idx)[0][0] + 1)
        ranks['M3_SoftAlpha_Decay'].append(np.where(torch.argsort(score_m3, descending=True).cpu().numpy() == true_idx)[0][0] + 1)

    # ========== 结果输出 ==========
    print("\n" + "=" * 85)
    col_w = 15
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'M1(Max+Alpha)':<{col_w}} | {'M2(LME_iKnow)':<{col_w}} | {'双动态M3(Ours)':<{col_w}}"
    print(header)
    print("-" * 85)

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    b_res = compute_metrics(ranks['Baseline'])
    m1_res = compute_metrics(ranks['M1_MaxAlpha'])
    m2_res = compute_metrics(ranks['M2_LME_iKnow'])
    m3_res = compute_metrics(ranks['M3_SoftAlpha_Decay'])

    for i, m_name in enumerate(metrics):
        print(f"{m_name:<8} | {b_res[i]:<12.2f} | {m1_res[i]:<15.2f} | {m2_res[i]:<15.2f} | {m3_res[i]:<15.2f}")

    print("=" * 85)

    # 统计双动态
    avg_alpha = np.mean(alpha_values)
    avg_gamma = np.mean(gamma_values) if len(gamma_values) > 0 else 1.0
    print(f"\n📊 [动态 Alpha (跨模态置信度)] 统计: 平均 = {avg_alpha:.4f}, 最小 = {np.min(alpha_values):.4f}, 最大 = {np.max(alpha_values):.4f}")
    print(f"📊 [动态 Gamma (知识内置信度)] 统计: 平均 = {avg_gamma:.4f}, 最小 = {np.min(gamma_values):.4f}, 最大 = {np.max(gamma_values):.4f}")

    print("\n💡 终极机制解析 (Dual-Dynamic Driven - ESC-50):")
    print("1. 动态 Alpha：利用原始模型的高置信度锚定底座，防止模态遗忘。")
    print("2. 动态 Gamma：让更靠谱的多跳实体获得高权重通行权，用图谱自建置信度（KGE scores）完美取代硬衰减机制！")

if __name__ == "__main__":
    main()