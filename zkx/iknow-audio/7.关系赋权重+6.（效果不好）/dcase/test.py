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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 绑定 GPU

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
DECAY_GAMMA = 0.85   # 二跳距离衰减系数
LOGIT_SCALE = 100.0

# 动态 α 参数范围（仍用于融合，但不输出统计）
ALPHA_MIN = 0.4
ALPHA_MAX = 0.8

# ========== 关系权重配置 (基于 DCASE17 特点，可调) ==========
RELATION_WEIGHTS = {
    'indicates': 1.0,
    'described by': 0.9,
    'used for': 0.8,
    'associated with environment': 0.6,
    'has parent': 1.0,
    'is instance of': 1.0,
}
DEFAULT_REL_WEIGHT = 0.7

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
# 3. 主程序：多跳 + LLM + 动态α + 关系权重
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 DCASE17 实验: [多跳图谱] + [LLM] + [动态α] + [关系权重]...")

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
    def get_top_m_tails(head_entity, relation, m=TOP_M):
        cache_key = (head_entity, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        if head_entity not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=head_entity, relation=relation,
                                  triples_factory=training_factory)
            tails = pred.df.sort_values(by="score", ascending=False).head(m)['tail_label'].tolist()
            kge_cache[cache_key] = tails
            return tails
        except:
            return []

    clean_classes = DCASE_17_CLASSES
    class_to_idx = {cat: i for i, cat in enumerate(clean_classes)}
    class_labels_set = set(clean_classes)

    text_embeds = to_tensor(clap_model.get_text_embeddings(clean_classes)).to(DEVICE).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

    df = pd.read_csv(DCASE_CSV)

    baseline_ranks, m3_ranks = [], []

    print(f"🎵 推理开始 (总计 {len(df)} 个多标签样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="DCASE M3+LLM+RelWeight"):
        audio_path = os.path.join(DCASE_AUDIO_DIR, row['audio_filename'])
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            continue

        # 多标签解析
        raw_label_str = str(row['paper_formatted_labels']).lower().strip()
        raw_label_str = raw_label_str.replace('fire engine, fire truck (siren)', 'C_FIRE').replace('air horn, truck horn', 'C_AIR')

        true_indices = []
        for part in raw_label_str.split(','):
            part = part.strip()
            if part == 'C_FIRE':
                true_indices.append(class_to_idx['fire engine, fire truck (siren)'])
            elif part == 'C_AIR':
                true_indices.append(class_to_idx['air horn, truck horn'])
            elif part in class_to_idx:
                true_indices.append(class_to_idx[part])

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
        baseline_ranks.append(min([np.where(sorted_indices_baseline == t_idx)[0][0] + 1 for t_idx in true_indices]))

        # 动态 α
        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim
        alpha_dynamic = max(ALPHA_MIN, min(ALPHA_MAX, alpha_dynamic))

        final_scores = cos_sim_orig.clone()
        top_k_indices = sorted_indices_baseline[:TOP_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(orig_class_name)

            candidate_info = {}  # tail -> {prompt, is_hop2, rel_weight}

            # 多跳检索
            for r1 in HOP1_RELATIONS:
                w1 = RELATION_WEIGHTS.get(r1, DEFAULT_REL_WEIGHT)
                for t1 in get_top_m_tails(kg_entity_name, r1, TOP_M):
                    t1_c = t1.lower().strip()
                    if t1_c != orig_class_name.lower() and t1_c not in class_labels_set:
                        lk1 = f"{kg_entity_name.lower()}||{r1.lower()}||{t1_c}"
                        p1 = prompt_map.get(lk1, f"{orig_class_name}, {t1}")
                        candidate_info[t1_c] = {
                            'prompt': p1,
                            'is_hop2': False,
                            'rel_weight': w1
                        }

                    for r2 in HOP2_RELATIONS:
                        w2 = RELATION_WEIGHTS.get(r2, DEFAULT_REL_WEIGHT)
                        for t2 in get_top_m_tails(t1, r2, TOP_M):
                            t2_c = t2.lower().strip()
                            if t2_c != orig_class_name.lower() and t2_c not in class_labels_set:
                                if t2_c not in candidate_info:
                                    lk2 = f"{t1_c}||{r2.lower()}||{t2_c}"
                                    p2 = prompt_map.get(lk2, f"{orig_class_name}, {t2}")
                                    candidate_info[t2_c] = {
                                        'prompt': p2,
                                        'is_hop2': True,
                                        'rel_weight': w2
                                    }

            if len(candidate_info) > 0:
                prompts = []
                gammas = []
                rel_weights = []
                for info in candidate_info.values():
                    prompts.append(info['prompt'])
                    gammas.append(DECAY_GAMMA if info['is_hop2'] else 1.0)
                    rel_weights.append(info['rel_weight'])

                # 合并衰减系数和关系权重
                combined_weights = torch.tensor([g * rw for g, rw in zip(gammas, rel_weights)]).to(DEVICE)

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

                # 应用组合权重进行剪枝
                weighted_scores = raw_p_scores * combined_weights
                _, top_p_idx = torch.topk(weighted_scores, min(TOP_P, len(prompts)))
                final_raw_scores = raw_p_scores[top_p_idx]
                final_combined_weights = combined_weights[top_p_idx]

                # 最终聚合：使用加权后的相似度做 LogSumExp
                weighted_logits = final_raw_scores * final_combined_weights * LOGIT_SCALE
                soft_prompt_score = (torch.logsumexp(weighted_logits, dim=0) -
                                      np.log(len(weighted_logits))) / LOGIT_SCALE
                final_scores[c_idx] = (alpha_dynamic * cos_sim_orig[c_idx]) + \
                                      ((1.0 - alpha_dynamic) * soft_prompt_score)

        # 记录排名（多标签取最优）
        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        m3_ranks.append(min([np.where(sorted_indices_kg == t_idx)[0][0] + 1 for t_idx in true_indices]))

    # ========== 输出结果（仅 Baseline 和 M3+关系权重） ==========
    b_res = compute_metrics(baseline_ranks)
    m3_res = compute_metrics(m3_ranks)

    col_w = 22
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'M3-RAG + LLM (关系权重)':<{col_w}}"

    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    for i, m_name in enumerate(metrics):
        row = f"{m_name:<8} | {b_res[i]:<{col_w}.2f} | {m3_res[i]:<{col_w}.2f}"
        print(row)
    print("=" * len(header))

    print("\n💡 实验说明：已为两跳检索均加入关系权重，权重依据 DCASE17 特性设定（可调）。")
    print("动态α用于自适应融合，关系权重可进一步抑制噪声，增强关键关系。")

if __name__ == "__main__":
    main()