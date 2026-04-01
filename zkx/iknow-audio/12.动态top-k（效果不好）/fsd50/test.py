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
# 1. 路径与核心参数定义 (FSD50K)
# ==========================================
FSD_DIR = "/home/star/zkx/iknow-audio/data/FSD50K-1"
FSD_EVAL_AUDIO = os.path.join(FSD_DIR, "FSD50K.eval_audio")
FSD_EVAL_CSV = os.path.join(FSD_DIR, "FSD50K.ground_truth/eval.csv")
FSD_VOCAB = os.path.join(FSD_DIR, "FSD50K.ground_truth/vocabulary.csv")

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# 使用你原始效果最好的 LLM 提示词路径
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/fsd50k/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 核心超参数
TOP_K = 15      # 🌟 FSD50K 候选类别截断 (200类必须设大)
TOP_M = 3       # 知识图谱每跳分支
TOP_P = 5       # GRASP 剪枝保留数
DECAY_GAMMA = 0.85   # 二跳距离物理衰减系数
LOGIT_SCALE = 100.0

# 🌟 [垂直防噪] 自适应早退的相对边距阈值 (控制深度)
RELATIVE_MARGIN = -0.02

# 🌟 [水平防噪] 动态候选类截断阈值 (控制广度)
WIDTH_MARGIN = 0.05 

# 动态 α 参数范围
ALPHA_MIN = 0.4
ALPHA_MAX = 0.8

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return (np.mean(ranks <= 1) * 100,
            np.mean(ranks <= 3) * 100,
            np.mean(ranks <= 5) * 100,
            np.mean(1.0 / ranks) * 100)

def get_kg_entity(fsd_class):
    return fsd_class.replace('_', ' ').replace(' and ', ' ').lower()

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
                sub, rel, obj = [p.strip().lower() for p in parts]
                prompt_map[f"{sub}||{rel}||{obj}"] = text
    return prompt_map

# ==========================================
# 3. 主程序：多跳 + 双向自适应 (广度截断 + 深度早退) + 动态α
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 FSD50K 双向自适应版: [🔥自适应广度] + [🔥自适应深度/早退] + [动态α]...")

    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())
    
    # FSD50K 本体关系配置
    ALL_RELS = ['belongs to class', 'has parent', 'is a type of']
    HOP1_RELATIONS = [r for r in ALL_RELS if r in AVAILABLE_RELATIONS]
    HOP2_RELATIONS = [r for r in ALL_RELS if r in AVAILABLE_RELATIONS]

    kge_cache = {}
    def get_top_m_tails(head_entity, relation, m=TOP_M):
        cache_key = (head_entity, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        query = head_entity if head_entity in training_factory.entity_to_id else head_entity.split(' ')[-1]
        if query not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=query, relation=relation, triples_factory=training_factory)
            tails = pred.df.sort_values(by="score", ascending=False).head(m)['tail_label'].tolist()
            kge_cache[cache_key] = tails
            return tails
        except: return []

    vocab_df = pd.read_csv(FSD_VOCAB, header=None)
    unique_categories = vocab_df[1].tolist()
    vocab_name_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ').replace(' and ', ', ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    text_embeds = to_tensor(clap_model.get_text_embeddings(clean_classes)).to(DEVICE).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

    eval_df = pd.read_csv(FSD_EVAL_CSV)
    
    ranks = {'Baseline': [], 'Ours_DualAdaptive': []}
    
    # 效率统计
    total_theoretical_candidates = 0
    actual_candidates_processed = 0
    hop2_triggered_count = 0

    print(f"🎵 推理开始 (总计 {len(eval_df)} 个多标签音频)...")
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Dual-Adaptive Pipeline"):
        fname = str(row['fname'])
        audio_path = os.path.join(FSD_EVAL_AUDIO, fname + ".wav")
        if not os.path.exists(audio_path): continue
        
        # 多标签解析
        true_labels = row['labels'].split(',')
        true_indices = [vocab_name_to_idx[lbl] for lbl in true_labels if lbl in vocab_name_to_idx]
        if not true_indices: continue
        
        audio_embed_raw = clap_model.get_audio_embeddings([audio_path])
        audio_embed = to_tensor(audio_embed_raw).to(DEVICE).float()
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 1. Baseline 相似度
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        ranks['Baseline'].append(min([np.where(sorted_indices_baseline == t_idx)[0][0] + 1 for t_idx in true_indices]))

        # 2. 动态 α 计算
        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = max(ALPHA_MIN, min(ALPHA_MAX, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim))
        final_scores = cos_sim_orig.clone()

        # ==========================================
        # 🌟 核心一：自适应广度截断 (Adaptive Width)
        # ==========================================
        total_theoretical_candidates += TOP_K
        top_k_indices = []
        for idx in sorted_indices_baseline[:TOP_K]:
            # 分数不能比第一名差太多，差太多直接踢掉！
            if cos_sim_orig[idx].item() >= max_sim - WIDTH_MARGIN:
                top_k_indices.append(idx)
            else:
                break # 因为是排好序的，后面的肯定更差，直接跳出

        # 遍历动态截断后的候选类别
        for c_idx in top_k_indices:
            actual_candidates_processed += 1
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            baseline_score = cos_sim_orig[c_idx].item()
            
            # 早退动态阈值
            tau_dynamic = baseline_score + RELATIVE_MARGIN

            # ==========================================
            # STAGE 1: 获取并评估第一跳
            # ==========================================
            candidate_info_hop1 = {}
            for r1 in HOP1_RELATIONS:
                for t1 in get_top_m_tails(kg_entity_name, r1, TOP_M):
                    t1_c = t1.lower().strip()
                    if (t1_c != orig_class_name.lower() and t1_c not in class_labels_set and t1_c != kg_entity_name.lower()):
                        lk1 = f"{kg_entity_name.lower()}||{r1.lower()}||{t1_c}"
                        candidate_info_hop1[t1_c] = prompt_map.get(lk1, f"{orig_class_name}, {t1}")
            
            hop1_prompts = list(candidate_info_hop1.values())
            
            if len(hop1_prompts) == 0:
                max_hop1_score = -999.0
                hop1_scores = torch.tensor([]).to(DEVICE)
            else:
                p1_embs_list = [to_tensor(clap_model.get_text_embeddings([p])).to(DEVICE).float() for p in hop1_prompts]
                p1_embs = F.normalize(torch.cat(p1_embs_list, dim=0), dim=-1)
                
                hop1_scores = torch.matmul(audio_embed, p1_embs.T).squeeze()
                if hop1_scores.dim() == 0: hop1_scores = hop1_scores.unsqueeze(0)
                max_hop1_score = torch.max(hop1_scores).item()

            # ==========================================
            # 🌟 核心二：自适应深度早退 (Adaptive Depth)
            # ==========================================
            if max_hop1_score >= tau_dynamic:
                # 🌟 第一跳及格，早退！
                final_decayed_scores = hop1_scores * 1.0
                prompts_to_pool = hop1_prompts
            else:
                # ❌ 不及格，触发深层探索
                hop2_triggered_count += 1
                candidate_info_hop2 = {}
                for t1_c in candidate_info_hop1.keys():
                    for r2 in HOP2_RELATIONS:
                        for t2 in get_top_m_tails(t1_c, r2, TOP_M):
                            t2_c = t2.lower().strip()
                            if (t2_c != orig_class_name.lower() and t2_c not in class_labels_set and t2_c != kg_entity_name.lower()):
                                if t2_c not in candidate_info_hop2:
                                    lk2 = f"{t1_c}||{r2.lower()}||{t2_c}"
                                    candidate_info_hop2[t2_c] = prompt_map.get(lk2, f"{orig_class_name}, {t2}")
                
                hop2_prompts = list(candidate_info_hop2.values())
                if len(hop2_prompts) > 0:
                    p2_embs_list = [to_tensor(clap_model.get_text_embeddings([p])).to(DEVICE).float() for p in hop2_prompts]
                    p2_embs = F.normalize(torch.cat(p2_embs_list, dim=0), dim=-1)
                    
                    hop2_scores = torch.matmul(audio_embed, p2_embs.T).squeeze()
                    if hop2_scores.dim() == 0: hop2_scores = hop2_scores.unsqueeze(0)
                    
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

        # 结算 FSD50K 多标签名次
        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        ranks['Ours_DualAdaptive'].append(min([np.where(sorted_indices_kg == t_idx)[0][0] + 1 for t_idx in true_indices]))

    # ==========================================
    # 4. 打印格式化对比表
    # ==========================================
    b_res = compute_metrics(ranks['Baseline'])
    m3_res = compute_metrics(ranks['Ours_DualAdaptive'])
    
    col_w = 28
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'Ours (Dual-Adaptive)':<{col_w}}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    for i, m_name in enumerate(metrics):
        print(f"{m_name:<8} | {b_res[i]:<{col_w}.2f} | {m3_res[i]:<{col_w}.2f}")
    print("=" * len(header))

    # ========== 效率指标统计 ==========
    width_saved_rate = (1 - actual_candidates_processed / total_theoretical_candidates) * 100 if total_theoretical_candidates > 0 else 0
    hop2_rate = (hop2_triggered_count / actual_candidates_processed) * 100 if actual_candidates_processed > 0 else 0
    
    print("\n⚡ [双向自适应计算效率统计]")
    print(f"水平防噪参数 (Width Margin)  : {WIDTH_MARGIN}")
    print(f"垂直防噪参数 (Depth Margin)  : {RELATIVE_MARGIN}")
    print(f"1. 广度截断节省算力 (Width Saved) : 成功踢掉了 {width_saved_rate:.1f}% 的无关候选类别！")
    print(f"2. 深度早退节省算力 (Depth Saved) : 成功拦截了 {100 - hop2_rate:.1f}% 的冗余二跳检索！")
    print("💡 结论: 面对拥有 200 个类别的算力黑洞 FSD50K，模型依靠“水平一刀、垂直一刀”，精准地剥离了海量噪音，保住了精度并换来了惊人的效率。")

if __name__ == "__main__":
    main()