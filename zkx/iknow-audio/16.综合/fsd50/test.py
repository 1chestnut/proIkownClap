import os
import sys
import time
import json
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 绑定 GPU

LOCAL_MODEL_DIR = "/home/star/zkx/iknow-audio/data/model"
CLAP_WEIGHTS_PATH = os.path.join(LOCAL_MODEL_DIR, "CLAP_weights_2023.pth")
GPT2_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "gpt2")
ROBERTA_LOCAL_PATH = "/home/star/zkx/CLAP/model/roberta-base"

import msclap.CLAPWrapper
def offline_hf_hub_download(*args, **kwargs):
    if not os.path.exists(CLAP_WEIGHTS_PATH):
        raise FileNotFoundError(f"未找到权重文件: {CLAP_WEIGHTS_PATH}")
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
# 1. 核心参数与路径定义 (FSD50K)
# ==========================================
FSD_DIR = "/home/star/zkx/iknow-audio/data/FSD50K-1"
FSD_EVAL_AUDIO = os.path.join(FSD_DIR, "FSD50K.eval_audio")
FSD_EVAL_CSV = os.path.join(FSD_DIR, "FSD50K.ground_truth/eval.csv")
FSD_VOCAB = os.path.join(FSD_DIR, "FSD50K.ground_truth/vocabulary.csv")

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/fsd50k/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# FSD50K 核心超参数
TOP_K = 15      # 候选类别截断 (200类必须大)
TOP_M = 3       
TOP_P = 5       # 用于 2-Hop 降噪剪枝
DECAY_GAMMA = 0.85   
LOGIT_SCALE = 100.0
RELATIVE_MARGIN = -0.02
ALPHA_MIN = 0.4
ALPHA_MAX = 0.8

def compute_metrics(ranks):
    ranks = np.array(ranks)
    if len(ranks) == 0: return 0,0,0,0
    return (np.mean(ranks <= 1) * 100, np.mean(ranks <= 3) * 100,
            np.mean(ranks <= 5) * 100, np.mean(1.0 / ranks) * 100)

def get_kg_entity(fsd_class):
    return fsd_class.replace('_', ' ').replace(' and ', ' ').lower()

def to_tensor(emb):
    if isinstance(emb, torch.Tensor): return emb
    return torch.from_numpy(emb)

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

def instance_aware_alpha(max_sim):
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim
    return max(ALPHA_MIN, min(ALPHA_MAX, alpha))

# 修复 msclap 长度不一致导致的 stack RuntimeError
def get_safe_text_embeddings(model, text_list, device):
    if not text_list: return torch.tensor([]).to(device)
    embs = []
    for t in text_list:
        e = model.get_text_embeddings([t])
        embs.append(to_tensor(e).to(device).float())
    return torch.cat(embs, dim=0)

# ==========================================
# 2. 主程序 (4-in-1 严格物理隔离+公式消融)
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 FSD50K 终极公式消融实验 (控制变量法 + 物理隔离)...")
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    AVAILABLE_RELS = list(training_factory.relation_to_id.keys())
    # FSD50K 纯向上分类树关系
    TARGET_RELS = ['belongs to class', 'has parent', 'is a type of']
    VALID_RELS = [r for r in TARGET_RELS if r in AVAILABLE_RELS]

    kge_cache = {}
    def get_tails(head, rel):
        key = (head, rel)
        if key in kge_cache: return kge_cache[key]
        h_query = head if head in training_factory.entity_to_id else head.split(' ')[-1]
        if h_query not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=h_query, relation=rel, triples_factory=training_factory)
            res = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
            kge_cache[key] = res
            return res
        except: return []

    vocab_df = pd.read_csv(FSD_VOCAB, header=None)
    unique_categories = vocab_df[1].tolist()
    vocab_name_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ').replace(' and ', ', ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])

    print("⏳ 正在预计算 200 个类别的文本特征...")
    text_embeds = F.normalize(get_safe_text_embeddings(clap_model, clean_classes, DEVICE), dim=-1)

    eval_df = pd.read_csv(FSD_EVAL_CSV)
    
    results = {
        "Baseline": {"ranks": [], "times": [], "prompts": []},
        "1-Hop_iKnow": {"ranks": [], "times": [], "prompts": []},
        "All_2Hop_iKnow_Eq": {"ranks": [], "times": [], "prompts": []}, # 消融 A (一锅炖)
        "All_2Hop_Ours_Eq": {"ranks": [], "times": [], "prompts": []},  # 消融 B (动态α)
        "Selective_Ours": {"ranks": [], "times": [], "prompts": [], "triggers": [], "alphas": []}
    }
    skipped_count = 0

    print(f"🎵 推理开始 (总计 {len(eval_df)} 个多标签音频)...")
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="FSD50K Pipeline"):
        fname = str(row['fname'])
        audio_path = os.path.join(FSD_EVAL_AUDIO, fname + ".wav")
        if not os.path.exists(audio_path):
            skipped_count += 1; continue
        
        true_labels = row['labels'].split(',')
        true_idx = [vocab_name_to_idx[lbl] for lbl in true_labels if lbl in vocab_name_to_idx]
        if not true_idx: continue

        try:
            t0_start = time.time()
            audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = F.normalize(to_tensor(audio_emb_raw).to(DEVICE).float(), dim=-1)
        except:
            skipped_count += 1; continue

        # --- 0. Baseline ---
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = instance_aware_alpha(max_sim)
        t0_cost = time.time() - t0_start

        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        results["Baseline"]["ranks"].append(min([np.where(sorted_indices_baseline == t)[0][0] + 1 for t in true_idx]))
        results["Baseline"]["times"].append(t0_cost * 1000)
        results["Baseline"]["prompts"].append(1)

        top_k_indices = sorted_indices_baseline[:TOP_K]

        # 初始化所有并行的 Score 板
        s_1hop = cos_sim_orig.clone()
        s_all2_iknow = cos_sim_orig.clone()
        s_all2_ours = cos_sim_orig.clone()
        s_selective = cos_sim_orig.clone()

        t1_start = time.time(); p_cnt_1hop = 0
        t2_start = time.time(); p_cnt_all2hop = 0
        t3_start = time.time(); p_cnt_selective = 0; triggered = False

        for c_idx in top_k_indices:
            orig_c = clean_classes[c_idx]
            kg_ent = get_kg_entity(unique_categories[c_idx])
            tau_dynamic = cos_sim_orig[c_idx].item() + RELATIVE_MARGIN

            # ==========================================
            # 模块 I: 严格原版 iKnow-audio 1-Hop (无剪枝, 纯逗号, 纯LME)
            # ==========================================
            tails_1h = set()
            for r in VALID_RELS:
                for t in get_tails(kg_ent, r):
                    t_c = t.lower().strip()
                    if t_c != orig_c.lower() and t_c not in class_labels_set: tails_1h.add(t)
            
            prompts_iknow = [f"{orig_c}, {t}" for t in tails_1h]
            if prompts_iknow:
                p_embs = F.normalize(get_safe_text_embeddings(clap_model, prompts_iknow, DEVICE), dim=-1)
                scs = torch.matmul(audio_embed, p_embs.T).squeeze()
                if scs.dim() == 0: scs = scs.unsqueeze(0)
                
                # 🌟 绝对不剪枝！直接全量 LME
                all_logits_iknow = torch.cat([cos_sim_orig[c_idx].unsqueeze(0)*LOGIT_SCALE, scs*LOGIT_SCALE])
                s_1hop[c_idx] = (torch.logsumexp(all_logits_iknow, dim=0) - np.log(len(all_logits_iknow))) / LOGIT_SCALE
                p_cnt_1hop += len(prompts_iknow)

            # ==========================================
            # 模块 II: 控制变量共享特征提取 (LLM + Hop2独享衰减 + Top-P 剪枝)
            # ==========================================
            h1_map, h2_proms = {}, []
            for r1 in VALID_RELS:
                for t1 in get_tails(kg_ent, r1):
                    t1_l = t1.lower().strip()
                    if t1_l != orig_c.lower() and t1_l not in class_labels_set:
                        h1_map[t1_l] = prompt_map.get(f"{kg_ent.lower()}||{r1.lower()}||{t1_l}", f"{orig_c}, {t1}")
                        
            for t1_l in h1_map.keys():
                for r2 in VALID_RELS:
                    for t2 in get_tails(t1_l, r2):
                        t2_l = t2.lower().strip()
                        if t2_l != orig_c.lower() and t2_l not in class_labels_set and t2_l not in h1_map:
                            h2_proms.append(prompt_map.get(f"{t1_l}||{r2.lower()}||{t2_l}", f"{orig_c}, {t2}"))
            
            sc1 = torch.tensor([]).to(DEVICE); sc2 = torch.tensor([]).to(DEVICE)
            if h1_map:
                sc1 = torch.matmul(audio_embed, F.normalize(get_safe_text_embeddings(clap_model, list(h1_map.values()), DEVICE), dim=-1).T).squeeze()
                if sc1.dim() == 0: sc1 = sc1.unsqueeze(0)
            if h2_proms:
                sc2 = torch.matmul(audio_embed, F.normalize(get_safe_text_embeddings(clap_model, h2_proms, DEVICE), dim=-1).T).squeeze()
                if sc2.dim() == 0: sc2 = sc2.unsqueeze(0)
            
            # 统一拼接，仅 Hop2 乘衰减系数
            all_s_pool = torch.cat([sc1, sc2 * DECAY_GAMMA]) if len(sc2) > 0 else sc1
            S_topP = torch.tensor([]).to(DEVICE)
            if len(all_s_pool) > 0:
                S_topP, _ = torch.topk(all_s_pool, min(TOP_P, len(all_s_pool)))
            p_cnt_all2hop += (len(h1_map) + len(h2_proms))

            # 🥊 控制变量 A：用 iKnow 原版一锅炖公式 (Base与TopP一起 LogSumExp)
            if len(S_topP) > 0:
                lme_inputs_A = torch.cat([cos_sim_orig[c_idx].unsqueeze(0)*LOGIT_SCALE, S_topP*LOGIT_SCALE])
                s_all2_iknow[c_idx] = (torch.logsumexp(lme_inputs_A, dim=0) - np.log(len(lme_inputs_A))) / LOGIT_SCALE
            
            # 🥊 控制变量 B：用 Ours 动态 α 公式 (软池化后 α 加权)
            if len(S_topP) > 0:
                soft_s_B = (torch.logsumexp(S_topP*LOGIT_SCALE, dim=0) - np.log(len(S_topP))) / LOGIT_SCALE
                s_all2_ours[c_idx] = (alpha_dynamic * cos_sim_orig[c_idx]) + ((1.0 - alpha_dynamic) * soft_s_B)

            # ==========================================
            # 模块 III: Selective (Ours 物理加速早退版)
            # ==========================================
            max_h1 = torch.max(sc1).item() if len(sc1)>0 else -999.0
            if max_h1 >= tau_dynamic:
                # ✅ 及格，物理跳过 Hop 2
                if len(sc1) > 0:
                    best_sc1, _ = torch.topk(sc1, min(TOP_P, len(sc1)))
                    soft_s_h1 = (torch.logsumexp(best_sc1*LOGIT_SCALE, dim=0) - np.log(len(best_sc1))) / LOGIT_SCALE
                    s_selective[c_idx] = (alpha_dynamic * cos_sim_orig[c_idx]) + ((1.0 - alpha_dynamic) * soft_s_h1)
                p_cnt_selective += len(h1_map)
            else:
                # ❌ 触发早退，直接借用实验 B 计算好的完整公式结果
                triggered = True
                s_selective[c_idx] = s_all2_ours[c_idx]
                p_cnt_selective += (len(h1_map) + len(h2_proms))

        # 结算 FSD50K 所有排名 (取多标签最优)
        results["1-Hop_iKnow"]["ranks"].append(min([np.where(torch.argsort(s_1hop, descending=True).cpu().numpy() == t)[0][0] + 1 for t in true_idx]))
        results["All_2Hop_iKnow_Eq"]["ranks"].append(min([np.where(torch.argsort(s_all2_iknow, descending=True).cpu().numpy() == t)[0][0] + 1 for t in true_idx]))
        results["All_2Hop_Ours_Eq"]["ranks"].append(min([np.where(torch.argsort(s_all2_ours, descending=True).cpu().numpy() == t)[0][0] + 1 for t in true_idx]))
        results["Selective_Ours"]["ranks"].append(min([np.where(torch.argsort(s_selective, descending=True).cpu().numpy() == t)[0][0] + 1 for t in true_idx]))

        # 结算真实耗时
        t_now = time.time()
        results["1-Hop_iKnow"]["times"].append((t0_cost + (t_now - t1_start))*1000); results["1-Hop_iKnow"]["prompts"].append(p_cnt_1hop)
        results["All_2Hop_iKnow_Eq"]["times"].append((t0_cost + (t_now - t2_start))*1000); results["All_2Hop_iKnow_Eq"]["prompts"].append(p_cnt_all2hop)
        
        # 物理加速时间的精准模拟
        if triggered:
            results["Selective_Ours"]["times"].append((t0_cost + (t_now - t2_start))*1000)
        else:
            results["Selective_Ours"]["times"].append((t0_cost + (t_now - t1_start))*1000) 
            
        results["Selective_Ours"]["prompts"].append(p_cnt_selective)
        results["Selective_Ours"]["triggers"].append(triggered)
        results["Selective_Ours"]["alphas"].append(alpha_dynamic)

    if skipped_count > 0: print(f"\n⚠️ 跳过了 {skipped_count} 个无效音频。")
    print_final_tables(results)

def print_final_tables(results):
    metrics = {m: compute_metrics(results[m]["ranks"]) for m in results}
    
    col_w = 20
    print("\n" + "="*110)
    print(f"{'Metric':<8} | {'Baseline':<10} | {'1-Hop (iKnow)':<15} | {'2-Hop (iKnow Eq)':<18} | {'2-Hop (Ours Eq)':<18} | {'Selective (Ours)':<18}")
    print("-" * 110)
    for i, m_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(f"{m_name:<8} | {metrics['Baseline'][i]:<10.2f} | {metrics['1-Hop_iKnow'][i]:<15.2f} | {metrics['All_2Hop_iKnow_Eq'][i]:<18.2f} | {metrics['All_2Hop_Ours_Eq'][i]:<18.2f} | {metrics['Selective_Ours'][i]:<18.2f}")
    
    print("\n" + "="*110)
    print(f"{'Method':<25} | {'Trig %':<8} | {'Avg Prompts':<12} | {'Avg Time (ms)':<15} | {'Hit@1':<8}")
    print("-" * 110)
    
    order = ["Baseline", "1-Hop_iKnow", "All_2Hop_iKnow_Eq", "All_2Hop_Ours_Eq", "Selective_Ours"]
    for m in order:
        t_rate = f"{np.mean(results[m]['triggers'])*100:.1f}%" if "triggers" in results[m] else "N/A"
        print(f"{m:<25} | {t_rate:<8} | {np.mean(results[m]['prompts']):<12.1f} | {np.mean(results[m]['times']):<15.1f} | {metrics[m][0]:<8.2f}")

if __name__ == "__main__":
    main()