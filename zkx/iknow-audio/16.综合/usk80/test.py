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
# 1. 核心参数与多模态 RAG 设置 (US8K)
# ==========================================
US8K_CSV = "/home/star/zkx/iknow-audio/data/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
US8K_AUDIO_DIR = "/home/star/zkx/iknow-audio/data/UrbanSound8K/UrbanSound8K/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/us8k/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 黄金胜利参数组合
TOP_K = 5       
TOP_M = 3       
TOP_P = 5       # 🌟 GRASP 剪枝保留数，防止 N 惩罚
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

def get_kg_entity(us8k_class):
    mapping = {
        'air_conditioner': 'air conditioner', 'car_horn': 'car horn',
        'children_playing': 'children playing', 'dog_bark': 'dog barking', 
        'drilling': 'drilling', 'engine_idling': 'engine idling',
        'gun_shot': 'gunshot', 'jackhammer': 'jackhammer',
        'siren': 'siren', 'street_music': 'street music'
    }
    return mapping.get(us8k_class, us8k_class.replace('_', ' '))

def to_tensor(emb):
    if isinstance(emb, torch.Tensor): return emb
    return torch.from_numpy(emb)

def load_llm_prompts(file_path):
    prompt_map = {}
    if not os.path.exists(file_path): return prompt_map
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {"||".join([p.strip().lower() for p in k.split("||")]): v for k, v in data.items()}

def instance_aware_alpha(max_sim):
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim
    return max(ALPHA_MIN, min(ALPHA_MAX, alpha))

# 🌟 修复 msclap 长度不一致导致的 stack RuntimeError
def get_safe_text_embeddings(model, text_list, device):
    if not text_list:
        return torch.tensor([]).to(device)
    embs = []
    for t in text_list:
        e = model.get_text_embeddings([t])
        embs.append(to_tensor(e).to(device).float())
    return torch.cat(embs, dim=0)

# ==========================================
# 2. 主程序 (4-in-1 严格物理隔离版)
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 US8K 终极物理隔离消融实验 (Top-P 降噪修正 & 防崩溃版)...")
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    AVAILABLE_RELS = list(training_factory.relation_to_id.keys())
    HOP1_RELATIONS = [r for r in ['overlaps with', 'occurs in', 'associated with environment', 'localized in', 'used for', 'part of scene'] if r in AVAILABLE_RELS]
    HOP2_RELATIONS = [r for r in ['overlaps with', 'part of scene', 'used for'] if r in AVAILABLE_RELS]

    kge_cache = {}
    def get_tails(head, rel):
        key = (head, rel)
        if key in kge_cache: return kge_cache[key]
        h_query = head if head in training_factory.entity_to_id else head.split(' ')[0]
        if h_query not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=h_query, relation=rel, triples_factory=training_factory)
            res = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
            kge_cache[key] = res
            return res
        except: return []

    df = pd.read_csv(US8K_CSV)
    unique_categories = sorted(df['class'].unique()) 
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])

    text_embeds = F.normalize(get_safe_text_embeddings(clap_model, clean_classes, DEVICE), dim=-1)

    results = {m: {"ranks": [], "times": [], "prompts": [], "triggers": [], "alphas": []} for m in ["Baseline", "1-Hop", "All 2-Hop", "Selective"]}
    skipped_count = 0

    print(f"🎵 推理开始 (总计 {len(df)} 样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="US8K Evaluation"):
        audio_path = os.path.join(US8K_AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
        if not os.path.exists(audio_path):
            skipped_count += 1; continue
        
        true_idx = class_to_idx[row['class']]

        try:
            t0_start = time.time()
            audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = F.normalize(to_tensor(audio_emb_raw).to(DEVICE).float(), dim=-1)
        except:
            skipped_count += 1; continue

        # --- T0: Baseline ---
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = instance_aware_alpha(max_sim)
        t0_cost = time.time() - t0_start

        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        results["Baseline"]["ranks"].append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)
        results["Baseline"]["times"].append(t0_cost * 1000)
        results["Baseline"]["prompts"].append(1)

        top_k_indices = sorted_indices_baseline[:TOP_K]

        # =========================================================
        # Method 2: iKnow-audio 1-hop (严格纯净隔离版：无LLM, 无Alpha)
        # =========================================================
        t1_start = time.time()
        score_1hop = cos_sim_orig.clone()
        p_count_1hop = 0
        for c_idx in top_k_indices:
            orig_c = clean_classes[c_idx]
            kg_ent = get_kg_entity(unique_categories[c_idx])
            tails_h1 = set()
            for r in HOP1_RELATIONS:
                for t in get_tails(kg_ent, r):
                    t_c = t.lower().strip()
                    if t_c != orig_c.lower() and t_c not in class_labels_set: tails_h1.add(t)
            
            prompts_iknow = [f"{orig_c}, {t}" for t in tails_h1]
            if prompts_iknow:
                p_embs = F.normalize(get_safe_text_embeddings(clap_model, prompts_iknow, DEVICE), dim=-1)
                scs = torch.matmul(audio_embed, p_embs.T).squeeze()
                if scs.dim() == 0: scs = scs.unsqueeze(0)
                
                # 🌟 Top-P 降噪
                best_scs, _ = torch.topk(scs, min(TOP_P, len(scs)))
                logits = torch.cat([cos_sim_orig[c_idx].unsqueeze(0)*LOGIT_SCALE, best_scs*LOGIT_SCALE])
                score_1hop[c_idx] = (torch.logsumexp(logits, dim=0) - np.log(len(logits))) / LOGIT_SCALE
                p_count_1hop += len(prompts_iknow)
                
        results["1-Hop"]["ranks"].append(np.where(torch.argsort(score_1hop, descending=True).cpu().numpy() == true_idx)[0][0] + 1)
        results["1-Hop"]["times"].append((t0_cost + (time.time()-t1_start)) * 1000)
        results["1-Hop"]["prompts"].append(p_count_1hop)

        # =========================================================
        # Method 3: All 2-hop (全路径开启 + LLM + Top-P 降噪)
        # =========================================================
        t2_start = time.time()
        score_all2hop = cos_sim_orig.clone()
        p_count_all2hop = 0
        for c_idx in top_k_indices:
            orig_c = clean_classes[c_idx]
            kg_ent = get_kg_entity(unique_categories[c_idx])
            h1_map, h2_proms = {}, []
            
            for r1 in HOP1_RELATIONS:
                for t1 in get_tails(kg_ent, r1):
                    t1_l = t1.lower().strip()
                    if t1_l != orig_c.lower() and t1_l not in class_labels_set:
                        h1_map[t1_l] = prompt_map.get(f"{kg_ent.lower()}||{r1.lower()}||{t1_l}", f"{orig_c}, {t1}")
                        
            for t1_l in h1_map.keys():
                for r2 in HOP2_RELATIONS:
                    for t2 in get_tails(t1_l, r2):
                        t2_l = t2.lower().strip()
                        if t2_l != orig_c.lower() and t2_l not in class_labels_set and t2_l not in h1_map:
                            h2_proms.append(prompt_map.get(f"{t1_l}||{r2.lower()}||{t2_l}", f"{orig_c}, {t2}"))
            
            s1 = torch.tensor([]).to(DEVICE); s2 = torch.tensor([]).to(DEVICE)
            if h1_map:
                em1 = F.normalize(get_safe_text_embeddings(clap_model, list(h1_map.values()), DEVICE), dim=-1)
                s1 = torch.matmul(audio_embed, em1.T).squeeze()
                if s1.dim() == 0: s1 = s1.unsqueeze(0)
            if h2_proms:
                em2 = F.normalize(get_safe_text_embeddings(clap_model, h2_proms, DEVICE), dim=-1)
                s2 = torch.matmul(audio_embed, em2.T).squeeze()
                if s2.dim() == 0: s2 = s2.unsqueeze(0)
            
            all_scores = torch.cat([s1, s2 * DECAY_GAMMA]) if len(s2)>0 else s1
            if len(all_scores) > 0:
                best_scs, _ = torch.topk(all_scores, min(TOP_P, len(all_scores)))
                soft_s = (torch.logsumexp(best_scs*LOGIT_SCALE, dim=0) - np.log(len(best_scs))) / LOGIT_SCALE
                score_all2hop[c_idx] = (alpha_dynamic * cos_sim_orig[c_idx]) + ((1.0-alpha_dynamic) * soft_s)
            p_count_all2hop += (len(h1_map) + len(h2_proms))
            
        results["All 2-Hop"]["ranks"].append(np.where(torch.argsort(score_all2hop, descending=True).cpu().numpy() == true_idx)[0][0] + 1)
        results["All 2-Hop"]["times"].append((t0_cost + (time.time()-t2_start)) * 1000)
        results["All 2-Hop"]["prompts"].append(p_count_all2hop)

        # =========================================================
        # Method 4: Selective 2-hop (真正的物理加速早退版)
        # =========================================================
        t3_start = time.time()
        score_sel = cos_sim_orig.clone()
        p_count_sel = 0; triggered = False
        
        for c_idx in top_k_indices:
            orig_c = clean_classes[c_idx]
            kg_ent = get_kg_entity(unique_categories[c_idx])
            tau = cos_sim_orig[c_idx].item() + RELATIVE_MARGIN
            
            h1_map_s = {}
            for r in HOP1_RELATIONS:
                for t in get_tails(kg_ent, r):
                    t_l = t.lower().strip()
                    if t_l != orig_c.lower() and t_l not in class_labels_set:
                        h1_map_s[t_l] = prompt_map.get(f"{kg_ent.lower()}||{r.lower()}||{t_l}", f"{orig_c}, {t}")
            
            s1_s = torch.tensor([]).to(DEVICE)
            if h1_map_s:
                em1_s = F.normalize(get_safe_text_embeddings(clap_model, list(h1_map_s.values()), DEVICE), dim=-1)
                s1_s = torch.matmul(audio_embed, em1_s.T).squeeze()
                if s1_s.dim() == 0: s1_s = s1_s.unsqueeze(0)
            max_h1 = torch.max(s1_s).item() if len(s1_s)>0 else -999.0

            if max_h1 >= tau:
                # ✅ 物理跳过二跳
                if len(s1_s) > 0:
                    best_scs, _ = torch.topk(s1_s, min(TOP_P, len(s1_s)))
                    soft_s = (torch.logsumexp(best_scs*LOGIT_SCALE, dim=0) - np.log(len(best_scs))) / LOGIT_SCALE
                    score_sel[c_idx] = (alpha_dynamic * cos_sim_orig[c_idx]) + ((1.0-alpha_dynamic) * soft_s)
                p_count_sel += len(h1_map_s)
            else:
                # ❌ 物理触发二跳
                triggered = True; h2_proms_s = []
                for t1_l in h1_map_s.keys():
                    for r2 in HOP2_RELATIONS:
                        for t2 in get_tails(t1_l, r2):
                            t2_l = t2.lower().strip()
                            if t2_l != orig_c.lower() and t2_l not in class_labels_set and t2_l not in h1_map_s:
                                h2_proms_s.append(prompt_map.get(f"{t1_l}||{r2.lower()}||{t2_l}", f"{orig_c}, {t2}"))
                
                s2_s = torch.tensor([]).to(DEVICE)
                if h2_proms_s:
                    em2_s = F.normalize(get_safe_text_embeddings(clap_model, h2_proms_s, DEVICE), dim=-1)
                    s2_s = torch.matmul(audio_embed, em2_s.T).squeeze()
                    if s2_s.dim() == 0: s2_s = s2_s.unsqueeze(0)
                
                all_s_s = torch.cat([s1_s, s2_s * DECAY_GAMMA]) if len(s2_s)>0 else s1_s
                if len(all_s_s) > 0:
                    best_scs, _ = torch.topk(all_s_s, min(TOP_P, len(all_s_s)))
                    soft_s = (torch.logsumexp(best_scs*LOGIT_SCALE, dim=0) - np.log(len(best_scs))) / LOGIT_SCALE
                    score_sel[c_idx] = (alpha_dynamic * cos_sim_orig[c_idx]) + ((1.0-alpha_dynamic) * soft_s)
                p_count_sel += (len(h1_map_s) + len(h2_proms_s))
                
        results["Selective"]["ranks"].append(np.where(torch.argsort(score_sel, descending=True).cpu().numpy() == true_idx)[0][0] + 1)
        results["Selective"]["times"].append((t0_cost + (time.time()-t3_start)) * 1000)
        results["Selective"]["prompts"].append(p_count_sel)
        results["Selective"]["triggers"].append(triggered)
        results["Selective"]["alphas"].append(alpha_dynamic)

    if skipped_count > 0:
        print(f"\n⚠️ 提示: 共跳过了 {skipped_count} 个损坏的音频文件。")

    print_final_tables(results)

def print_final_tables(results):
    metrics = {m: compute_metrics(results[m]["ranks"]) for m in results}
    print("\n" + "="*85)
    print(f"{'Metric':<10} | {'Baseline':<12} | {'1-Hop (iKnow)':<15} | {'All 2-Hop (M3)':<15} | {'Selective (Ours)':<15}")
    print("-" * 85)
    for i, m_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(f"{m_name:<10} | {metrics['Baseline'][i]:<12.2f} | {metrics['1-Hop'][i]:<15.2f} | {metrics['All 2-Hop'][i]:<15.2f} | {metrics['Selective'][i]:<15.2f}")
    
    print("\n" + "="*95)
    print(f"{'Method':<25} | {'Trig %':<10} | {'Avg Prompts':<12} | {'Avg Time (ms)':<15} | {'Hit@1':<8}")
    print("-" * 95)
    for m in ["Baseline", "1-Hop", "All 2-Hop", "Selective"]:
        t_rate = f"{np.mean(results[m]['triggers'])*100:.1f}%" if m == "Selective" else "N/A"
        print(f"{m:<25} | {t_rate:<10} | {np.mean(results[m]['prompts']):<12.1f} | {np.mean(results[m]['times']):<15.1f} | {metrics[m][0]:<8.2f}")

if __name__ == "__main__":
    main()