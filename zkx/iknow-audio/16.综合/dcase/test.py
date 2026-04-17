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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
# 1. 核心参数与路径定义
# ==========================================
DCASE_CSV = "/home/star/zkx/iknow-audio/data/DCASE17-T4/my_evaluation_dataset.csv"
DCASE_AUDIO_DIR = "/home/star/zkx/iknow-audio/data/DCASE17-T4/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/dcase17/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TOP_K = 5
TOP_M = 3
TOP_P = 5      
DECAY_GAMMA = 0.85 
LOGIT_SCALE = 100.0
RELATIVE_MARGIN = -0.02
ALPHA_MIN = 0.4
ALPHA_MAX = 0.8

DCASE_17_CLASSES = [
    'ambulance (siren)', 'bicycle', 'bus', 'car', 'car alarm', 'car passing by',
    'civil defense siren', 'fire engine, fire truck (siren)', 'motorcycle',
    'police car (siren)', 'reversing beeps', 'screaming', 'skateboard',
    'train', 'train horn', 'truck', 'air horn, truck horn'
]

def to_tensor(emb):
    if isinstance(emb, torch.Tensor): return emb
    return torch.from_numpy(emb)

def compute_metrics(ranks):
    ranks = np.array(ranks)
    if len(ranks) == 0: return 0,0,0,0
    return (np.mean(ranks <= 1) * 100, np.mean(ranks <= 3) * 100,
            np.mean(ranks <= 5) * 100, np.mean(1.0 / ranks) * 100)

def get_kg_entity(class_name):
    mapping = {'ambulance (siren)': 'ambulance', 'fire engine, fire truck (siren)': 'fire engine',
               'police car (siren)': 'police car', 'air horn, truck horn': 'horn',
               'civil defense siren': 'siren', 'reversing beeps': 'beep', 'car passing by': 'car'}
    return mapping.get(class_name, class_name)

def load_llm_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {"||".join([p.strip().lower() for p in k.split("||")]): v for k, v in data.items()}

@torch.no_grad()
def main():
    print("🚀 启动 DCASE17 终极无污染物理隔离版...")
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    ALL_RELS = ['indicates', 'described by', 'used for', 'associated with environment', 'has parent', 'is instance of']
    VALID_RELS = [r for r in ALL_RELS if r in training_factory.relation_to_id]

    kge_cache = {}
    def get_tails(head, rel):
        key = (head, rel)
        if key in kge_cache: return kge_cache[key]
        if head not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=head, relation=rel, triples_factory=training_factory)
            res = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
            kge_cache[key] = res
            return res
        except: return []

    clean_classes = DCASE_17_CLASSES
    class_to_idx = {cat: i for i, cat in enumerate(clean_classes)}
    class_labels_set = set(clean_classes)
    text_embeds = F.normalize(to_tensor(clap_model.get_text_embeddings(clean_classes)).to(DEVICE).float(), dim=-1)

    df = pd.read_csv(DCASE_CSV)
    
    # 🌟 修复了 KeyError: 'alphas' 的初始化问题
    results = {
        "Baseline":  {"ranks": [], "times": [], "prompts": [], "triggers": []},
        "1-Hop":     {"ranks": [], "times": [], "prompts": [], "triggers": []},
        "All 2-Hop": {"ranks": [], "times": [], "prompts": [], "triggers": []},
        "Selective": {"ranks": [], "times": [], "prompts": [], "triggers": [], "alphas": []}
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluation"):
        audio_path = os.path.join(DCASE_AUDIO_DIR, row['audio_filename'])
        if not os.path.exists(audio_path): continue

        raw_label = str(row['paper_formatted_labels']).lower().strip().replace('fire engine, fire truck (siren)', 'C_FIRE').replace('air horn, truck horn', 'C_AIR')
        true_idx = [class_to_idx[p.strip().replace('C_FIRE', 'fire engine, fire truck (siren)').replace('C_AIR', 'air horn, truck horn')] for p in raw_label.split(',') if p.strip()]
        if not true_idx: continue

        try:
            t0_s = time.time()
            audio_embed = F.normalize(to_tensor(clap_model.get_audio_embeddings([audio_path])).to(DEVICE).float(), dim=-1)
        except: continue

        # ---------------------------------------------------------
        # Method 1: Baseline
        # ---------------------------------------------------------
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim
        alpha_dynamic = max(ALPHA_MIN, min(ALPHA_MAX, alpha_dynamic))
        t0_c = time.time() - t0_s
        
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        results["Baseline"]["ranks"].append(min([np.where(sorted_indices == t)[0][0] + 1 for t in true_idx]))
        results["Baseline"]["times"].append(t0_c * 1000)
        results["Baseline"]["prompts"].append(1)

        top_indices = sorted_indices[:TOP_K]

        # ---------------------------------------------------------
        # Method 2: iKnow-audio 1-Hop (独立隔离：纯逗号，无 Alpha)
        # ---------------------------------------------------------
        t1_s = time.time()
        score_1hop = cos_sim_orig.clone(); p1_cnt = 0
        for ci in top_indices:
            orig_c = clean_classes[ci]; kg_ent = get_kg_entity(orig_c)
            h1_tails = set()
            for r in VALID_RELS:
                for t in get_tails(kg_ent, r):
                    if t.lower() != orig_c.lower() and t not in class_labels_set: h1_tails.add(t)
            
            prompts_iknow = [f"{orig_c}, {t}" for t in h1_tails]
            if prompts_iknow:
                p_embs = F.normalize(to_tensor(clap_model.get_text_embeddings(prompts_iknow)).to(DEVICE).float(), dim=-1)
                scores = torch.matmul(audio_embed, p_embs.T).squeeze()
                if scores.dim() == 0: scores = scores.unsqueeze(0)
                
                # Top-P 切割
                best_sc, _ = torch.topk(scores, min(TOP_P, len(scores)))
                logits = torch.cat([cos_sim_orig[ci].unsqueeze(0)*LOGIT_SCALE, best_sc*LOGIT_SCALE])
                score_1hop[ci] = (torch.logsumexp(logits, dim=0) - np.log(len(logits))) / LOGIT_SCALE
                p1_cnt += len(prompts_iknow)
        
        results["1-Hop"]["ranks"].append(min([np.where(torch.argsort(score_1hop, descending=True).cpu().numpy() == t)[0][0] + 1 for t in true_idx]))
        results["1-Hop"]["times"].append((t0_c + (time.time()-t1_s))*1000); results["1-Hop"]["prompts"].append(p1_cnt)

        # ---------------------------------------------------------
        # Method 3: All 2-Hop (你原始的优秀 M3)
        # ---------------------------------------------------------
        t2_s = time.time()
        score_all2hop = cos_sim_orig.clone(); p2_cnt = 0
        for ci in top_indices:
            orig_c = clean_classes[ci]; kg_ent = get_kg_entity(orig_c)
            h1_map = {}; h2_prompts = []
            
            # Fetch 1-hop
            for r1 in VALID_RELS:
                for t1 in get_tails(kg_ent, r1):
                    t1_l = t1.lower().strip()
                    if t1_l != orig_c.lower() and t1_l not in class_labels_set:
                        h1_map[t1_l] = prompt_map.get(f"{kg_ent.lower()}||{r1.lower()}||{t1_l}", f"{orig_c}, {t1}")
            
            # Fetch 2-hop
            for t1_l in h1_map.keys():
                for r2 in VALID_RELS:
                    for t2 in get_tails(t1_l, r2):
                        if t2.lower() != orig_c.lower() and t2 not in class_labels_set and t2.lower() not in h1_map:
                            h2_prompts.append(prompt_map.get(f"{t1_l}||{r2.lower()}||{t2.lower()}", f"{orig_c}, {t2}"))
            
            s1 = torch.tensor([]).to(DEVICE); s2 = torch.tensor([]).to(DEVICE)
            if h1_map:
                em1 = F.normalize(to_tensor(clap_model.get_text_embeddings(list(h1_map.values()))).to(DEVICE).float(), dim=-1)
                s1 = torch.matmul(audio_embed, em1.T).squeeze()
                if s1.dim() == 0: s1 = s1.unsqueeze(0)
            if h2_prompts:
                em2 = F.normalize(to_tensor(clap_model.get_text_embeddings(h2_prompts)).to(DEVICE).float(), dim=-1)
                s2 = torch.matmul(audio_embed, em2.T).squeeze()
                if s2.dim() == 0: s2 = s2.unsqueeze(0)
            
            all_s = torch.cat([s1, s2 * DECAY_GAMMA]) if len(s2)>0 else s1
            if len(all_s) > 0:
                best_s, _ = torch.topk(all_s, min(TOP_P, len(all_s)))
                soft_s = (torch.logsumexp(best_s * LOGIT_SCALE, dim=0) - np.log(len(best_s))) / LOGIT_SCALE
                score_all2hop[ci] = (alpha_dynamic * cos_sim_orig[ci]) + ((1.0-alpha_dynamic) * soft_s)
            p2_cnt += (len(h1_map) + len(h2_prompts))
            
        results["All 2-Hop"]["ranks"].append(min([np.where(torch.argsort(score_all2hop, descending=True).cpu().numpy() == t)[0][0] + 1 for t in true_idx]))
        results["All 2-Hop"]["times"].append((t0_c + (time.time()-t2_s))*1000); results["All 2-Hop"]["prompts"].append(p2_cnt)

        # ---------------------------------------------------------
        # Method 4: Selective 2-Hop (物理早退逻辑)
        # ---------------------------------------------------------
        t3_s = time.time()
        score_sel = cos_sim_orig.clone(); p3_cnt = 0; triggered = False
        for ci in top_indices:
            orig_c = clean_classes[ci]; kg_ent = get_kg_entity(orig_c)
            tau = cos_sim_orig[ci].item() + RELATIVE_MARGIN
            
            # Fetch & Encode 1-hop only
            h1_map_s = {}
            for r in VALID_RELS:
                for t in get_tails(kg_ent, r):
                    t_l = t.lower().strip()
                    if t_l != orig_c.lower() and t_l not in class_labels_set:
                        h1_map_s[t_l] = prompt_map.get(f"{kg_ent.lower()}||{r.lower()}||{t_l}", f"{orig_c}, {t}")
            
            s1_s = torch.tensor([]).to(DEVICE)
            if h1_map_s:
                em1_s = F.normalize(to_tensor(clap_model.get_text_embeddings(list(h1_map_s.values()))).to(DEVICE).float(), dim=-1)
                s1_s = torch.matmul(audio_embed, em1_s.T).squeeze()
                if s1_s.dim() == 0: s1_s = s1_s.unsqueeze(0)
            max_h1 = torch.max(s1_s).item() if len(s1_s)>0 else -999.0

            if max_h1 >= tau:
                # 物理跳过二跳
                if len(s1_s) > 0:
                    best_s, _ = torch.topk(s1_s, min(TOP_P, len(s1_s)))
                    soft_s = (torch.logsumexp(best_s * LOGIT_SCALE, dim=0) - np.log(len(best_s))) / LOGIT_SCALE
                    score_sel[ci] = (alpha_dynamic * cos_sim_orig[ci]) + ((1.0-alpha_dynamic) * soft_s)
                p3_cnt += len(h1_map_s)
            else:
                # 触发二跳
                triggered = True; h2_proms_s = []
                for t1_l in h1_map_s.keys():
                    for r2 in VALID_RELS:
                        for t2 in get_tails(t1_l, r2):
                            if t2.lower() != orig_c.lower() and t2 not in class_labels_set and t2.lower() not in h1_map_s:
                                h2_proms_s.append(prompt_map.get(f"{t1_l}||{r2.lower()}||{t2.lower()}", f"{orig_c}, {t2}"))
                
                s2_s = torch.tensor([]).to(DEVICE)
                if h2_proms_s:
                    em2_s = F.normalize(to_tensor(clap_model.get_text_embeddings(h2_proms_s)).to(DEVICE).float(), dim=-1)
                    s2_s = torch.matmul(audio_embed, em2_s.T).squeeze()
                    if s2_s.dim() == 0: s2_s = s2_s.unsqueeze(0)
                
                all_s_s = torch.cat([s1_s, s2_s * DECAY_GAMMA]) if len(s2_s)>0 else s1_s
                if len(all_s_s) > 0:
                    best_s, _ = torch.topk(all_s_s, min(TOP_P, len(all_s_s)))
                    soft_s = (torch.logsumexp(best_s * LOGIT_SCALE, dim=0) - np.log(len(best_s))) / LOGIT_SCALE
                    score_sel[ci] = (alpha_dynamic * cos_sim_orig[ci]) + ((1.0-alpha_dynamic) * soft_s)
                p3_cnt += (len(h1_map_s) + len(h2_proms_s))

        results["Selective"]["ranks"].append(min([np.where(torch.argsort(score_sel, descending=True).cpu().numpy() == t)[0][0] + 1 for t in true_idx]))
        results["Selective"]["times"].append((t0_c + (time.time()-t3_s))*1000)
        results["Selective"]["prompts"].append(p3_cnt); results["Selective"]["triggers"].append(triggered)
        results["Selective"]["alphas"].append(alpha_dynamic)

    print_final_tables(results)

def print_final_tables(results):
    metrics = {m: compute_metrics(results[m]["ranks"]) for m in results}
    print("\n" + "="*85 + "\nTable 1: Main Performance\n" + "-"*85)
    print(f"{'Metric':<10} | {'Baseline':<12} | {'1-Hop':<12} | {'All 2-Hop':<12} | {'Selective':<12}")
    for i, m in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(f"{m:<10} | {metrics['Baseline'][i]:<12.2f} | {metrics['1-Hop'][i]:<12.2f} | {metrics['All 2-Hop'][i]:<12.2f} | {metrics['Selective'][i]:<12.2f}")
    print("\n" + "="*95 + "\nTable 2: Efficiency Analysis\n" + "-"*95)
    print(f"{'Method':<25} | {'Trig %':<10} | {'Avg Prompts':<12} | {'Avg Time (ms)':<15} | {'Hit@1':<8}")
    for m in results:
        t = f"{np.mean(results[m]['triggers'])*100:.1f}%" if m=="Selective" else "N/A"
        print(f"{m:<25} | {t:<10} | {np.mean(results[m]['prompts']):<12.1f} | {np.mean(results[m]['times']):<15.1f} | {metrics[m][0]:<8.2f}")

if __name__ == "__main__":
    main()