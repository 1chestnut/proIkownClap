import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json

# ==========================================
# 0. 环境拦截与 GPU 绑定
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

LOCAL_MODEL_DIR = "/home/star/zkx/iknow-audio/data/model"
CLAP_WEIGHTS_PATH = os.path.join(LOCAL_MODEL_DIR, "CLAP_weights_2023.pth")
GPT2_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "gpt2")
ROBERTA_LOCAL_PATH = "/home/star/zkx/CLAP/model/roberta-base"

import msclap.CLAPWrapper
msclap.CLAPWrapper.hf_hub_download = lambda *args, **kwargs: CLAP_WEIGHTS_PATH

import transformers
def patch_transformers(cls_name):
    cls = getattr(transformers, cls_name)
    orig = cls.from_pretrained
    @classmethod
    def my_from(cls_inner, name_or_path, *args, **kwargs):
        target = GPT2_LOCAL_PATH if "gpt2" in str(name_or_path).lower() else ROBERTA_LOCAL_PATH
        kwargs["local_files_only"] = True
        return orig.__func__(cls_inner, target, *args, **kwargs)
    setattr(cls, "from_pretrained", my_from)

for name in ["AutoModel", "AutoConfig", "AutoTokenizer", "GPT2Tokenizer", "RobertaTokenizer"]:
    try: patch_transformers(name)
    except: continue

from msclap import CLAP
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

# ==========================================
# 1. 路径与核心参数 (FSD50K)
# ==========================================
FSD_CSV = "/home/star/zkx/iknow-audio/data/FSD50K/vocabulary_and_metadata/my_evaluation_dataset.csv"
FSD_AUDIO_DIR = "/home/star/zkx/iknow-audio/data/FSD50K/FSD50K.dev_test"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/fsd50k/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# FSD50K 类别多，适当扩大检索视野
TOP_K = 10 
TOP_M = 3
TOP_P = 5
DECAY_GAMMA = 0.85 
LOGIT_SCALE = 100.0

ALPHA_MIN = 0.4
ALPHA_MAX = 0.8

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return (np.mean(ranks <= 1) * 100, np.mean(ranks <= 3) * 100, 
            np.mean(ranks <= 5) * 100, np.mean(1.0 / ranks) * 100)

def to_tensor(emb):
    if isinstance(emb, np.ndarray): return torch.from_numpy(emb).to(DEVICE).float()
    return emb.to(DEVICE).float()

def load_llm_prompts(file_path):
    pm = {}
    if not os.path.exists(file_path): return pm
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for k, v in data.items():
            pm[k.lower().strip()] = v
    return pm

# ==========================================
# 2. FSD50K 专属实体清理
# ==========================================
def get_kg_entity(label):
    # FSD 标签通常是 "Accelerating_and_revving_and_vroom" 这种形式
    clean = label.replace('_', ' ').replace(',', ' ').replace('/', ' or ')
    return clean.lower().strip()

# ==========================================
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 FSD50K 熵驱动版: [全自动自适应早退] + [TOP_K=10]...")
    
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version='2023', use_cuda=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    # 加载 FSD 词表
    df = pd.read_csv(FSD_CSV)
    # 假设 CSV 中有 'labels' (多标签，逗号分隔) 和 'audio_filename'
    all_vocab = sorted(list(set([l.strip() for sublist in df['paper_formatted_labels'].str.split(',') for l in sublist])))
    class_to_idx = {c: i for i, c in enumerate(all_vocab)}
    clean_classes = [c.replace('_', ' ') for c in all_vocab]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    text_embeds = F.normalize(to_tensor(clap_model.get_text_embeddings(clean_classes)), dim=-1)

    # 关系定义
    AVAILABLE_RELS = list(training_factory.relation_to_id.keys())
    HOP1_RELS = [r for r in ['indicates', 'described by', 'used for', 'associated with environment', 'has parent'] if r in AVAILABLE_RELS]
    HOP2_RELS = [r for r in ['indicates', 'described by', 'used for', 'has parent'] if r in AVAILABLE_RELS]

    kge_cache = {}
    def get_tails(head, rel):
        ckey = (head, rel)
        if ckey in kge_cache: return ckey
        if head not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=head, relation=rel, triples_factory=training_factory)
            return pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
        except: return []

    ranks = {'Baseline': [], 'Ours_Entropy': []}
    recorded_margins = []
    hop2_count = 0
    total_cand = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="FSD-Entropy"):
        audio_path = os.path.join(FSD_AUDIO_DIR, str(row['audio_filename']) + ".wav")
        if not os.path.exists(audio_path): continue
        
        # 多标签索引处理
        true_labels = [l.strip() for l in str(row['paper_formatted_labels']).split(',')]
        true_indices = [class_to_idx[tl] for tl in true_labels if tl in class_to_idx]
        if not true_indices: continue
        
        audio_emb = F.normalize(to_tensor(clap_model.get_audio_embeddings([audio_path])), dim=-1)
        
        # 1. Baseline
        cos_sim = torch.matmul(audio_emb, text_embeds.T).squeeze(0)
        sorted_idx = torch.argsort(cos_sim, descending=True).cpu().numpy()
        # 多标签 Rank 取最高的那一个
        ranks['Baseline'].append(min([np.where(sorted_idx == ti)[0][0] for ti in true_indices]) + 1)

        # 2. 🔥 核心：计算熵并生成动态 Margin
        top_k_scores = cos_sim[sorted_idx[:TOP_K]]
        probs = F.softmax(top_k_scores * 10, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        
        norm_entropy = entropy / np.log(TOP_K)
        # FSD 类别多，熵通常高，映射系数设为 0.1 保持灵敏
        dynamic_margin = (norm_entropy - 0.5) * 0.1 
        recorded_margins.append(dynamic_margin)

        # 3. 动态 α 融合
        max_s = torch.max(cos_sim).item()
        alpha = max(ALPHA_MIN, min(ALPHA_MAX, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_s))
        final_scores = cos_sim.clone()

        for c_idx in sorted_idx[:TOP_K]:
            total_cand += 1
            orig_name = clean_classes[c_idx]
            kg_name = get_kg_entity(all_vocab[c_idx])
            b_score = cos_sim[c_idx].item()
            tau = b_score + dynamic_margin

            # --- Hop-1 ---
            h1_info = {}
            for r1 in HOP1_RELS:
                for t1 in get_tails(kg_name, r1):
                    t1_l = t1.lower().strip()
                    if t1_l != orig_name.lower() and t1_l not in class_labels_set:
                        lkey = f"{kg_name.lower()}||{r1.lower()}||{t1_l}"
                        h1_info[t1_l] = prompt_map.get(lkey, f"{orig_name}, {t1}")
            
            h1_prompts = list(h1_info.values())
            if not h1_prompts:
                max_h1 = -999.0
            else:
                p1_embs = F.normalize(to_tensor(clap_model.get_text_embeddings(h1_prompts)), dim=-1)
                h1_scores = torch.matmul(audio_emb, p1_embs.T).squeeze()
                if h1_scores.dim() == 0: h1_scores = h1_scores.unsqueeze(0)
                max_h1 = torch.max(h1_scores).item()

            # --- 熵驱动裁决 ---
            if max_h1 >= tau:
                p_scores = h1_scores
            else:
                hop2_count += 1
                h2_info = {}
                for t1_l in h1_info.keys():
                    for r2 in HOP2_RELS:
                        for t2 in get_tails(t1_l, r2):
                            t2_l = t2.lower().strip()
                            if t2_l != orig_name.lower() and t2_l not in class_labels_set:
                                if t2_l not in h2_info:
                                    lkey2 = f"{t1_l}||{r2.lower()}||{t2_l}"
                                    h2_info[t2_l] = prompt_map.get(lkey2, f"{orig_name}, {t2}")
                
                h2_prompts = list(h2_info.values())
                if h2_prompts:
                    p2_embs = F.normalize(to_tensor(clap_model.get_text_embeddings(h2_prompts)), dim=-1)
                    h2_scores = torch.matmul(audio_emb, p2_embs.T).squeeze()
                    if h2_scores.dim() == 0: h2_scores = h2_scores.unsqueeze(0)
                    p_scores = torch.cat([h1_scores, h2_scores * DECAY_GAMMA])
                else:
                    p_scores = h1_scores

            # --- LogSumExp ---
            if p_scores.numel() > 0:
                top_p_val, _ = torch.topk(p_scores, min(TOP_P, p_scores.numel()))
                soft_p = (torch.logsumexp(top_p_val * LOGIT_SCALE, dim=0) - np.log(len(top_p_val))) / LOGIT_SCALE
                final_scores[c_idx] = (alpha * b_score) + ((1.0 - alpha) * soft_p)

        sorted_idx_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        ranks['Ours_Entropy'].append(min([np.where(sorted_idx_kg == ti)[0][0] for ti in true_indices]) + 1)

    # 输出结果
    b_res = compute_metrics(ranks['Baseline'])
    m3_res = compute_metrics(ranks['Ours_Entropy'])
    print("\n" + "="*75)
    print(f"{'Metric':<10} | {'Baseline':<25} | {'Ours (FSD Entropy)':<25}")
    print("-" * 75)
    for i, m in enumerate(['Hit@1', 'Hit@3', 'Hit@5', 'MRR']):
        print(f"{m:<10} | {b_res[i]:<25.2f} | {m3_res[i]:<25.2f}")
    print("="*75)
    print(f"📊 平均动态 Margin: {np.mean(recorded_margins):.4f}")
    print(f"⚡ 二跳触发率: {(hop2_count/total_cand)*100:.1f}%")

if __name__ == "__main__":
    main()