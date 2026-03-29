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
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 🌟 绑定 GPU 2

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
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/fsd50k/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 组合方案核心参数
ALPHA = 0.6       # 固定的最优权重
TOP_K = 15        # 基准 Top-K 增强 (FSD类别多，需设大些)
TOP_M = 3         # 每跳关系检索候选
TOP_P = 5         # GRASP剪枝保留数
LOGIT_SCALE = 100.0

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

def get_kg_entity(fsd_class):
    return fsd_class.replace('_', ' ').replace(' and ', ' ').lower()

def to_tensor(emb):
    if isinstance(emb, np.ndarray): return torch.from_numpy(emb)
    return emb

def load_llm_prompts(file_path):
    prompt_map = {}
    if not os.path.exists(file_path): 
        print(f"⚠️ 找不到文件: {file_path}")
        return prompt_map
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, text in data.items():
            parts = key.split("||")
            if len(parts) == 3:
                prompt_map["||".join([p.strip().lower() for p in parts])] = text
    return prompt_map

# ==========================================
# 2. 推理主程序
# ==========================================
@torch.no_grad()
def main():
    print(f"🚀 启动 FSD50K 终极组合: 多跳 + 语义扩充 + 剪枝 (Alpha={ALPHA})")
    
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())
    TARGET_RELATIONS = ['belongs to class', 'has parent', 'is a type of']
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in AVAILABLE_RELATIONS]

    # 读取 FSD50K Vocabulary
    vocab_df = pd.read_csv(FSD_VOCAB, header=None)
    unique_categories = vocab_df[1].tolist()
    vocab_name_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ').replace(' and ', ', ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    # 类别文本特征
    text_embeds = to_tensor(clap_model.get_text_embeddings(clean_classes)).to(DEVICE).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

    eval_df = pd.read_csv(FSD_EVAL_CSV)
    
    baseline_ranks, lme_ranks, combined_ranks = [], [], []
    kge_cache = {}

    def cached_predict(head, rel):
        key = (head, rel)
        if key in kge_cache: return kge_cache[key]
        q = head if head in training_factory.entity_to_id else head.split(' ')[-1]
        if q not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=q, relation=rel, triples_factory=training_factory)
            tails = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
            kge_cache[key] = tails
            return tails
        except: return []

    print(f"🎵 推理中 (多标签评测)...")
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        fname = str(row['fname'])
        audio_path = os.path.join(FSD_EVAL_AUDIO, fname + ".wav")
        if not os.path.exists(audio_path): continue
        
        # 多标签解析
        true_labels = row['labels'].split(',')
        true_indices = [vocab_name_to_idx[lbl] for lbl in true_labels if lbl in vocab_name_to_idx]
        if not true_indices: continue
        
        # 音频特征
        audio_embed = to_tensor(clap_model.get_audio_embeddings([audio_path])).to(DEVICE).float()
        audio_embed = F.normalize(audio_embed, dim=-1)
        
        # 1. Baseline
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        # FSD50K特有：多标签取命中最高的一个名次
        baseline_ranks.append(min([np.where(sorted_indices == t_idx)[0][0] + 1 for t_idx in true_indices]))

        # 2. 增强逻辑
        score_lme = cos_sim_orig.clone()
        score_comb = cos_sim_orig.clone()
        top_k_indices = sorted_indices[:TOP_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_head = get_kg_entity(unique_categories[c_idx])
            
            multi_hop_candidates = [] 
            raw_tails = set()

            # 多跳检索
            for r1 in VALID_RELATIONS:
                tails1 = cached_predict(kg_head, r1)
                for t1 in tails1:
                    t1_l = t1.lower()
                    if t1_l != orig_class_name.lower() and t1_l not in class_labels_set and t1_l != kg_head.lower():
                        multi_hop_candidates.append((kg_head, r1, t1))
                        raw_tails.add(t1)
                    # 第二跳
                    for r2 in VALID_RELATIONS:
                        tails2 = cached_predict(t1, r2)
                        for t2 in tails2:
                            t2_l = t2.lower()
                            if t2_l != orig_class_name.lower() and t2_l not in class_labels_set and t2_l != kg_head.lower():
                                multi_hop_candidates.append((t1, r2, t2))
                                raw_tails.add(t2)

            if raw_tails:
                # GRASP 剪枝
                tail_list = list(raw_tails)
                tail_embs = to_tensor(clap_model.get_text_embeddings(tail_list)).to(DEVICE).float()
                tail_embs = F.normalize(tail_embs, dim=-1)
                grasp_scores = torch.matmul(audio_embed, tail_embs.T).squeeze()
                if grasp_scores.dim() == 0: grasp_scores = grasp_scores.unsqueeze(0)
                
                _, top_p_idx = torch.topk(grasp_scores, min(TOP_P, len(tail_list)))
                pruned_tails_set = set([tail_list[i].lower() for i in top_p_idx.cpu().numpy()])

                # LLM 映射生成最终 prompt
                final_prompts = []
                for sub, rel, obj in multi_hop_candidates:
                    if obj.lower() in pruned_tails_set:
                        lk = f"{sub.lower()}||{rel.lower()}||{obj.lower()}"
                        final_prompts.append(prompt_map.get(lk, f"{orig_class_name}, {obj}"))
                
                if final_prompts:
                    final_prompts = list(set(final_prompts))
                    p_embs = to_tensor(clap_model.get_text_embeddings(final_prompts)).to(DEVICE).float()
                    p_embs = F.normalize(p_embs, dim=-1)
                    p_scores = torch.matmul(audio_embed, p_embs.T).squeeze()
                    if p_scores.dim() == 0: p_scores = p_scores.unsqueeze(0)
                    
                    # 轨道 A: LME 对齐
                    s_c_logit = cos_sim_orig[c_idx] * LOGIT_SCALE
                    s_p_logits = p_scores * LOGIT_SCALE
                    all_logits = torch.cat([s_c_logit.unsqueeze(0), s_p_logits])
                    score_lme[c_idx] = (torch.logsumexp(all_logits, dim=0) - np.log(len(all_logits))) / LOGIT_SCALE

                    # 轨道 B: Final Combination (Alpha=0.6)
                    best_p_score = torch.max(p_scores)
                    score_comb[c_idx] = (ALPHA * cos_sim_orig[c_idx]) + ((1.0 - ALPHA) * best_p_score)

        # 结算 FSD50K 多标签名次
        ranks_lme = torch.argsort(score_lme, descending=True).cpu().numpy()
        lme_ranks.append(min([np.where(ranks_lme == t_idx)[0][0] + 1 for t_idx in true_indices]))
        
        ranks_comb = torch.argsort(score_comb, descending=True).cpu().numpy()
        combined_ranks.append(min([np.where(ranks_comb == t_idx)[0][0] + 1 for t_idx in true_indices]))

    # 4. 打印对齐结果
    b_m = compute_metrics(baseline_ranks)
    l_m = compute_metrics(lme_ranks)
    c_m = compute_metrics(combined_ranks)
    
    print("\n" + "="*70)
    print(f"{'Metric':<10} | {'Baseline':<12} | {'LME':<12} | {'Final (Alpha=0.6)':<18}")
    print("-" * 70)
    for i, m in enumerate(['Hit@1', 'Hit@3', 'Hit@5', 'MRR']):
        print(f"{m:<10} | {b_m[i]:<12.2f} | {l_m[i]:<12.2f} | {c_m[i]:<18.2f}")
    print("="*70)

if __name__ == "__main__":
    main()