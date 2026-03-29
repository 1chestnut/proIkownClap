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
# 0. 断网防御与最强离线拦截锁
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

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
# 1. 路径与核心参数定义 (FSD50K)
# ==========================================
FSD_DIR = "/home/star/zkx/iknow-audio/data/FSD50K-1"
FSD_EVAL_AUDIO = os.path.join(FSD_DIR, "FSD50K.eval_audio")
FSD_EVAL_CSV = os.path.join(FSD_DIR, "FSD50K.ground_truth/eval.csv")
FSD_VOCAB = os.path.join(FSD_DIR, "FSD50K.ground_truth/vocabulary.csv")

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# 🌟 FSD50K 自然语言字典路径
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/提示词扩展/fsd50k/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 全局锁定参数
TOP_K = 15   
TOP_M = 3   

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try: yield
        finally: sys.stdout = old_stdout

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

def get_kg_entity(fsd_class):
    clean_name = fsd_class.replace('_', ' ').replace(' and ', ' ').lower()
    return clean_name

def to_tensor(emb):
    if isinstance(emb, np.ndarray):
        return torch.from_numpy(emb)
    return emb

def load_llm_prompts(file_path):
    prompt_map = {}
    if not os.path.exists(file_path):
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
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 iKnow-audio FSD50K (统一创新版：自然语言 + Max Pooling + Alpha=0.85)...")
    
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    print(f"📚 成功加载 {len(prompt_map)} 条自然语言提示词。")
    
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())
    TARGET_RELATIONS = ['belongs to class', 'has parent', 'is a type of']
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in AVAILABLE_RELATIONS]

    vocab_df = pd.read_csv(FSD_VOCAB, header=None)
    unique_categories = vocab_df[1].tolist()
    vocab_name_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    
    clean_classes = [cat.replace('_', ' ').replace(' and ', ', ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    text_embeds_raw = clap_model.get_text_embeddings(clean_classes)
    text_embeds = to_tensor(text_embeds_raw).to(DEVICE).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

    eval_df = pd.read_csv(FSD_EVAL_CSV)
    print(f"🎵 推理开始 (总计 {len(eval_df)} 个多标签音频)...")
    baseline_ranks, kg_ranks = [], []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        fname = str(row['fname'])
        audio_path = os.path.join(FSD_EVAL_AUDIO, fname + ".wav")
        if not os.path.exists(audio_path): continue
        
        true_labels = row['labels'].split(',')
        true_indices = [vocab_name_to_idx[lbl] for lbl in true_labels if lbl in vocab_name_to_idx]
        if not true_indices: continue
        
        # 1. 音频特征
        audio_embed_raw = clap_model.get_audio_embeddings([audio_path])
        audio_embed = to_tensor(audio_embed_raw).to(DEVICE).float()
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 2. 基准得分与排名
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        b_ranks = [np.where(sorted_indices_baseline == t_idx)[0][0] + 1 for t_idx in true_indices]
        baseline_ranks.append(min(b_ranks))

        final_scores = cos_sim_orig.clone() 
        top_k_indices = sorted_indices_baseline[:TOP_K]

        # 3. 提取与融合知识
        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            
            unique_prompts = set()
            for r in VALID_RELATIONS:
                try:
                    query = kg_entity_name if kg_entity_name in training_factory.entity_to_id else kg_entity_name.split(' ')[-1]
                    pred = predict_target(model=kge_model, head=query, relation=r, triples_factory=training_factory)
                    tails = pred.df.sort_values(by="score", ascending=False).head(TOP_M)['tail_label'].tolist()
                    for t in tails:
                        t_clean = t.lower().strip()
                        if t_clean != orig_class_name.lower() and t_clean not in class_labels_set:
                            # 🌟 查表获取自然语言
                            lookup_key = f"{kg_entity_name.lower()}||{r.lower()}||{t_clean}"
                            if lookup_key in prompt_map:
                                unique_prompts.add(prompt_map[lookup_key])
                            else:
                                unique_prompts.add(f"{orig_class_name}, {t}")
                except: continue
            
            enriched_prompts = list(unique_prompts)
            
            # 🌟 统一核心逻辑：后融合
            if len(enriched_prompts) > 0:
                embs_list = []
                for prompt in enriched_prompts:
                    # 获取张量并转换类型，防止报错
                    single_emb_raw = clap_model.get_text_embeddings([prompt])
                    single_emb = to_tensor(single_emb_raw).to(DEVICE).float()
                    embs_list.append(single_emb)
                
                prompt_embs = torch.cat(embs_list, dim=0)
                prompt_embs = F.normalize(prompt_embs, dim=-1)
                
                # 计算与自然语言提示词的相似度
                prompt_scores = torch.matmul(audio_embed, prompt_embs.T).squeeze()
                
                # 🌟 统一策略 1：Max Pooling
                if prompt_scores.dim() == 0:
                    best_prompt_score = prompt_scores
                else:
                    best_prompt_score = torch.max(prompt_scores)
                
                # 🌟 统一策略 2：ALPHA = 0.85
                ALPHA = 0.85
                original_score = cos_sim_orig[c_idx]
                
                new_score = (ALPHA * original_score) + ((1.0 - ALPHA) * best_prompt_score)
                final_scores[c_idx] = new_score

        # 多标签 KG 排名
        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        k_ranks = [np.where(sorted_indices_kg == t_idx)[0][0] + 1 for t_idx in true_indices]
        kg_ranks.append(min(k_ranks))

    b_h1, b_h3, b_h5, b_mrr = compute_metrics(baseline_ranks)
    kg_h1, kg_h3, kg_h5, kg_mrr = compute_metrics(kg_ranks)

    print("\n" + "="*60)
    print(f"{'Metric':<12} | {'MS-CLAP (Baseline)':<18} | {'MS-CLAP +KG&NL':<15}")
    print("-" * 60)
    print(f"{'Hit@1':<12} | {b_h1:<18.2f} | {kg_h1:<15.2f}")
    print(f"{'Hit@3':<12} | {b_h3:<18.2f} | {kg_h3:<15.2f}")
    print(f"{'Hit@5':<12} | {b_h5:<18.2f} | {kg_h5:<15.2f}")
    print(f"{'MRR':<12} | {b_mrr:<18.2f} | {kg_mrr:<15.2f}")
    print("="*60)

if __name__ == "__main__":
    main()