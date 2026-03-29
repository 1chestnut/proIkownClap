import os
import sys
import contextlib
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
# 1. 路径与核心参数定义
# ==========================================
DCASE_CSV = "/home/star/zkx/iknow-audio/data/DCASE17-T4/my_evaluation_dataset.csv"
DCASE_AUDIO_DIR = "/home/star/zkx/iknow-audio/data/DCASE17-T4/audio"

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 黄金参数
TOP_K = 5   
TOP_M = 3              
LOGIT_SCALE = 100.0  

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

# ==========================================
# 2. DCASE 官方 17 类别白名单与映射
# ==========================================
# 🚨 DCASE17 官方的 17 个独立类别，绝不能被错误拆分
DCASE_17_CLASSES = [
    'ambulance (siren)', 'bicycle', 'bus', 'car', 'car alarm', 'car passing by', 
    'civil defense siren', 'fire engine, fire truck (siren)', 'motorcycle', 
    'police car (siren)', 'reversing beeps', 'screaming', 'skateboard', 
    'train', 'train horn', 'truck', 'air horn, truck horn'
]

def get_kg_entity(class_name):
    # 针对带括号和复杂名称的类别进行精准去噪，直指图谱核心词
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
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 iKnow-audio DCASE17-T4 (彻底修复 19类内耗 Bug 版)...")
    
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    # ------------------ 🌟 关系设定 🌟 ------------------
    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())
    TARGET_RELATIONS = [
        'indicates', 'described by', 'used for', 
        'associated with environment', 'has parent', 'is instance of'
    ]
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in AVAILABLE_RELATIONS]
    print(f"🔗 使用对 DCASE17 最优的 {len(VALID_RELATIONS)} 个核心关系...")

    def get_top_m_tails(head_entity, relation, m=3):
        if head_entity not in training_factory.entity_to_id:
            return []
        try:
            pred = predict_target(model=kge_model, head=head_entity, relation=relation, triples_factory=training_factory)
            return pred.df.sort_values(by="score", ascending=False).head(m)['tail_label'].tolist()
        except: return []

    # ------------------ 准备数据 ------------------
    df = pd.read_csv(DCASE_CSV)
    
    clean_classes = DCASE_17_CLASSES
    class_to_idx = {cat: i for i, cat in enumerate(clean_classes)}
    class_labels_set = set(clean_classes)
    
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    print(f"🎵 推理开始 (总计 {len(df)} 样本, 严格锁定 {len(clean_classes)} 个独有类别)...")
    baseline_ranks, kg_ranks = [], []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference Progress"):
        audio_path = os.path.join(DCASE_AUDIO_DIR, row['audio_filename'])
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0: 
            skipped += 1
            continue
        
        # 🌟 核心修复：白名单替换法解析多标签，彻底防止逗号错拆！
        raw_label_str = str(row['paper_formatted_labels']).lower().strip()
        # 提前把带逗号的类别换成安全占位符
        raw_label_str = raw_label_str.replace('fire engine, fire truck (siren)', 'C_FIRE')
        raw_label_str = raw_label_str.replace('air horn, truck horn', 'C_AIR')
        
        true_indices = []
        for part in raw_label_str.split(','):
            part = part.strip()
            if part == 'C_FIRE':
                true_indices.append(class_to_idx['fire engine, fire truck (siren)'])
            elif part == 'C_AIR':
                true_indices.append(class_to_idx['air horn, truck horn'])
            elif part in class_to_idx:
                true_indices.append(class_to_idx[part])
                
        # 去重并过滤空值
        true_indices = list(set(true_indices))
        if not true_indices: continue
        
        try:
            audio_embed = clap_model.get_audio_embeddings([audio_path])
        except:
            skipped += 1
            continue
            
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        
        ranks_baseline = [np.where(sorted_indices_baseline == t_idx)[0][0] + 1 for t_idx in true_indices]
        baseline_ranks.append(min(ranks_baseline))

        final_scores = cos_sim_orig.clone() 
        top_k_indices = sorted_indices_baseline[:TOP_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(orig_class_name) 
            
            unique_tails = set()
            for r in VALID_RELATIONS:
                tails = get_top_m_tails(kg_entity_name, r, TOP_M)
                for t in tails:
                    t_clean = t.lower().strip()
                    if t_clean == orig_class_name.lower() or t_clean == kg_entity_name.lower():
                        continue
                    if t_clean in class_labels_set:
                        continue
                    unique_tails.add(t)
            
            # 恢复最纯粹的逗号分隔
            enriched_prompts = [f"{orig_class_name}, {t}" for t in unique_tails]
            
            if len(enriched_prompts) > 0:
                prompt_embs = clap_model.get_text_embeddings(enriched_prompts)
                prompt_embs = F.normalize(prompt_embs, dim=-1)
                
                cos_sim_prompts = torch.matmul(audio_embed, prompt_embs.T).squeeze(0)
                
                s_c = cos_sim_orig[c_idx] * LOGIT_SCALE
                s_p_j = cos_sim_prompts * LOGIT_SCALE
                
                all_s = torch.cat([s_c.unsqueeze(0), s_p_j]) 
                s_tilde_c = (torch.logsumexp(all_s, dim=0) - np.log(len(all_s))) / LOGIT_SCALE
                
                final_scores[c_idx] = s_tilde_c

        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        
        ranks_kg = [np.where(sorted_indices_kg == t_idx)[0][0] + 1 for t_idx in true_indices]
        kg_ranks.append(min(ranks_kg))

    if skipped > 0:
        print(f"\n⚠️ 提示: 共跳过了 {skipped} 个损坏的音频文件。")

    b_h1, b_h3, b_h5, b_mrr = compute_metrics(baseline_ranks)
    kg_h1, kg_h3, kg_h5, kg_mrr = compute_metrics(kg_ranks)

    print("\n" + "="*60)
    print(f"{'Metric':<12} | {'MS-CLAP (Baseline)':<20} | {'MS-CLAP +KG':<20}")
    print("-" * 60)
    print(f"{'Hit@1':<12} | {b_h1:<20.2f} | {kg_h1:<20.2f}")
    print(f"{'Hit@3':<12} | {b_h3:<20.2f} | {kg_h3:<20.2f}")
    print(f"{'Hit@5':<12} | {b_h5:<20.2f} | {kg_h5:<20.2f}")
    print(f"{'MRR':<12} | {b_mrr:<20.2f} | {kg_mrr:<20.2f}")
    print("="*60)

if __name__ == "__main__":
    main()