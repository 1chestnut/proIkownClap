import os
import sys
import contextlib
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json # 🌟 新增：导入 json 解析模块

# ==========================================
# 0. 断网防御与最强离线拦截锁
# ==========================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 🌟 路径定义
LOCAL_MODEL_DIR = "/home/star/zkx/iknow-audio/data/model"
CLAP_WEIGHTS_PATH = os.path.join(LOCAL_MODEL_DIR, "CLAP_weights_2023.pth")
GPT2_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "gpt2")
ROBERTA_LOCAL_PATH = "/home/star/zkx/CLAP/model/roberta-base"

# 🌟 离线锁 1：拦截权重下载，重定向到本地 .pth 文件
import msclap.CLAPWrapper
def offline_hf_hub_download(*args, **kwargs):
    if not os.path.exists(CLAP_WEIGHTS_PATH):
        raise FileNotFoundError(f"未找到权重: {CLAP_WEIGHTS_PATH}")
    return CLAP_WEIGHTS_PATH
msclap.CLAPWrapper.hf_hub_download = offline_hf_hub_download

# 🌟 离线锁 2：拦截所有文本模型/分词器请求
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
# 1. 路径与核心参数定义 (适配 US8K)
# ==========================================
US8K_CSV = "/home/star/zkx/iknow-audio/data/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
US8K_AUDIO_DIR = "/home/star/zkx/iknow-audio/data/UrbanSound8K/UrbanSound8K/audio"

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# 🌟 新增：US8K 的 LLM 自然语言提示词文件路径 (请确认路径是否正确！)
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/提示词扩展/us8k/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟 黄金胜利参数组合
TOP_K = 5   
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

# 🌟 新增：加载并解析 LLM 提示词
def load_llm_prompts(file_path):
    prompt_map = {}
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 找不到提示词文件 {file_path}，将回退到基础拼接格式。")
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
# 2. US8K 专属实体映射字典
# ==========================================
def get_kg_entity(us8k_class):
    mapping = {
        'air_conditioner': 'air conditioner',
        'car_horn': 'car horn',
        'children_playing': 'children playing',
        'dog_bark': 'dog barking', 
        'drilling': 'drilling',
        'engine_idling': 'engine idling',
        'gun_shot': 'gunshot',
        'jackhammer': 'jackhammer',
        'siren': 'siren',
        'street_music': 'street music'
    }
    return mapping.get(us8k_class, us8k_class.replace('_', ' '))

# ==========================================
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 iKnow-audio US8K (创新版：自然语言 + Late Fusion)...")
    
    # 🌟 加载 US8K 的自然语言字典
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    print(f"📚 成功加载 {len(prompt_map)} 条自然语言提示词。")
    
    # 初始化微软 CLAP
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    # ------------------ 🌟 关系设定 🌟 ------------------
    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())
    TARGET_RELATIONS = [
        'overlaps with', 'occurs in', 'associated with environment', 
        'localized in', 'used for', 'part of scene'
    ]
    VALID_RELATIONS = [r for r in TARGET_RELATIONS if r in AVAILABLE_RELATIONS]
    print(f"🔗 使用对 US8K 最优的 {len(VALID_RELATIONS)} 个场景环境关系...")

    def get_top_m_tails(head_entity, relation, m=3):
        query_entity = head_entity
        if query_entity not in training_factory.entity_to_id:
            fallback = query_entity.split(' ')[0] 
            if fallback in training_factory.entity_to_id:
                query_entity = fallback
            else:
                return []
        try:
            pred = predict_target(model=kge_model, head=query_entity, relation=relation, triples_factory=training_factory)
            return pred.df.sort_values(by="score", ascending=False).head(m)['tail_label'].tolist()
        except: return []

    # ------------------ 准备数据 ------------------
    df = pd.read_csv(US8K_CSV)
    unique_categories = sorted(df['class'].unique()) 
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    
    clean_classes = [cat.replace('_', ' ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    # 获取基准文本特征
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    print(f"🎵 推理开始 (总计 {len(df)} 样本)...")
    baseline_ranks, kg_ranks = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        audio_path = os.path.join(US8K_AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['class']]
        
        # 1. 音频特征提取
        audio_embed = clap_model.get_audio_embeddings([audio_path])
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 2. 计算基准相似度
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        baseline_ranks.append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        final_scores = cos_sim_orig.clone() 
        top_k_indices = sorted_indices_baseline[:TOP_K]

        # 3. 提取与融合知识 (🌟 创新点：查表获取自然语言)
        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            
            unique_prompts = set()
            for r in VALID_RELATIONS:
                tails = get_top_m_tails(kg_entity_name, r, TOP_M)
                for t in tails:
                    t_clean = t.lower().strip()
                    if t_clean == orig_class_name.lower() or t_clean == kg_entity_name.lower():
                        continue
                    if t_clean in class_labels_set:
                        continue
                    
                    # 🔍 查表：如果有自然语言则用自然语言，没有则兜底拼接
                    lookup_key = f"{kg_entity_name.lower()}||{r.lower()}||{t_clean}"
                    if lookup_key in prompt_map:
                        unique_prompts.add(prompt_map[lookup_key])
                    else:
                        unique_prompts.add(f"{orig_class_name}, {t}")
            
            enriched_prompts = list(unique_prompts)
            
            if len(enriched_prompts) > 0:
                # 1. 提取自然语言提示词的特征
                embs_list = []
                for prompt in enriched_prompts:
                    single_emb = clap_model.get_text_embeddings([prompt])
                    embs_list.append(single_emb)
                
                prompt_embs = torch.cat(embs_list, dim=0) 
                prompt_embs = F.normalize(prompt_embs, dim=-1)
                
                # 2. 计算音频与所有提示词的相似度得分
                prompt_scores = torch.matmul(audio_embed, prompt_embs.T).squeeze()
                
                # 🌟 统一策略 1：坚决使用 Max Pooling（最高分机制），防止特征稀释！
                if prompt_scores.dim() == 0:
                    best_prompt_score = prompt_scores
                else:
                    best_prompt_score = torch.max(prompt_scores)
                
                # 🌟 统一策略 2：坚决锁定 ALPHA = 0.85（原特征占主导，自然语言打辅助）
                ALPHA = 0.85
                original_score = cos_sim_orig[c_idx]
                
                # Late Fusion 加权融合
                new_score = (ALPHA * original_score) + ((1.0 - ALPHA) * best_prompt_score)
                final_scores[c_idx] = new_score

        # 5. 最终预测重排
        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        kg_ranks.append(np.where(sorted_indices_kg == true_idx)[0][0] + 1)

    b_h1, b_h3, b_h5, b_mrr = compute_metrics(baseline_ranks)
    kg_h1, kg_h3, kg_h5, kg_mrr = compute_metrics(kg_ranks)

    print("\n" + "="*60)
    print(f"{'Metric':<12} | {'MS-CLAP (Baseline)':<20} | {'MS-CLAP +KG&NL (创新)':<20}")
    print("-" * 60)
    print(f"{'Hit@1':<12} | {b_h1:<20.2f} | {kg_h1:<20.2f}")
    print(f"{'Hit@3':<12} | {b_h3:<20.2f} | {kg_h3:<20.2f}")
    print(f"{'Hit@5':<12} | {b_h5:<20.2f} | {kg_h5:<20.2f}")
    print(f"{'MRR':<12} | {b_mrr:<20.2f} | {kg_mrr:<20.2f}")
    print("="*60)

if __name__ == "__main__":
    main()