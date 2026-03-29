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
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 🌟 绑定 2 号 GPU

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

# 🌟 LLM 自然语言提示词文件路径
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/fsd50k/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟🌟🌟 新增：动态阈值自适应超参数 (Adaptive Thresholding) 🌟🌟🌟
MARGIN_K = 0.05      # 类别截断边距：与 Top-1 相似度差距在 0.05 以内的类别全保留
MAX_K = 15           # FSD50K 类别多，动态类别池最大容量兜底放宽到 15

MARGIN_M = 1.5       # 图谱尾实体边距：与 Top-1 预测分差在 1.5 以内的保留
MAX_M = 5            # 动态尾实体池最大容量兜底

TOP_P = 5            # GRASP 剪枝保留数
ALPHA = 0.6          # 🌟 类锚定权重 (防模态遗忘)
DECAY_GAMMA = 0.85   # 🌟 多跳衰减系数 (打压二跳泛化噪音)
LOGIT_SCALE = 100.0

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

def get_kg_entity(fsd_class):
    return fsd_class.replace('_', ' ').replace(' and ', ' ').lower()

def to_tensor(emb):
    if isinstance(emb, np.ndarray): return torch.from_numpy(emb)
    return emb

# 🌟 加载并解析 LLM 提示词
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
# 3. 主程序：自适应截断 + 多跳 + LLM自然语义 + M3聚合
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 FSD50K 终极版: [自适应动态截断] + [多跳图谱] + [LLM自然语义] + [M3衰减聚合]...")
    
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    print(f"📚 成功加载 {len(prompt_map)} 条 LLM 增强提示词。")
    
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())
    
    # 💡 FSD50K 本体关系配置
    ALL_RELS = ['belongs to class', 'has parent', 'is a type of']
    HOP1_RELATIONS = [r for r in ALL_RELS if r in AVAILABLE_RELATIONS]
    HOP2_RELATIONS = [r for r in ALL_RELS if r in AVAILABLE_RELATIONS]

    kge_cache = {}
    # 🌟🌟🌟 新增：自适应尾实体挖掘 (Adaptive M) 🌟🌟🌟
    def get_adaptive_tails(head_entity, relation):
        cache_key = (head_entity, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        query = head_entity if head_entity in training_factory.entity_to_id else head_entity.split(' ')[-1]
        if query not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=query, relation=relation, triples_factory=training_factory)
            df_sorted = pred.df.sort_values(by="score", ascending=False)
            if len(df_sorted) == 0: return []
            
            best_score = df_sorted.iloc[0]['score']
            # 动态截断
            valid_df = df_sorted[df_sorted['score'] >= (best_score - MARGIN_M)]
            tails = valid_df.head(MAX_M)['tail_label'].tolist()
            
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
    
    baseline_ranks, m3_llm_ranks = [], []

    print(f"🎵 推理开始 (总计 {len(eval_df)} 个多标签音频)...")
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Adaptive M3+LLM"):
        fname = str(row['fname'])
        audio_path = os.path.join(FSD_EVAL_AUDIO, fname + ".wav")
        if not os.path.exists(audio_path): continue
        
        # 多标签解析
        true_labels = row['labels'].split(',')
        true_indices = [vocab_name_to_idx[lbl] for lbl in true_labels if lbl in vocab_name_to_idx]
        if not true_indices: continue
        
        audio_embed = to_tensor(clap_model.get_audio_embeddings([audio_path])).to(DEVICE).float()
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # 1. Baseline 计算
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        baseline_ranks.append(min([np.where(sorted_indices_baseline == t_idx)[0][0] + 1 for t_idx in true_indices]))

        final_scores = cos_sim_orig.clone()
        
        # 🌟🌟🌟 新增：自适应类别截断 (Adaptive K) 🌟🌟🌟
        best_base_score = cos_sim_orig[sorted_indices_baseline[0]].item()
        adaptive_k_indices = []
        for idx in sorted_indices_baseline:
            if (best_base_score - cos_sim_orig[idx].item()) <= MARGIN_K:
                adaptive_k_indices.append(idx.item())
            else:
                break
        
        top_k_indices = adaptive_k_indices[:MAX_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            
            # 记录合并信息: dict[tail_name, {'prompt': str, 'is_hop2': bool}]
            candidate_info = {}
            
            # --- Hop 1 检索 ---
            for r1 in HOP1_RELATIONS:
                for t1 in get_adaptive_tails(kg_entity_name, r1):
                    t1_c = t1.lower().strip()
                    if t1_c != orig_class_name.lower() and t1_c not in class_labels_set and t1_c != kg_entity_name.lower():
                        # 查表匹配 LLM 描述
                        lk1 = f"{kg_entity_name.lower()}||{r1.lower()}||{t1_c}"
                        p1 = prompt_map.get(lk1, f"{orig_class_name}, {t1}")
                        candidate_info[t1_c] = {'prompt': p1, 'is_hop2': False}
                        
                        # --- Hop 2 检索 ---
                        for r2 in HOP2_RELATIONS:
                            for t2 in get_adaptive_tails(t1, r2):
                                t2_c = t2.lower().strip()
                                if t2_c != orig_class_name.lower() and t2_c not in class_labels_set and t2_c != kg_entity_name.lower():
                                    # 查表匹配 LLM 描述
                                    lk2 = f"{t1_c}||{r2.lower()}||{t2_c}"
                                    p2 = prompt_map.get(lk2, f"{orig_class_name}, {t2}")
                                    # 保留最高优先级 (Hop1 优先)
                                    if t2_c not in candidate_info:
                                        candidate_info[t2_c] = {'prompt': p2, 'is_hop2': True}
            
            if len(candidate_info) > 0:
                prompts = []
                gammas = []
                for info in candidate_info.values():
                    prompts.append(info['prompt'])
                    gammas.append(DECAY_GAMMA if info['is_hop2'] else 1.0)
                
                gamma_tensor = torch.tensor(gammas).to(DEVICE)
                
                # 分批提取文本特征 (防 FSD50K 显存 OOM)
                p_embs_list = []
                for p in prompts:
                    p_emb_raw = clap_model.get_text_embeddings([p])
                    p_embs_list.append(to_tensor(p_emb_raw).to(DEVICE).float())
                
                p_embs = torch.cat(p_embs_list, dim=0)
                p_embs = F.normalize(p_embs, dim=-1)
                
                raw_p_scores = torch.matmul(audio_embed, p_embs.T).squeeze()
                if raw_p_scores.dim() == 0: raw_p_scores = raw_p_scores.unsqueeze(0)
                
                # 💡 GRASP 剪枝：应用衰减系数进行惩罚过滤
                decayed_p_scores = raw_p_scores * gamma_tensor
                _, top_p_idx = torch.topk(decayed_p_scores, min(TOP_P, len(prompts)))
                
                # 提取存活的衰减分数
                final_decayed_scores = decayed_p_scores[top_p_idx]
                
                # 💡 M3 终极聚合机制 (Soft+Decay + Alpha)
                decayed_p_logits = final_decayed_scores * LOGIT_SCALE
                # 软池化
                soft_prompt_score = (torch.logsumexp(decayed_p_logits, dim=0) - np.log(len(decayed_p_logits))) / LOGIT_SCALE
                # Alpha 加权锚定
                final_scores[c_idx] = (ALPHA * cos_sim_orig[c_idx]) + ((1.0 - ALPHA) * soft_prompt_score)

        # 结算 FSD50K 多标签名次
        sorted_indices_kg = torch.argsort(final_scores, descending=True).cpu().numpy()
        m3_llm_ranks.append(min([np.where(sorted_indices_kg == t_idx)[0][0] + 1 for t_idx in true_indices]))

    # ==========================================
    # 4. 打印格式化对比表
    # ==========================================
    b_res = compute_metrics(baseline_ranks)
    m3_res = compute_metrics(m3_llm_ranks)
    
    col_w = 26
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'Adaptive M3 + LLM 增强':<{col_w}}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    for i, m_name in enumerate(metrics):
        row_str = f"{m_name:<8} | {b_res[i]:<{col_w}.2f} | {m3_res[i]:<{col_w}.2f}"
        print(row_str)
    print("=" * len(header))
    
    print("\n💡 终极版机制解析 (FSD50K):")
    print("1. 动态自适应截断：FSD50K 类太多，如果最高分类得分极高，自适应截断会瞬间剪除无关类，防止长尾噪音感染！")
    print("2. 深度整合：分类学图谱构建广度，LLM赋能场景深度。")
    print("3. 衰减拦截：M3 的 0.85 衰减严密把控了 Taxonomy 分类学向极度宽泛父类滑落的风险！")

if __name__ == "__main__":
    main()