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
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 🌟 绑定 2 号 GPU

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
# 1. 核心参数与多模态 RAG 设置 (ESC-50)
# ==========================================
ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv"
ESC50_AUDIO_DIR = "/home/star/zkx/CLAP/data/ESC-50/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/esc50/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🌟🌟🌟 新增：动态阈值自适应超参数 (Adaptive Thresholding) 🌟🌟🌟
MARGIN_K = 0.05      # 类别截断边距：与 Top-1 相似度差距在 0.05 以内的类别全保留
MAX_K = 10           # 动态类别池最大容量兜底 (防极端情况全选)

MARGIN_M = 1.5       # 图谱尾实体边距：与 Top-1 预测分差在 1.5 以内的保留 (PyKEEN分数为负数距离)
MAX_M = 5            # 动态尾实体池最大容量兜底 (防发散)

TOP_P = 5            # GRASP 剪枝保留数
ALPHA = 0.6          # 🌟 类锚定权重
DECAY_GAMMA = 0.85   # 🌟 二跳距离衰减系数
LOGIT_SCALE = 100.0 

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return np.mean(ranks <= 1)*100, np.mean(ranks <= 3)*100, np.mean(ranks <= 5)*100, np.mean(1.0/ranks)*100

def get_kg_entity(esc_class):
    clean_name = esc_class.replace('_', ' ').strip()
    if '(' in clean_name: clean_name = clean_name.split('(')[0].strip()
    return clean_name.replace(' - ', ' ')

# 🌟 加载并解析 LLM 提示词
def load_llm_prompts(file_path):
    prompt_map = {}
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 找不到提示词文件 {file_path}")
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
# 3. 主程序：多跳 + LLM扩充 + 动态阈值 + M3竞技
# ==========================================
@torch.no_grad()
def main():
    print(f"🚀 启动 ESC-50 终极版: [自适应动态截断] + [多跳挖掘] + [LLM自然语义] + [M3衰减聚合]...")
    
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    print(f"📚 成功加载 {len(prompt_map)} 条 LLM 增强提示词。")

    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())
    kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    ALL_RELS = ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children']
    HOP1_RELATIONS = [r for r in ALL_RELS if r in training_factory.relation_to_id]
    HOP2_RELATIONS = [r for r in ['belongs to class', 'has parent', 'event composed of', 'has children'] if r in training_factory.relation_to_id]

    kge_cache = {}
    # 🌟🌟🌟 新增：自适应尾实体挖掘 (Adaptive M) 🌟🌟🌟
    def get_adaptive_tails(head, relation):
        cache_key = (head, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        if head not in training_factory.entity_to_id:
            fallback = head.split(' ')[-1]
            if fallback not in training_factory.entity_to_id: return []
            head = fallback
        try:
            pred = predict_target(model=kge_model, head=head, relation=relation, triples_factory=training_factory)
            df_sorted = pred.df.sort_values(by="score", ascending=False)
            if len(df_sorted) == 0: return []
            
            best_score = df_sorted.iloc[0]['score']
            # 保留分数在 (best_score - MARGIN_M) 以内的实体，最多 MAX_M 个
            valid_df = df_sorted[df_sorted['score'] >= (best_score - MARGIN_M)]
            tails = valid_df.head(MAX_M)['tail_label'].tolist()
            
            kge_cache[cache_key] = tails
            return tails
        except: return []

    df = pd.read_csv(ESC50_CSV)
    unique_categories = sorted(df['category'].unique()) 
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    clean_classes = [cat.replace('_', ' ') for cat in unique_categories]
    class_labels_set = set([c.lower() for c in clean_classes])
    
    text_embeds = clap_model.get_text_embeddings(clean_classes)
    text_embeds = F.normalize(text_embeds, dim=-1)

    ranks = {
        'Baseline': [],
        'M1_MaxAlpha': [],
        'M2_LME_iKnow': [],
        'M3_SoftAlpha_Decay': []
    }

    print(f"🎵 推理开始 ({len(df)} 样本)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adaptive M3+LLM Pipeline"):
        audio_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
        if not os.path.exists(audio_path): continue
        true_idx = class_to_idx[row['category']]
        
        audio_embed = clap_model.get_audio_embeddings([audio_path])
        audio_embed = F.normalize(audio_embed, dim=-1) 
        
        # [0] Baseline 计算
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        ranks['Baseline'].append(np.where(sorted_indices_baseline == true_idx)[0][0] + 1)

        score_m1 = cos_sim_orig.clone()
        score_m2 = cos_sim_orig.clone()
        score_m3 = cos_sim_orig.clone()

        # 🌟🌟🌟 新增：自适应类别截断 (Adaptive K) 🌟🌟🌟
        best_base_score = cos_sim_orig[sorted_indices_baseline[0]].item()
        adaptive_k_indices = []
        for idx in sorted_indices_baseline:
            # 只要候选类别得分与第一名相差在 MARGIN_K (0.05) 之内，我们就给它图谱扩充的机会
            if (best_base_score - cos_sim_orig[idx].item()) <= MARGIN_K:
                adaptive_k_indices.append(idx.item())
            else:
                break
        
        # 兜底：最多保留 MAX_K 个，防止过于模糊的音频把所有类都放进来
        top_k_indices = adaptive_k_indices[:MAX_K]

        for c_idx in top_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = get_kg_entity(unique_categories[c_idx]) 
            
            candidate_info = {}
            
            # --- 动态受限的多跳检索 ---
            for r1 in HOP1_RELATIONS:
                # 换用动态尾实体截断
                hop1_tails = get_adaptive_tails(kg_entity_name, r1)
                for t1 in hop1_tails:
                    t1_clean = t1.lower().strip()
                    if t1_clean != orig_class_name.lower() and t1_clean not in class_labels_set:
                        lk1 = f"{kg_entity_name.lower()}||{r1.lower()}||{t1_clean}"
                        p1 = prompt_map.get(lk1, f"{orig_class_name}, {t1}")
                        candidate_info[t1_clean] = {'prompt': p1, 'is_hop2': False}
                    
                    for r2 in HOP2_RELATIONS:
                        hop2_tails = get_adaptive_tails(t1, r2)
                        for t2 in hop2_tails:
                            t2_clean = t2.lower().strip()
                            if t2_clean != orig_class_name.lower() and t2_clean not in class_labels_set:
                                if t2_clean not in candidate_info:
                                    lk2 = f"{t1_clean}||{r2.lower()}||{t2_clean}"
                                    p2 = prompt_map.get(lk2, f"{orig_class_name}, {t2}")
                                    candidate_info[t2_clean] = {'prompt': p2, 'is_hop2': True}
            
            if len(candidate_info) > 0:
                prompts = []
                gammas = []
                for info in candidate_info.values():
                    prompts.append(info['prompt'])
                    gammas.append(DECAY_GAMMA if info['is_hop2'] else 1.0)
                
                gamma_tensor = torch.tensor(gammas).to(DEVICE)
                
                # 逐句提取文本特征
                p_embs_list = []
                for p in prompts:
                    p_emb_raw = clap_model.get_text_embeddings([p])
                    p_embs_list.append(p_emb_raw)
                
                p_embs = torch.cat(p_embs_list, dim=0)
                p_embs = F.normalize(p_embs, dim=-1)
                
                raw_p_scores = torch.matmul(audio_embed, p_embs.T).squeeze()
                if raw_p_scores.dim() == 0: raw_p_scores = raw_p_scores.unsqueeze(0)
                
                # 💡 剪枝层：Hop-Aware 的 GRASP
                decayed_p_scores = raw_p_scores * gamma_tensor
                _, top_p_idx = torch.topk(decayed_p_scores, min(TOP_P, len(prompts)))
                
                final_p_scores = raw_p_scores[top_p_idx]       
                final_decayed_scores = decayed_p_scores[top_p_idx] 
                
                # ==========================================
                # 💡 聚合层计算
                # ==========================================
                best_p_val = torch.max(final_p_scores)
                score_m1[c_idx] = (ALPHA * cos_sim_orig[c_idx]) + ((1.0 - ALPHA) * best_p_val)
                
                c_logit = cos_sim_orig[c_idx] * LOGIT_SCALE
                p_logits = final_p_scores * LOGIT_SCALE
                all_logits_m2 = torch.cat([c_logit.unsqueeze(0), p_logits])
                score_m2[c_idx] = (torch.logsumexp(all_logits_m2, dim=0) - np.log(len(all_logits_m2))) / LOGIT_SCALE
                
                decayed_p_logits = final_decayed_scores * LOGIT_SCALE
                soft_prompt_score = (torch.logsumexp(decayed_p_logits, dim=0) - np.log(len(decayed_p_logits))) / LOGIT_SCALE
                score_m3[c_idx] = (ALPHA * cos_sim_orig[c_idx]) + ((1.0 - ALPHA) * soft_prompt_score)

        # 结算排名
        ranks['M1_MaxAlpha'].append(np.where(torch.argsort(score_m1, descending=True).cpu().numpy() == true_idx)[0][0] + 1)
        ranks['M2_LME_iKnow'].append(np.where(torch.argsort(score_m2, descending=True).cpu().numpy() == true_idx)[0][0] + 1)
        ranks['M3_SoftAlpha_Decay'].append(np.where(torch.argsort(score_m3, descending=True).cpu().numpy() == true_idx)[0][0] + 1)

    # ==========================================
    # 4. 展示横向对比表格
    # ==========================================
    print("\n" + "=" * 85)
    print(f"{'Metric':<8} | {'Baseline':<12} | {'M1(Max+Alpha)':<15} | {'M2(LME_iKnow)':<15} | {'M3(Soft+Decay)':<15}")
    print("-" * 85)
    
    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    b_res = compute_metrics(ranks['Baseline'])
    m1_res = compute_metrics(ranks['M1_MaxAlpha'])
    m2_res = compute_metrics(ranks['M2_LME_iKnow'])
    m3_res = compute_metrics(ranks['M3_SoftAlpha_Decay'])

    for i, m_name in enumerate(metrics):
        row = f"{m_name:<8} | {b_res[i]:<12.2f} | {m1_res[i]:<15.2f} | {m2_res[i]:<15.2f} | {m3_res[i]:<15.2f}"
        print(row)
    print("=" * 85)
    
    print("\n💡 自适应截断 (Adaptive Thresholding) 总结:")
    print("1. 相比死板的 TOP_K=5，现在只有跟第一名分数差距小于 0.05 的‘强竞争对手’才配去查图谱，减少了对极低分备胎做无用功。")
    print("2. 相比死板的 TOP_M=3，图谱如果只给出一个高分实体，就不会强行拉另外两个垃圾实体凑数，纯净度大幅上升！")

if __name__ == "__main__":
    main()