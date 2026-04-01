import os
import sys
import re
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 🌟 绑定 GPU

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
# 1. 路径与核心参数定义 (AudioSet)
# ==========================================
AUDIO_DIR = "/data/zkx/zkx/iknow-audio/data/audioset/eval_wavs"
ONTOLOGY_JSON = "/data/zkx/zkx/iknow-audio/data/audioset/ontology.json"
AUDIO_LABELS_CSV = "/data/zkx/zkx/iknow-audio/data/audioset/audio_labels.csv"

KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")

# LLM 自然语言提示词
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/audioset/llm_prompts.json"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 基础超参数
TOP_K = 15           
TOP_M = 3            
TOP_P = 5            
DECAY_GAMMA = 0.85   # 用于第3列（单动态）的固定二跳衰减
LOGIT_SCALE = 100.0

MARGIN_K = 0.05
MAX_K = 15
MARGIN_M = 1.5
MAX_M = 5

ALPHA_MIN = 0.4
ALPHA_MAX = 0.8
GAMMA_MIN = 0.70
GAMMA_MAX = 1.00

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return (np.mean(ranks <= 1) * 100,
            np.mean(ranks <= 3) * 100,
            np.mean(ranks <= 5) * 100,
            np.mean(1.0 / ranks) * 100)

def to_tensor(emb):
    if isinstance(emb, np.ndarray): return torch.from_numpy(emb)
    return emb

def load_llm_prompts(file_path):
    prompt_map = {}
    if not os.path.exists(file_path): return prompt_map
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, text in data.items():
            parts = key.split("||")
            if len(parts) == 3: prompt_map["||".join([p.strip().lower() for p in parts])] = text
    return prompt_map

# ==========================================
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    print("🚀 启动 AudioSet 终极评测: 将输出 5 列超硬核对比结果！")

    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)
    AVAILABLE_RELATIONS = list(training_factory.relation_to_id.keys())

    AUDIOSET_RELS = [
        'has parent', 'overlaps with', 'indicates', 'can be heard in', 
        'transcribed as', 'occurs in', 'localized in', 'caused by', 
        'is sound of', 'is variant of'
    ]
    HOP1_RELATIONS = [r for r in AUDIOSET_RELS if r in AVAILABLE_RELATIONS]
    HOP2_RELATIONS = [r for r in ['has parent', 'overlaps with', 'occurs in', 'caused by'] if r in AVAILABLE_RELATIONS]

    kge_cache = {}
    def get_top_m_tails_with_scores(head_entity, relation, m=TOP_M):
        cache_key = (head_entity, relation)
        if cache_key in kge_cache: return kge_cache[cache_key]
        if head_entity not in training_factory.entity_to_id: return []
        try:
            pred = predict_target(model=kge_model, head=head_entity, relation=relation, triples_factory=training_factory)
            df_sorted = pred.df.sort_values(by="score", ascending=False)
            if len(df_sorted) == 0: return []
            
            best_score = df_sorted.iloc[0]['score']
            valid_df = df_sorted[df_sorted['score'] >= (best_score - MARGIN_M)].head(MAX_M)
            tails_with_scores = list(zip(valid_df['tail_label'], valid_df['score']))
            kge_cache[cache_key] = tails_with_scores
            return tails_with_scores
        except: return []

    # ==========================================
    # 🌟 解析 AudioSet Ontology
    # ==========================================
    with open(ONTOLOGY_JSON, 'r', encoding='utf-8') as f:
        ontology_data = json.load(f)
    
    mid_to_name = {}
    mid_to_desc = {}
    mid_to_children_names = {} 
    clean_classes = []
    
    for item in ontology_data:
        mid = item['id']
        name = item['name'].replace('"', '').strip()
        mid_to_name[mid] = name
        mid_to_desc[mid] = item.get('description', '').strip()
        clean_classes.append(name)
        
    for item in ontology_data:
        mid = item['id']
        children_names = []
        for child_mid in item.get('child_ids', []):
            if child_mid in mid_to_name:
                children_names.append(mid_to_name[child_mid])
        mid_to_children_names[mid] = children_names
    
    class_to_idx = {name: i for i, name in enumerate(clean_classes)}
    class_labels_set = set([c.lower() for c in clean_classes])

    text_embeds = to_tensor(clap_model.get_text_embeddings(clean_classes)).to(DEVICE).float()
    text_embeds = F.normalize(text_embeds, dim=-1)

    eval_df = pd.read_csv(AUDIO_LABELS_CSV, dtype=str)
    
    ranks = {
        'Baseline': [],
        'iKnow_Rep': [],
        'Ours_DynAlpha': [],   # 第3列
        'Ours_DualDyn': [],    # 第4列
        'Ours_DualDyn_Child': [] # 第5列
    }

    # ==========================================
    # 聚合算分函数 1：单动态 Alpha + 固定 Decay (第3列专用)
    # ==========================================
    def compute_m3_score_fixed_gamma(c_info, alpha_dyn, cos_orig_c, audio_emb):
        if not c_info: return cos_orig_c
        prompts, gammas = [], []
        for info in c_info.values():
            prompts.append(info['prompt'])
            gammas.append(info['gamma'])  # 使用硬编码的 gamma (1.0 或 0.85)
            
        gamma_tensor = torch.tensor(gammas).to(DEVICE).float()
        
        p_embs_list = [to_tensor(clap_model.get_text_embeddings([p])).to(DEVICE).float() for p in prompts]
        p_embs = torch.cat(p_embs_list, dim=0)
        p_embs = F.normalize(p_embs, dim=-1)
        
        raw_p_scores = torch.matmul(audio_emb, p_embs.T).squeeze()
        if raw_p_scores.dim() == 0: raw_p_scores = raw_p_scores.unsqueeze(0)
        
        decayed_p_scores = raw_p_scores * gamma_tensor
        _, top_p_idx = torch.topk(decayed_p_scores, min(TOP_P, len(prompts)))
        final_decayed_scores = decayed_p_scores[top_p_idx]
        
        decayed_p_logits = final_decayed_scores * LOGIT_SCALE
        soft_prompt_score = (torch.logsumexp(decayed_p_logits, dim=0) - np.log(len(decayed_p_logits))) / LOGIT_SCALE
        return (alpha_dyn * cos_orig_c) + ((1.0 - alpha_dyn) * soft_prompt_score)

    # ==========================================
    # 聚合算分函数 2：双动态 M3 (第4列和第5列共用)
    # ==========================================
    def compute_m3_score_dual_dyn(c_info, alpha_dyn, cos_orig_c, audio_emb):
        if not c_info: return cos_orig_c
        prompts, raw_kge_scores = [], []
        for info in c_info.values():
            prompts.append(info['prompt'])
            raw_kge_scores.append(info['kge_score'])
            
        raw_kge_scores = np.array(raw_kge_scores)
        kg_scores_only = [s for s in raw_kge_scores if s < 90.0] 
        min_s = min(kg_scores_only) if kg_scores_only else 0
        max_s = max(kg_scores_only) if kg_scores_only else 0
        
        gammas = []
        for s in raw_kge_scores:
            if s > 90.0:  # 官方描述或子类，给满权重 1.0
                gammas.append(1.0)
            elif max_s - min_s < 1e-5:
                gammas.append(1.0)
            else:
                gammas.append(GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * (s - min_s) / (max_s - min_s))
        
        gamma_tensor = torch.tensor(gammas).to(DEVICE).float()
        
        p_embs_list = [to_tensor(clap_model.get_text_embeddings([p])).to(DEVICE).float() for p in prompts]
        p_embs = torch.cat(p_embs_list, dim=0)
        p_embs = F.normalize(p_embs, dim=-1)
        
        raw_p_scores = torch.matmul(audio_emb, p_embs.T).squeeze()
        if raw_p_scores.dim() == 0: raw_p_scores = raw_p_scores.unsqueeze(0)
        
        decayed_p_scores = raw_p_scores * gamma_tensor
        _, top_p_idx = torch.topk(decayed_p_scores, min(TOP_P, len(prompts)))
        final_decayed_scores = decayed_p_scores[top_p_idx]
        
        decayed_p_logits = final_decayed_scores * LOGIT_SCALE
        soft_prompt_score = (torch.logsumexp(decayed_p_logits, dim=0) - np.log(len(decayed_p_logits))) / LOGIT_SCALE
        return (alpha_dyn * cos_orig_c) + ((1.0 - alpha_dyn) * soft_prompt_score)

    print(f"🎵 推理开始 (总计 {len(eval_df)} 个多标签音频)...")
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="AudioSet Pipeline"):
        fname = str(row.iloc[0])
        if not fname.endswith('.wav'): fname += '.wav'
        audio_path = os.path.join(AUDIO_DIR, fname)
        if not os.path.exists(audio_path): continue
        
        raw_labels = str(row.iloc[1])
        mids = re.findall(r'/m/\w+|/t/\w+', raw_labels)
        
        true_indices = []
        if mids:
            for mid in mids:
                if mid in mid_to_name: true_indices.append(class_to_idx[mid_to_name[mid]])
        else:
            parts = [p.replace('"', '').strip() for p in raw_labels.split(',')]
            for p in parts:
                if p in class_to_idx: true_indices.append(class_to_idx[p])
                    
        true_indices = list(set(true_indices))
        if not true_indices: continue

        try: audio_embed = clap_model.get_audio_embeddings([audio_path])
        except: continue

        audio_embed = to_tensor(audio_embed).to(DEVICE).float()
        audio_embed = F.normalize(audio_embed, dim=-1)

        # 1. Baseline
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices_baseline = torch.argsort(cos_sim_orig, descending=True).cpu().numpy()
        ranks['Baseline'].append(min([np.where(sorted_indices_baseline == t_idx)[0][0] + 1 for t_idx in true_indices]))

        # ========== 🌟 动态 Alpha 计算 ==========
        max_sim = torch.max(cos_sim_orig).item()
        alpha_dynamic = max(ALPHA_MIN, min(ALPHA_MAX, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim))

        score_iknow = cos_sim_orig.clone()
        score_ours_dyn_alpha = cos_sim_orig.clone()
        score_ours_dual_dyn = cos_sim_orig.clone()
        score_ours_dual_dyn_child = cos_sim_orig.clone()

        best_base_score = cos_sim_orig[sorted_indices_baseline[0]].item()
        adaptive_k_indices = [idx.item() for idx in sorted_indices_baseline if (best_base_score - cos_sim_orig[idx].item()) <= MARGIN_K][:MAX_K]

        for c_idx in adaptive_k_indices:
            orig_class_name = clean_classes[c_idx]
            kg_entity_name = orig_class_name.lower()
            
            # ==========================================
            # 🟢 Column 2: iKnow 官方复现
            # ==========================================
            iknow_tails = set()
            for r1 in HOP1_RELATIONS:
                for t1, _ in get_top_m_tails_with_scores(kg_entity_name, r1, m=3):
                    if t1.lower() != kg_entity_name and t1.lower() not in class_labels_set:
                        iknow_tails.add(t1)
            
            iknow_prompts = [f"{orig_class_name}, {t}" for t in iknow_tails]
            if len(iknow_prompts) > 0:
                iknow_embs = to_tensor(clap_model.get_text_embeddings(iknow_prompts)).to(DEVICE).float()
                iknow_embs = F.normalize(iknow_embs, dim=-1)
                iknow_sims = torch.matmul(audio_embed, iknow_embs.T).squeeze(0)
                if iknow_sims.dim() == 0: iknow_sims = iknow_sims.unsqueeze(0)
                s_c_iknow = cos_sim_orig[c_idx] * LOGIT_SCALE
                s_p_iknow = iknow_sims * LOGIT_SCALE
                all_iknow = torch.cat([s_c_iknow.unsqueeze(0), s_p_iknow])
                score_iknow[c_idx] = (torch.logsumexp(all_iknow, dim=0) - np.log(len(all_iknow))) / LOGIT_SCALE

            # ==========================================
            # 🔴 Column 3, 4 & 5: Ours 各种版本构建
            # ==========================================
            c_info_3 = {} # 仅动态 Alpha (固定衰减)
            c_info_4 = {} # 双动态 M3
            c_info_5 = {} # 双动态 M3 + Child IDs
            
            target_mid = [mid for mid, name in mid_to_name.items() if name == orig_class_name]
            if target_mid:
                mid_str = target_mid[0]
                # 注入官方描述
                if mid_to_desc.get(mid_str):
                    desc_prompt = f"{orig_class_name}. {mid_to_desc[mid_str]}"
                    c_info_3['ontology_desc'] = {'prompt': desc_prompt, 'gamma': 1.0}  # 强制 1.0
                    c_info_4['ontology_desc'] = {'prompt': desc_prompt, 'kge_score': 99.0}
                    c_info_5['ontology_desc'] = {'prompt': desc_prompt, 'kge_score': 99.0}
                
                # 注入官方子类 (仅存在于 Column 5)
                if mid_to_children_names.get(mid_str):
                    for child_name in mid_to_children_names[mid_str]:
                        if child_name.lower() not in class_labels_set:
                            c_info_5[f"child_{child_name}"] = {'prompt': f"{orig_class_name}, {child_name}", 'kge_score': 98.0}

            # 多跳检索注入 (应用于 3, 4, 5)
            for r1 in HOP1_RELATIONS:
                for t1, s1 in get_top_m_tails_with_scores(kg_entity_name, r1, TOP_M):
                    t1_c = t1.lower().strip()
                    if t1_c != kg_entity_name and t1_c not in class_labels_set:
                        lk1 = f"{kg_entity_name}||{r1.lower()}||{t1_c}"
                        p1 = prompt_map.get(lk1, f"{orig_class_name}, {t1}")
                        
                        if t1_c not in c_info_4 or c_info_4[t1_c]['kge_score'] < s1:
                            c_info_3[t1_c] = {'prompt': p1, 'gamma': 1.0} # 1跳满权重
                            c_info_4[t1_c] = {'prompt': p1, 'kge_score': s1}
                            c_info_5[t1_c] = {'prompt': p1, 'kge_score': s1}

                    for r2 in HOP2_RELATIONS:
                        for t2, s2 in get_top_m_tails_with_scores(t1, r2, TOP_M):
                            t2_c = t2.lower().strip()
                            if t2_c != kg_entity_name and t2_c not in class_labels_set:
                                avg_path_score = (s1 + s2) / 2.0
                                if t2_c not in c_info_4 or c_info_4[t2_c]['kge_score'] < avg_path_score:
                                    lk2 = f"{t1_c}||{r2.lower()}||{t2_c}"
                                    p2 = prompt_map.get(lk2, f"{orig_class_name}, {t2}")
                                    
                                    c_info_3[t2_c] = {'prompt': p2, 'gamma': DECAY_GAMMA} # 2跳死板衰减 0.85
                                    c_info_4[t2_c] = {'prompt': p2, 'kge_score': avg_path_score}
                                    c_info_5[t2_c] = {'prompt': p2, 'kge_score': avg_path_score}
            
            # 结算分数列
            score_ours_dyn_alpha[c_idx] = compute_m3_score_fixed_gamma(c_info_3, alpha_dynamic, cos_sim_orig[c_idx], audio_embed)
            score_ours_dual_dyn[c_idx] = compute_m3_score_dual_dyn(c_info_4, alpha_dynamic, cos_sim_orig[c_idx], audio_embed)
            score_ours_dual_dyn_child[c_idx] = compute_m3_score_dual_dyn(c_info_5, alpha_dynamic, cos_sim_orig[c_idx], audio_embed)

        # 记录多标签排名
        ranks['iKnow_Rep'].append(min([np.where(torch.argsort(score_iknow, descending=True).cpu().numpy() == t_idx)[0][0] + 1 for t_idx in true_indices]))
        ranks['Ours_DynAlpha'].append(min([np.where(torch.argsort(score_ours_dyn_alpha, descending=True).cpu().numpy() == t_idx)[0][0] + 1 for t_idx in true_indices]))
        ranks['Ours_DualDyn'].append(min([np.where(torch.argsort(score_ours_dual_dyn, descending=True).cpu().numpy() == t_idx)[0][0] + 1 for t_idx in true_indices]))
        ranks['Ours_DualDyn_Child'].append(min([np.where(torch.argsort(score_ours_dual_dyn_child, descending=True).cpu().numpy() == t_idx)[0][0] + 1 for t_idx in true_indices]))

    # ==========================================
    # 4. 打印最终学术对比表 (5列)
    # ==========================================
    b_res = compute_metrics(ranks['Baseline'])
    iknow_res = compute_metrics(ranks['iKnow_Rep'])
    ours_3_res = compute_metrics(ranks['Ours_DynAlpha'])
    ours_4_res = compute_metrics(ranks['Ours_DualDyn'])
    ours_5_res = compute_metrics(ranks['Ours_DualDyn_Child'])
    
    col_w = 17
    header = f"{'Metric':<8} | {'Baseline':<{col_w}} | {'iKnow(MS-CLAP)':<{col_w}} | {'Ours(DynAlpha)':<{col_w}} | {'Ours(DualDyn)':<{col_w}} | {'Ours(+Child)':<{col_w}}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
    for i, m_name in enumerate(metrics):
        row_str = f"{m_name:<8} | {b_res[i]:<{col_w}.2f} | {iknow_res[i]:<{col_w}.2f} | {ours_3_res[i]:<{col_w}.2f} | {ours_4_res[i]:<{col_w}.2f} | {ours_5_res[i]:<{col_w}.2f}"
        print(row_str)
    print("=" * len(header))
    
    print("\n💡 终极消融实验解析 (AudioSet 5列验证):")
    print("1. Baseline: 纯净 CLAP 裸考。")
    print("2. iKnow: 官方方法，简单粗暴 (单跳+无衰减)。")
    print("3. Ours(DynAlpha): 仅使用动态 Alpha，二跳知识采用死板的 0.85 衰减。")
    print("4. Ours(DualDyn): 引入图谱分数计算动态 Gamma，打破 0.85 的物理限制！")
    print("5. Ours(+Child): 再叠加一层官方无幻觉的 child_ids，探寻 SOTA 的天花板！")

if __name__ == "__main__":
    main()