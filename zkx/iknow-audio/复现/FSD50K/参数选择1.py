import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import random

sys.path.insert(0, "/home/star/zkx/CLAP/code/CLAP-main/src")
from laion_clap import CLAP_Module
from pykeen.triples import TriplesFactory

# ================== 配置 ==================
MODEL_PATH = "/home/star/zkx/CLAP/model/630k-audioset-fusion-best.pt"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FSD50K_DEV_AUDIO = "/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.dev_audio"
FSD50K_EVAL_AUDIO = "/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.eval_audio"
FSD50K_DEV_CSV = "/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.ground_truth/dev.csv"
FSD50K_VOCAB_CSV = "/home/star/zkx/iknow-audio/data/FSD50K-1/FSD50K.ground_truth/vocabulary.csv"

TOP_K_CLAP = 10
TOP_M_PER_REL = 1

# 候选关系
CANDIDATE_RELATIONS = [
    'has parent',
    'occurs in',
    'co-occurs with',
    'caused by',
    'associated with environment',
    'is a type of',
    'emitted by',
    'has pitch',
]

def load_kge():
    print("加载 KGE 模型...")
    triples_file = os.path.join(KGE_MODEL_DIR, 'AKG_train_triples.tsv')
    triples_factory = TriplesFactory.from_path(triples_file)
    entity_to_id = triples_factory.entity_to_id
    print(f"KG实体总数: {len(entity_to_id)}")
    # 打印前50个实体，了解命名风格
    print("KG实体示例（前50个）:")
    for i, entity in enumerate(list(entity_to_id.keys())[:50]):
        print(f"  {i+1}: {entity}")
    model_path = os.path.join(KGE_MODEL_DIR, 'trained_model.pkl')
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()
    return triples_factory, model

def normalize_entity_name(name):
    return name.lower().replace('_', ' ')

def kge_predict(head_label, relation_label, model, triples_factory, top_m):
    entity_to_id = triples_factory.entity_to_id
    relation_to_id = triples_factory.relation_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}

    head_norm = normalize_entity_name(head_label)
    if head_norm not in entity_to_id or relation_label not in relation_to_id:
        return []
    h_id = entity_to_id[head_norm]
    r_id = relation_to_id[relation_label]

    batch = torch.tensor([[h_id, r_id]], device=DEVICE)
    scores = model.score_t(batch)
    scores = scores.squeeze(0)
    top_indices = torch.topk(scores, k=min(top_m, len(scores))).indices.cpu().numpy()
    return [id_to_entity[idx] for idx in top_indices]

def create_prompt(label, tail):
    return f"{label} {tail}"

def evaluate_sample(audio_path, true_labels, clap_model, text_embeds, class_labels,
                    kge_model, triples_factory, relations_q, agg_mode, alpha=None, temperature=None):
    if not os.path.exists(audio_path):
        return None
    try:
        audio_embed = clap_model.get_audio_embedding_from_filelist([audio_path])
        audio_embed = torch.from_numpy(audio_embed).to(DEVICE)
        audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
    except:
        return None

    sim = torch.matmul(audio_embed, text_embeds.T).squeeze(0).cpu().numpy()

    if agg_mode == 'baseline':
        ranked_indices = np.argsort(sim)[::-1]
        ranks = [np.where(ranked_indices == idx)[0][0] + 1 for idx in true_labels if 0 <= idx < len(class_labels)]
        return min(ranks) if ranks else len(class_labels)+1

    # 知识增强
    top_k_indices = np.argsort(sim)[::-1][:TOP_K_CLAP]
    top_k_labels = [class_labels[i] for i in top_k_indices]
    top_k_scores = sim[top_k_indices]
    new_scores = sim.copy()

    for idx, (label, orig_score) in enumerate(zip(top_k_labels, top_k_scores)):
        enhanced_scores = []
        for rel in relations_q:
            tails = kge_predict(label, rel, kge_model, triples_factory, TOP_M_PER_REL)
            for tail in tails:
                prompt = create_prompt(label, tail)
                prompt_embed = clap_model.get_text_embedding([prompt])
                prompt_embed = torch.from_numpy(prompt_embed).to(DEVICE)
                prompt_embed = prompt_embed / prompt_embed.norm(dim=-1, keepdim=True)
                sim_prompt = torch.matmul(audio_embed, prompt_embed.T).item()
                enhanced_scores.append(sim_prompt)

        if agg_mode == 'max':
            final_score = max(enhanced_scores + [orig_score]) if enhanced_scores else orig_score
        elif agg_mode == 'mean':
            final_score = np.mean(enhanced_scores + [orig_score]) if enhanced_scores else orig_score
        elif agg_mode == 'weighted_mean':
            if enhanced_scores:
                avg_enhanced = np.mean(enhanced_scores)
                final_score = alpha * orig_score + (1 - alpha) * avg_enhanced
            else:
                final_score = orig_score
        elif agg_mode == 'logsumexp':
            if enhanced_scores:
                scores_tensor = torch.tensor(enhanced_scores + [orig_score], device=DEVICE) * temperature
                final_score = torch.logsumexp(scores_tensor, dim=0).item() / temperature
            else:
                final_score = orig_score
        else:
            final_score = orig_score
        new_scores[top_k_indices[idx]] = final_score

    ranked_indices = np.argsort(new_scores)[::-1]
    ranks = [np.where(ranked_indices == idx)[0][0] + 1 for idx in true_labels if 0 <= idx < len(class_labels)]
    return min(ranks) if ranks else len(class_labels)+1

def evaluate_subset(df, clap_model, text_embeds, class_labels, class_to_idx,
                    kge_model, triples_factory, relations_q, agg_mode, alpha=None, temperature=None):
    hit1 = hit3 = hit5 = 0
    rr_sum = 0.0
    total = 0
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = str(row['fname'])  # 转为字符串
        labels_str = row['labels']
        label_list = [l.strip() for l in labels_str.split(',')]
        true_indices = []
        for lbl in label_list:
            if lbl in class_to_idx:
                true_indices.append(class_to_idx[lbl])
            else:
                # 如果标签不在词汇表中，忽略（通常不会发生）
                pass
        if not true_indices:
            skipped += 1
            continue
        audio_path = os.path.join(FSD50K_DEV_AUDIO, fname + '.wav')
        rank = evaluate_sample(audio_path, true_indices, clap_model, text_embeds, class_labels,
                               kge_model, triples_factory, relations_q, agg_mode, alpha, temperature)
        if rank is None:
            skipped += 1
            continue
        if rank == 1:
            hit1 += 1
        if rank <= 3:
            hit3 += 1
        if rank <= 5:
            hit5 += 1
        rr_sum += 1.0 / rank
        total += 1
    if total == 0:
        return 0,0,0,0
    return hit1/total*100, hit3/total*100, hit5/total*100, rr_sum/total

def main():
    print("🚀 加载 CLAP 模型...")
    clap_model = CLAP_Module(enable_fusion=True, amodel='HTSAT-tiny', tmodel='roberta')
    clap_model.load_ckpt(MODEL_PATH)
    clap_model.to(DEVICE)
    clap_model.eval()

    triples_factory, kge_model = load_kge()

    # 读取类别词汇表
    vocab_df = pd.read_csv(FSD50K_VOCAB_CSV, header=None, names=['id', 'label', 'mid'])
    class_labels = vocab_df['label'].tolist()
    class_to_idx = {label: i for i, label in enumerate(class_labels)}
    print(f"类别总数: {len(class_labels)}")

    # 检查KG覆盖率
    entity_to_id = triples_factory.entity_to_id
    kg_entities = set(entity_to_id.keys())
    matched = 0
    unmatched = []
    for lbl in class_labels:
        norm = normalize_entity_name(lbl)
        if norm in kg_entities:
            matched += 1
        else:
            unmatched.append((lbl, norm))
    print(f"KG中找到的类别数: {matched}/{len(class_labels)} ({matched/len(class_labels)*100:.1f}%)")
    if unmatched:
        print("未匹配示例:", unmatched[:10])

    # 预计算文本嵌入
    print("📝 预计算类别文本嵌入...")
    prompt_list = [f"This is a sound of {label.replace('_', ' ')}." for label in class_labels]
    text_embeds = clap_model.get_text_embedding(prompt_list)
    text_embeds = torch.from_numpy(text_embeds).to(DEVICE)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # 读取开发集元数据
    dev_df = pd.read_csv(FSD50K_DEV_CSV)
    # 随机抽取2000个样本作为验证集
    val_df = dev_df.sample(n=2000, random_state=42)

    # 基线
    print("\n========== 基线 ==========")
    base_h1, base_h3, base_h5, base_mrr = evaluate_subset(
        val_df, clap_model, text_embeds, class_labels, class_to_idx,
        kge_model, triples_factory, relations_q=[], agg_mode='baseline')
    print(f"基线 Hit@1: {base_h1:.2f}%, Hit@3: {base_h3:.2f}%, Hit@5: {base_h5:.2f}%, MRR: {base_mrr:.4f}")

    # 探索关系组合（加权平均 alpha=0.8）
    print("\n========== 探索关系组合 (加权平均, alpha=0.8) ==========")
    relation_sets = [
        ['has parent', 'occurs in'],
        ['has parent', 'occurs in', 'co-occurs with'],
        ['has parent', 'occurs in', 'caused by'],
        ['has parent', 'occurs in', 'associated with environment'],
        ['co-occurs with', 'caused by', 'associated with environment'],
        ['has parent', 'occurs in', 'co-occurs with', 'caused by'],
    ]
    for rel_set in relation_sets:
        h1, h3, h5, mrr = evaluate_subset(
            val_df, clap_model, text_embeds, class_labels, class_to_idx,
            kge_model, triples_factory, rel_set, agg_mode='weighted_mean', alpha=0.8)
        print(f"关系 {rel_set}: Hit@1 = {h1:.2f}%, MRR = {mrr:.4f}")

    # 选择最佳关系集（从结果中手动选出，这里用第一个作为示例，但实际应取结果最好的）
    # 您可以根据输出替换下面的 best_rel
    best_rel = ['has parent', 'occurs in', 'co-occurs with']  # 请根据实际输出调整
    print(f"\n========== 探索聚合方式 (关系: {best_rel}) ==========")

    for alpha in [0.6, 0.7, 0.8, 0.9]:
        h1, h3, h5, mrr = evaluate_subset(
            val_df, clap_model, text_embeds, class_labels, class_to_idx,
            kge_model, triples_factory, best_rel, agg_mode='weighted_mean', alpha=alpha)
        print(f"加权平均 alpha={alpha}: Hit@1 = {h1:.2f}%, MRR = {mrr:.4f}")

    h1, h3, h5, mrr = evaluate_subset(
        val_df, clap_model, text_embeds, class_labels, class_to_idx,
        kge_model, triples_factory, best_rel, agg_mode='max')
    print(f"max 聚合: Hit@1 = {h1:.2f}%, MRR = {mrr:.4f}")

    h1, h3, h5, mrr = evaluate_subset(
        val_df, clap_model, text_embeds, class_labels, class_to_idx,
        kge_model, triples_factory, best_rel, agg_mode='mean')
    print(f"mean 聚合: Hit@1 = {h1:.2f}%, MRR = {mrr:.4f}")

    for temp in [5, 10, 20]:
        h1, h3, h5, mrr = evaluate_subset(
            val_df, clap_model, text_embeds, class_labels, class_to_idx,
            kge_model, triples_factory, best_rel, agg_mode='logsumexp', temperature=temp)
        print(f"logsumexp 温度={temp}: Hit@1 = {h1:.2f}%, MRR = {mrr:.4f}")

if __name__ == "__main__":
    main()