import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/home/star/zkx/CLAP/code/CLAP-main/src")
from laion_clap import CLAP_Module
from pykeen.triples import TriplesFactory

# ================== 配置 ==================
MODEL_PATH = "/home/star/zkx/CLAP/model/630k-audioset-fusion-best.pt"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# TUT2017 路径
TUT_DEV_ROOT = "/home/star/zkx/iknow-audio/data/TUT2017/development/TUT-acoustic-scenes-2017-development"
TUT_EVAL_ROOT = "/home/star/zkx/iknow-audio/data/TUT2017/evaluation/TUT-acoustic-scenes-2017-evaluation"

TOP_K_CLAP = 10
TOP_M_PER_REL = 1

# 候选关系（根据AKG实际存在的关系选择）
CANDIDATE_RELATIONS = [
    'has parent',
    'occurs in',
    'associated with environment',
    'co-occurs with',
    'caused by',
    'emitted by',
    'is a type of'
]

def load_kge():
    print("加载 KGE 模型...")
    triples_file = os.path.join(KGE_MODEL_DIR, 'AKG_train_triples.tsv')
    triples_factory = TriplesFactory.from_path(triples_file)
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

def evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                  kge_model, triples_factory, relations_q, agg_mode, alpha=None, temperature=None):
    hit1 = 0
    total = 0
    skipped = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # 注意：test_df中的文件名可能带有 "audio/" 前缀，需要正确拼接
        filename = row['filename'].strip()
        if filename.startswith('audio/'):
            rel_path = filename
        else:
            rel_path = f"audio/{filename}"
        audio_path = os.path.join(TUT_DEV_ROOT, rel_path)  # 开发集
        if not os.path.exists(audio_path):
            # 尝试评估集路径（如果测试集来自评估集）
            audio_path = os.path.join(TUT_EVAL_ROOT, rel_path)
        if not os.path.exists(audio_path):
            skipped += 1
            continue

        true_label = row['scene_label']
        true_idx = class_to_idx[true_label]

        try:
            audio_embed = clap_model.get_audio_embedding_from_filelist([audio_path])
            audio_embed = torch.from_numpy(audio_embed).to(DEVICE)
            audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
        except Exception as e:
            print(f"\n警告：跳过文件 {audio_path}，原因：{e}")
            skipped += 1
            continue

        sim = torch.matmul(audio_embed, text_embeds.T).squeeze(0).cpu().numpy()

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
        try:
            rank = np.where(ranked_indices == true_idx)[0][0] + 1
        except:
            rank = len(class_labels) + 1

        if rank == 1:
            hit1 += 1
        total += 1

    if skipped > 0:
        print(f"\n⚠️ 跳过 {skipped} 个缺失文件")
    return hit1 / total * 100

def main():
    print("🚀 加载 CLAP 模型...")
    clap_model = CLAP_Module(enable_fusion=True, amodel='HTSAT-tiny', tmodel='roberta')
    clap_model.load_ckpt(MODEL_PATH)
    clap_model.to(DEVICE)
    clap_model.eval()

    triples_factory, kge_model = load_kge()

    # 检查 KG 实体覆盖率（针对 TUT 场景类别）
    entity_to_id = triples_factory.entity_to_id
    kg_entities = set(entity_to_id.keys())
    print(f"KG 实体总数: {len(kg_entities)}")
    print("KG 实体示例（前20）:", list(kg_entities)[:20])

    # TUT2017 场景类别（15类）
    scene_classes = [
        'beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
        'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
        'office', 'park', 'residential_area', 'train', 'tram'
    ]
    # 注意：类别名中的 '/' 需要处理，KG 中可能没有斜杠，但 normalize_entity_name 会保留空格？
    # 我们先将类别标准化为小写并替换下划线，但斜杠如何处理？可能 KG 中为 "cafe restaurant" 或类似。
    # 先按原样传入，但需要手动映射。先打印匹配情况。
    class_labels = [c.replace('_', ' ') for c in scene_classes]  # 用于 prompt 的标签（下划线转空格）
    class_to_idx = {c: i for i, c in enumerate(scene_classes)}

    print("📝 预计算类别文本嵌入...")
    prompt_list = [f"This is a sound of {label}." for label in class_labels]
    text_embeds = clap_model.get_text_embedding(prompt_list)
    text_embeds = torch.from_numpy(text_embeds).to(DEVICE)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # 读取开发集的一个 fold 作为测试（例如 fold1_evaluate.txt）
    fold_file = os.path.join(TUT_DEV_ROOT, 'evaluation_setup', 'fold1_evaluate.txt')
    test_df = pd.read_csv(fold_file, sep='\t', header=None, names=['filename', 'scene_label'])
    print(f"\n📌 测试 Fold 1 (样本数: {len(test_df)})")

    # 计算基线
    baseline = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        filename = row['filename'].strip()
        if filename.startswith('audio/'):
            rel_path = filename
        else:
            rel_path = f"audio/{filename}"
        audio_path = os.path.join(TUT_DEV_ROOT, rel_path)
        if not os.path.exists(audio_path):
            continue
        true_label = row['scene_label']
        true_idx = class_to_idx[true_label]
        audio_embed = clap_model.get_audio_embedding_from_filelist([audio_path])
        audio_embed = torch.from_numpy(audio_embed).to(DEVICE)
        audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
        sim = torch.matmul(audio_embed, text_embeds.T).squeeze(0).cpu().numpy()
        pred_idx = np.argmax(sim)
        if pred_idx == true_idx:
            baseline += 1
    baseline_hit1 = baseline / len(test_df) * 100
    print(f"基线 Hit@1: {baseline_hit1:.2f}%")

    # 探索不同关系组合（加权平均，alpha=0.8）
    print("\n========== 探索关系组合 (加权平均, alpha=0.8) ==========")
    relation_sets = [
        ['has parent', 'occurs in'],
        ['has parent', 'occurs in', 'associated with environment'],
        ['has parent', 'occurs in', 'co-occurs with'],
        ['has parent', 'occurs in', 'caused by'],
        ['occurs in', 'associated with environment', 'co-occurs with'],
        ['has parent', 'occurs in', 'associated with environment', 'co-occurs with'],
    ]
    for rel_set in relation_sets:
        hit1 = evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                              kge_model, triples_factory, rel_set,
                              agg_mode='weighted_mean', alpha=0.8)
        print(f"关系 {rel_set}: Hit@1 = {hit1:.2f}%")

    # 选择最佳关系集（根据上面结果手动选一个，这里假设最佳为 ['has parent', 'occurs in', 'co-occurs with']）
    best_rel = ['has parent', 'occurs in', 'co-occurs with']
    print(f"\n========== 探索聚合方式 (关系: {best_rel}) ==========")

    # 加权平均不同 alpha
    for alpha in [0.6, 0.7, 0.8, 0.9]:
        hit1 = evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                              kge_model, triples_factory, best_rel,
                              agg_mode='weighted_mean', alpha=alpha)
        print(f"加权平均 alpha={alpha}: Hit@1 = {hit1:.2f}%")

    # max 聚合
    hit1 = evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                          kge_model, triples_factory, best_rel,
                          agg_mode='max')
    print(f"max 聚合: Hit@1 = {hit1:.2f}%")

    # mean 聚合
    hit1 = evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                          kge_model, triples_factory, best_rel,
                          agg_mode='mean')
    print(f"mean 聚合: Hit@1 = {hit1:.2f}%")

    # logsumexp 不同温度
    for temp in [5, 10, 20]:
        hit1 = evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                              kge_model, triples_factory, best_rel,
                              agg_mode='logsumexp', temperature=temp)
        print(f"logsumexp 温度={temp}: Hit@1 = {hit1:.2f}%")

if __name__ == "__main__":
    main()