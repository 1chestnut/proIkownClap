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
ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv"
ESC50_AUDIO_DIR = "/home/star/zkx/CLAP/data/ESC-50/audio"
KGE_MODEL_DIR = "/home/star/zkx/KGE_models/001/TFVpYwo2_RotatE_False"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TOP_K_CLAP = 10
TOP_M_PER_REL = 1
RELATIONS_Q = ['has parent', 'occurs in']  # 核心关系

def create_prompt(label, tail):
    return f"{label} {tail}"

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

def evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                  kge_model, triples_factory, agg_mode, alpha=None, temperature=None):
    hit1 = 0
    total = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        audio_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
        if not os.path.exists(audio_path):
            continue
        true_label = row['category']
        true_idx = class_to_idx[true_label]

        audio_embed = clap_model.get_audio_embedding_from_filelist([audio_path])
        audio_embed = torch.from_numpy(audio_embed).to(DEVICE)
        audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)

        sim = torch.matmul(audio_embed, text_embeds.T).squeeze(0).cpu().numpy()

        top_k_indices = np.argsort(sim)[::-1][:TOP_K_CLAP]
        top_k_labels = [class_labels[i] for i in top_k_indices]
        top_k_scores = sim[top_k_indices]
        new_scores = sim.copy()

        for idx, (label, orig_score) in enumerate(zip(top_k_labels, top_k_scores)):
            enhanced_scores = []
            for rel in RELATIONS_Q:
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
                scores = torch.tensor(enhanced_scores + [orig_score], device=DEVICE) * temperature
                final_score = torch.logsumexp(scores, dim=0).item() / temperature
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

    return hit1 / total * 100

def main():
    print("🚀 加载 CLAP 模型...")
    clap_model = CLAP_Module(enable_fusion=True, amodel='HTSAT-tiny', tmodel='roberta')
    clap_model.load_ckpt(MODEL_PATH)
    clap_model.to(DEVICE)
    clap_model.eval()

    triples_factory, kge_model = load_kge()

    df = pd.read_csv(ESC50_CSV)
    unique_categories = sorted(df['category'].unique())
    class_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    class_labels = [cat.replace('_', ' ') for cat in unique_categories]

    print("📝 预计算类别文本嵌入...")
    prompt_list = [f"This is a sound of {label}." for label in class_labels]
    text_embeds = clap_model.get_text_embedding(prompt_list)
    text_embeds = torch.from_numpy(text_embeds).to(DEVICE)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    test_df = df[df['fold'] == 1]
    print(f"\n📌 测试 Fold 1 (样本数: {len(test_df)})")

    # 基线
    baseline_hit1 = evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                                   kge_model, triples_factory, agg_mode='max')  # 实际上我们只是利用该函数但mode无关紧要，但为了一致，我们单独计算基线
    # 重新计算基线（直接用原始相似度）
    baseline = 0
    for _, row in test_df.iterrows():
        audio_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
        if not os.path.exists(audio_path):
            continue
        true_label = row['category']
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

    print("\n========== 探索 kg_agg 参数 ==========")

    # 测试不同 alpha 的加权平均
    for alpha in [0.7, 0.8, 0.9, 0.95]:
        hit1 = evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                              kge_model, triples_factory, agg_mode='weighted_mean', alpha=alpha)
        print(f"加权平均 (alpha={alpha}): Hit@1 = {hit1:.2f}%")

    # 测试 max 聚合
    hit1 = evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                          kge_model, triples_factory, agg_mode='max')
    print(f"max 聚合: Hit@1 = {hit1:.2f}%")

    # 测试 mean 聚合
    hit1 = evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                          kge_model, triples_factory, agg_mode='mean')
    print(f"mean 聚合: Hit@1 = {hit1:.2f}%")

    # 测试 logsumexp 不同温度
    for temp in [5, 10, 20, 50]:
        hit1 = evaluate_fold(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                              kge_model, triples_factory, agg_mode='logsumexp', temperature=temp)
        print(f"logsumexp (温度={temp}): Hit@1 = {hit1:.2f}%")

if __name__ == "__main__":
    main()