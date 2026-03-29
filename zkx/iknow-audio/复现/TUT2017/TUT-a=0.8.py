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
# 最佳参数
RELATIONS_Q = ['has parent', 'occurs in', 'co-occurs with']
ALPHA = 0.8   # 加权平均权重

# TUT2017 场景类别（15类）
SCENE_CLASSES = [
    'beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
    'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
    'office', 'park', 'residential_area', 'train', 'tram'
]
# 用于 prompt 的标签（下划线转空格）
PROMPT_LABELS = [c.replace('_', ' ') for c in SCENE_CLASSES]
CLASS_TO_IDX = {c: i for i, c in enumerate(SCENE_CLASSES)}

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

def evaluate_fold(test_df, clap_model, text_embeds, mode):
    hit1 = hit3 = hit5 = 0
    rr_sum = 0.0
    total = 0
    skipped = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        filename = row['filename'].strip()
        if filename.startswith('audio/'):
            rel_path = filename
        else:
            rel_path = f"audio/{filename}"
        audio_path = os.path.join(TUT_DEV_ROOT, rel_path)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(TUT_EVAL_ROOT, rel_path)
        if not os.path.exists(audio_path):
            skipped += 1
            continue

        true_label = row['scene_label']
        true_idx = CLASS_TO_IDX[true_label]

        try:
            audio_embed = clap_model.get_audio_embedding_from_filelist([audio_path])
            audio_embed = torch.from_numpy(audio_embed).to(DEVICE)
            audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
        except Exception as e:
            print(f"\n警告：跳过文件 {audio_path}，原因：{e}")
            skipped += 1
            continue

        sim = torch.matmul(audio_embed, text_embeds.T).squeeze(0).cpu().numpy()

        if mode == 'baseline':
            ranked_indices = np.argsort(sim)[::-1]
            try:
                rank = np.where(ranked_indices == true_idx)[0][0] + 1
            except:
                rank = len(SCENE_CLASSES) + 1
        else:  # mode == 'kg_agg'
            top_k_indices = np.argsort(sim)[::-1][:TOP_K_CLAP]
            top_k_labels = [PROMPT_LABELS[i] for i in top_k_indices]
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

                if enhanced_scores:
                    avg_enhanced = np.mean(enhanced_scores)
                    final_score = ALPHA * orig_score + (1 - ALPHA) * avg_enhanced
                else:
                    final_score = orig_score
                new_scores[top_k_indices[idx]] = final_score

            ranked_indices = np.argsort(new_scores)[::-1]
            try:
                rank = np.where(ranked_indices == true_idx)[0][0] + 1
            except:
                rank = len(SCENE_CLASSES) + 1

        if rank == 1:
            hit1 += 1
        if rank <= 3:
            hit3 += 1
        if rank <= 5:
            hit5 += 1
        rr_sum += 1.0 / rank
        total += 1

    if skipped > 0:
        print(f"\n⚠️ 跳过 {skipped} 个缺失文件")
    return hit1, hit3, hit5, rr_sum, total

def main():
    print("🚀 加载 CLAP 模型...")
    clap_model = CLAP_Module(enable_fusion=True, amodel='HTSAT-tiny', tmodel='roberta')
    clap_model.load_ckpt(MODEL_PATH)
    clap_model.to(DEVICE)
    clap_model.eval()

    global kge_model, triples_factory
    triples_factory, kge_model = load_kge()

    print("📝 预计算类别文本嵌入...")
    prompt_list = [f"This is a sound of {label}." for label in PROMPT_LABELS]
    text_embeds = clap_model.get_text_embedding(prompt_list)
    text_embeds = torch.from_numpy(text_embeds).to(DEVICE)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # ========== 开发集 4 折交叉验证 ==========
    folds = [1, 2, 3, 4]
    print(f"🔁 使用 {len(folds)} 折交叉验证...")
    results_dev = {}
    for mode in ['baseline', 'kg_agg']:
        print(f"\n========== 开发集 {mode} 模式 ==========")
        total_hit1 = total_hit3 = total_hit5 = 0
        total_rr = 0.0
        total_samples = 0

        for fold in folds:
            fold_file = os.path.join(TUT_DEV_ROOT, 'evaluation_setup', f'fold{fold}_evaluate.txt')
            test_df = pd.read_csv(fold_file, sep='\t', header=None, names=['filename', 'scene_label'])
            print(f"\n📌 Fold {fold} (测试集 {len(test_df)} 个样本)")
            hit1, hit3, hit5, rr_sum, n = evaluate_fold(test_df, clap_model, text_embeds, mode=mode)
            total_hit1 += hit1
            total_hit3 += hit3
            total_hit5 += hit5
            total_rr += rr_sum
            total_samples += n
            print(f"   Hit@1: {hit1/n*100:.2f}%, Hit@3: {hit3/n*100:.2f}%, Hit@5: {hit5/n*100:.2f}%, MRR: {rr_sum/n:.4f}")

        avg_hit1 = total_hit1 / total_samples * 100
        avg_hit3 = total_hit3 / total_samples * 100
        avg_hit5 = total_hit5 / total_samples * 100
        avg_mrr = total_rr / total_samples
        results_dev[mode] = (avg_hit1, avg_hit3, avg_hit5, avg_mrr)

    print("\n" + "="*60)
    print("📊 TUT2017 开发集 4 折平均结果")
    print(f"{'Mode':<12} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'MRR':>8}")
    print("-"*60)
    for mode, (h1, h3, h5, mrr) in results_dev.items():
        print(f"{mode:<12} {h1:>7.2f}% {h3:>7.2f}% {h5:>7.2f}% {mrr:>7.4f}")
    print("="*60)

    # ========== 评估集测试 ==========
    print("\n========== 评估集测试 ==========")
    eval_file = os.path.join(TUT_EVAL_ROOT, 'evaluation_setup', 'evaluate.txt')
    eval_df = pd.read_csv(eval_file, sep='\t', header=None, names=['filename', 'scene_label'])
    print(f"\n📌 评估集 (样本数: {len(eval_df)})")

    results_eval = {}
    for mode in ['baseline', 'kg_agg']:
        hit1, hit3, hit5, rr_sum, n = evaluate_fold(eval_df, clap_model, text_embeds, mode=mode)
        print(f"\n{mode} 模式:")
        print(f"   Hit@1: {hit1/n*100:.2f}%, Hit@3: {hit3/n*100:.2f}%, Hit@5: {hit5/n*100:.2f}%, MRR: {rr_sum/n:.4f}")
        results_eval[mode] = (hit1/n*100, hit3/n*100, hit5/n*100, rr_sum/n)

    print("\n" + "="*60)
    print("📊 TUT2017 评估集结果")
    print(f"{'Mode':<12} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'MRR':>8}")
    print("-"*60)
    for mode, (h1, h3, h5, mrr) in results_eval.items():
        print(f"{mode:<12} {h1:>7.2f}% {h3:>7.2f}% {h5:>7.2f}% {mrr:>7.4f}")
    print("="*60)

if __name__ == "__main__":
    main()