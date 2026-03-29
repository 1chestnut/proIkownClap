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
US8K_CSV = "/home/star/zkx/iknow-audio/data/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
US8K_AUDIO_ROOT = "/home/star/zkx/iknow-audio/data/UrbanSound8K/UrbanSound8K/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TOP_K_CLAP = 10
TOP_M_PER_REL = 1

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

def evaluate_fold_debug(test_df, clap_model, text_embeds, class_labels, class_to_idx,
                        kge_model, triples_factory, relations_q, agg_mode, alpha=None, temperature=None):
    hit1 = 0
    total = 0
    skipped = 0
    debug_printed = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        fold = row['fold']
        filename = row['slice_file_name']
        audio_path = os.path.join(US8K_AUDIO_ROOT, f'fold{fold}', filename)
        if not os.path.exists(audio_path):
            skipped += 1
            continue

        true_label = row['class']
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
        candidate_details = []

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
                scores = torch.tensor(enhanced_scores + [orig_score], device=DEVICE) * temperature
                final_score = torch.logsumexp(scores, dim=0).item() / temperature
            else:
                final_score = orig_score
            new_scores[top_k_indices[idx]] = final_score
            candidate_details.append((label, orig_score, final_score, enhanced_scores))

        ranked_indices = np.argsort(new_scores)[::-1]
        try:
            rank = np.where(ranked_indices == true_idx)[0][0] + 1
        except:
            rank = len(class_labels) + 1

        if debug_printed < 3:
            debug_printed += 1
            print(f"\n========== 调试样本 {debug_printed} ==========")
            print(f"真实类别: {true_label}")
            print("原始 top-10 候选 (标签, 原始得分):")
            for i, (lbl, sc) in enumerate(zip(top_k_labels, top_k_scores)):
                print(f"  {i+1}. {lbl}: {sc:.4f}")
            print("各候选的增强详情:")
            for cd in candidate_details:
                lbl, orig, final, enh_scores = cd
                print(f" 候选: {lbl}")
                print(f"   原始得分: {orig:.4f}, 最终得分: {final:.4f}")
                if enh_scores:
                    print(f"   增强得分均值: {np.mean(enh_scores):.4f}, 最大值: {max(enh_scores):.4f}, 数量: {len(enh_scores)}")
                else:
                    print("   无增强提示")
            print(f"真实类别排名: {rank}")

        if rank == 1:
            hit1 += 1
        total += 1

    if skipped > 0:
        print(f"\n⚠️ 跳过 {skipped} 个损坏或缺失的文件")
    return hit1 / total * 100

def main():
    print("🚀 加载 CLAP 模型...")
    clap_model = CLAP_Module(enable_fusion=True, amodel='HTSAT-tiny', tmodel='roberta')
    clap_model.load_ckpt(MODEL_PATH)
    clap_model.to(DEVICE)
    clap_model.eval()

    triples_factory, kge_model = load_kge()

    # ========== 打印 KG 覆盖率 ==========
    entity_to_id = triples_factory.entity_to_id
    kg_entities = set(entity_to_id.keys())
    print(f"KG 实体总数: {len(kg_entities)}")
    print("KG 实体示例（前20个）:", list(kg_entities)[:20])

    # 读取 US8K 元数据
    df = pd.read_csv(US8K_CSV)
    unique_classes = sorted(df['class'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    class_labels = [cls.replace('_', ' ') for cls in unique_classes]

    # 覆盖率统计
    matched = 0
    unmatched = []
    for cls in unique_classes:
        norm_cls = normalize_entity_name(cls)
        if norm_cls in kg_entities:
            matched += 1
        else:
            unmatched.append((cls, norm_cls))
    print(f"US8K 类别总数: {len(unique_classes)}")
    print(f"在 KG 中找到的类别数: {matched} ({matched/len(unique_classes)*100:.1f}%)")
    if unmatched:
        print("未匹配的类别示例 (原始名 -> 标准化后):")
        for orig, norm in unmatched:
            print(f"  {orig} -> {norm}")

    print("📝 预计算类别文本嵌入...")
    prompt_list = [f"This is a sound of {label}." for label in class_labels]
    text_embeds = clap_model.get_text_embedding(prompt_list)
    text_embeds = torch.from_numpy(text_embeds).to(DEVICE)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # 只测试 Fold 1
    test_df = df[df['fold'] == 1]
    print(f"\n📌 测试 Fold 1 (样本数: {len(test_df)})")

    # 计算基线
    baseline = 0
    for _, row in test_df.iterrows():
        audio_path = os.path.join(US8K_AUDIO_ROOT, f'fold{row["fold"]}', row['slice_file_name'])
        if not os.path.exists(audio_path):
            continue
        true_label = row['class']
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

    # 使用最佳候选关系进行调试（根据之前探索）
    best_rel = ['caused by', 'co-occurs with', 'associated with environment']
    alpha_best = 0.9
    print(f"\n========== 调试模式: 关系 {best_rel}, 加权平均 alpha={alpha_best} ==========")
    hit1 = evaluate_fold_debug(
        test_df, clap_model, text_embeds, class_labels, class_to_idx,
        kge_model, triples_factory, best_rel,
        agg_mode='weighted_mean', alpha=alpha_best
    )
    print(f"\n调试模式 Hit@1: {hit1:.2f}%")

    # 可选：测试其他聚合方式
    print("\n========== 尝试其他聚合方式 ==========")
    # max 聚合
    hit1_max = evaluate_fold_debug(
        test_df, clap_model, text_embeds, class_labels, class_to_idx,
        kge_model, triples_factory, best_rel,
        agg_mode='max'
    )
    print(f"max 聚合 Hit@1: {hit1_max:.2f}%")
    # logsumexp 温度20
    hit1_lse = evaluate_fold_debug(
        test_df, clap_model, text_embeds, class_labels, class_to_idx,
        kge_model, triples_factory, best_rel,
        agg_mode='logsumexp', temperature=20
    )
    print(f"logsumexp 温度20 Hit@1: {hit1_lse:.2f}%")

if __name__ == "__main__":
    main()