# -*- coding: utf-8 -*-
import importlib.util
import inspect
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


BASE_FILENAMES = {
    "esc50": "esc_router_base.py",
    "tut2017": "tut_base.py",
    "dcase": "dcase_base.py",
    "fsd50": "fsd_base.py",
    "usk80": "usk_base.py",
    "audioset": "audioset_base.py",
}

FEATURE_COLS = [
    "baseline_top1",
    "baseline_margin",
    "entropy",
    "hop2_activation",
    "prompt_count_log1p",
    "base_i",
    "orig_minus_base_i",
    "ours_minus_base_i",
    "orig_minus_ours_i",
]


def load_base(module_path):
    spec = importlib.util.spec_from_file_location("oracle38_base_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_audio_embedding(base, clap_model, audio_path, device):
    if hasattr(base, "get_audio_embedding_safe"):
        return base.get_audio_embedding_safe(clap_model, audio_path, device)
    emb = clap_model.get_audio_embeddings([audio_path])
    if hasattr(base, "to_tensor"):
        return base.to_tensor(emb).to(device).float()
    if isinstance(emb, torch.Tensor):
        return emb.to(device).float()
    return torch.from_numpy(emb).to(device).float()


def compute_alpha_dynamic(base, cos_sim_orig):
    max_sim = float(torch.max(cos_sim_orig).item())
    if hasattr(base, "static_alpha_from_maxsim"):
        return float(base.static_alpha_from_maxsim(max_sim))
    if hasattr(base, "instance_alpha"):
        return float(base.instance_alpha(max_sim))
    alpha_min = float(getattr(base, "ALPHA_MIN", 0.35))
    alpha_max = float(getattr(base, "ALPHA_MAX", 0.75))
    alpha = alpha_min + (alpha_max - alpha_min) * max_sim
    return float(max(alpha_min, min(alpha_max, alpha)))


def call_build_h1_map(base, kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map):
    sig = inspect.signature(base.build_h1_map)
    if "use_llm" in sig.parameters:
        return base.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map, True)
    return base.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map)


def call_build_h2_prompts(base, h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map):
    sig = inspect.signature(base.build_h2_prompts)
    if "use_llm" in sig.parameters:
        return base.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map, True)
    return base.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map)


def run_ours_head(
    base,
    clap_model,
    audio_embed,
    cos_sim_orig,
    top_indices,
    label_classes,
    kg_classes,
    class_labels_set,
    hop1_relations,
    hop2_relations,
    get_tails,
    prompt_map,
    alpha_dynamic,
):
    if hasattr(base, "run_selective2hop"):
        ret = base.run_selective2hop(
            clap_model,
            audio_embed,
            cos_sim_orig,
            top_indices,
            label_classes,
            kg_classes,
            class_labels_set,
            hop1_relations,
            hop2_relations,
            get_tails,
            prompt_map,
            alpha_dynamic,
        )
        if len(ret) == 4:
            score, prompts, extra, _ = ret
        else:
            score, prompts, extra = ret
        return score, prompts, extra

    if hasattr(base, "run_selective_variant"):
        score, prompts, extra = base.run_selective_variant(
            clap_model,
            audio_embed,
            cos_sim_orig,
            top_indices,
            label_classes,
            kg_classes,
            class_labels_set,
            hop1_relations,
            hop2_relations,
            get_tails,
            prompt_map,
            alpha_dynamic,
            True,
            "dynamic",
        )
        return score, prompts, extra

    raise AttributeError("Base module has neither run_selective2hop nor run_selective_variant")


def score_gap(score_vec, true_idx):
    true_score = float(score_vec[true_idx].item())
    competitor = score_vec.clone()
    competitor[true_idx] = -1e9
    best_other = float(torch.max(competitor).item())
    return true_score - best_other


def normalized_entropy(score_vec, top_indices):
    if len(top_indices) <= 1:
        return 0.0
    vals = score_vec[top_indices]
    probs = torch.softmax(vals, dim=0)
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
    return float(ent / np.log(len(top_indices)))


def candidate_originalagg_score(
    base,
    clap_model,
    audio_embed,
    cos_sim_orig,
    ci,
    label_classes,
    kg_classes,
    class_labels_set,
    hop1_relations,
    hop2_relations,
    get_tails,
    prompt_map,
):
    class_name = label_classes[ci]
    kg_ent = base.get_kg_entity(kg_classes[ci])
    tau = cos_sim_orig[ci].item() + base.RELATIVE_MARGIN
    h1_map = call_build_h1_map(base, kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map)
    s1 = base.score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
    max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
    if s1.numel() > 0 and max_h1 >= tau:
        logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * base.LOGIT_SCALE, s1 * base.LOGIT_SCALE])
        score = (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / base.LOGIT_SCALE
        return score, len(h1_map), False
    h2_prompts = call_build_h2_prompts(base, h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map)
    s2 = base.score_prompt_list(clap_model, audio_embed, h2_prompts)
    all_scores = torch.cat([s1, s2 * base.DECAY_GAMMA]) if s2.numel() > 0 else s1
    if all_scores.numel() == 0:
        return cos_sim_orig[ci], len(h1_map) + len(h2_prompts), True
    logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * base.LOGIT_SCALE, all_scores * base.LOGIT_SCALE])
    score = (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / base.LOGIT_SCALE
    return score, len(h1_map) + len(h2_prompts), True


def run_originalagg(
    base,
    clap_model,
    audio_embed,
    cos_sim_orig,
    top_indices,
    label_classes,
    kg_classes,
    class_labels_set,
    hop1_relations,
    hop2_relations,
    get_tails,
    prompt_map,
):
    score = cos_sim_orig.clone()
    prompt_count = 0
    hop2_flags = []
    for ci in top_indices:
        s_i, used_prompts, hop2_flag = candidate_originalagg_score(
            base,
            clap_model,
            audio_embed,
            cos_sim_orig,
            ci,
            label_classes,
            kg_classes,
            class_labels_set,
            hop1_relations,
            hop2_relations,
            get_tails,
            prompt_map,
        )
        score[ci] = s_i
        prompt_count += used_prompts
        hop2_flags.append(hop2_flag)
    extras = {
        "hop2_activated": any(hop2_flags),
        "candidate_level_activation_rate": float(np.mean(hop2_flags)) if hop2_flags else 0.0,
    }
    return score, prompt_count, extras


def choose_candidate_oracle(base, orig_score, ours_score, top_indices, true_indices):
    true_idx = int(true_indices[0])
    best_key = None
    best = None
    for mask_bits in np.ndindex(*(2 for _ in range(len(top_indices)))):
        final_score = ours_score.clone()
        choose_orig = 0
        for ci, bit in zip(top_indices, mask_bits):
            if bit == 1:
                final_score[ci] = orig_score[ci]
                choose_orig += 1
            else:
                final_score[ci] = ours_score[ci]
        rank = base.rank_of_true(final_score, true_indices)
        gap = score_gap(final_score, true_idx)
        key = (-rank, gap)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "score": final_score,
                "choose_orig_ratio": float(choose_orig / max(1, len(top_indices))),
                "mask_bits": list(mask_bits),
                "gap": gap,
            }
    return best


def init_results():
    return {
        "Selective2Hop_OriginalAgg": {"ranks": [], "times": [], "prompts": [], "route_choose_orig": []},
        "Selective2Hop_Ours": {"ranks": [], "times": [], "prompts": [], "route_choose_orig": []},
        "OracleCandidate": {"ranks": [], "times": [], "prompts": [], "route_choose_orig": []},
        "OracleGuidedCandidate": {"ranks": [], "times": [], "prompts": [], "route_choose_orig": []},
    }


def compute_key_candidates(top_indices, baseline_scores, orig_scores, ours_scores, true_indices=None, include_true=False):
    top_indices = list(top_indices)
    baseline_top = top_indices[: min(3, len(top_indices))]
    orig_sorted = sorted(top_indices, key=lambda i: float(orig_scores[i].item()), reverse=True)[: min(3, len(top_indices))]
    ours_sorted = sorted(top_indices, key=lambda i: float(ours_scores[i].item()), reverse=True)[: min(3, len(top_indices))]
    disagreement_sorted = sorted(
        top_indices,
        key=lambda i: abs(float(orig_scores[i].item()) - float(ours_scores[i].item())),
        reverse=True,
    )[: min(2, len(top_indices))]
    chosen = set(baseline_top + orig_sorted + ours_sorted + disagreement_sorted)
    if include_true and true_indices is not None:
        chosen.update(int(i) for i in true_indices if int(i) in top_indices)
    return sorted(chosen)


def print_tables(base, results, title):
    metrics = {name: base.compute_metrics(results[name]["ranks"]) for name in results}
    order = ["Selective2Hop_OriginalAgg", "Selective2Hop_Ours", "OracleCandidate", "OracleGuidedCandidate"]
    print("\n" + "=" * 160)
    print(title)
    print("-" * 160)
    print(f"{'Metric':<10} | {'OriginalAgg':<14} | {'Ours':<14} | {'OracleCandidate':<18} | {'OracleGuided':<18}")
    print("-" * 160)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(
            f"{metric_name:<10} | "
            f"{metrics['Selective2Hop_OriginalAgg'][idx]:<14.2f} | "
            f"{metrics['Selective2Hop_Ours'][idx]:<14.2f} | "
            f"{metrics['OracleCandidate'][idx]:<18.2f} | "
            f"{metrics['OracleGuidedCandidate'][idx]:<18.2f}"
        )
    print("\n" + "=" * 170)
    print(f"{'Method':<24} | {'Avg prompts':<12} | {'Avg time (ms)':<14} | {'Hit@1':<8} | {'MRR':<8} | {'Route->Orig':<12}")
    print("-" * 170)
    for name in order:
        route_rate = "N/A"
        if "route_choose_orig" in results[name]:
            route_rate = f"{np.mean(results[name]['route_choose_orig']) * 100:.1f}%"
        print(
            f"{name:<24} | {np.mean(results[name]['prompts']):<12.1f} | {np.mean(results[name]['times']):<14.1f} | "
            f"{metrics[name][0]:<8.2f} | {metrics[name][3]:<8.2f} | {route_rate:<12}"
        )


def save_results(base, results, output_json, meta):
    metrics = {name: base.compute_metrics(results[name]["ranks"]) for name in results}
    payload = {"metrics": {}, "stats": {}, "meta": meta}
    for name in results:
        payload["metrics"][name] = {
            "Hit@1": metrics[name][0],
            "Hit@3": metrics[name][1],
            "Hit@5": metrics[name][2],
            "MRR": metrics[name][3],
        }
        payload["stats"][name] = {
            "avg_prompts": float(np.mean(results[name]["prompts"])) if results[name]["prompts"] else 0.0,
            "avg_time_ms": float(np.mean(results[name]["times"])) if results[name]["times"] else 0.0,
            "route_choose_orig": float(np.mean(results[name]["route_choose_orig"])) if results[name]["route_choose_orig"] else 0.0,
        }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_or_create_records(dataset_name, module_path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cand_csv = os.path.join(cache_dir, "candidate_records.csv")
    sample_csv = os.path.join(cache_dir, "sample_records.csv")
    if os.path.exists(cand_csv) and os.path.exists(sample_csv):
        return pd.read_csv(cand_csv), pd.read_csv(sample_csv)

    base = load_base(module_path)
    print(f"[Collect] {dataset_name}: building oracle-guided training cache")
    prompt_map = base.load_llm_prompts(base.LLM_PROMPTS_PATH)
    clap_model = base.CLAP(version="2023", use_cuda=torch.cuda.is_available())
    with base.warnings.catch_warnings():
        base.warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(base.KGE_MODEL_DIR, "trained_model.pkl"), map_location=base.DEVICE)
    kge_model.eval()
    training_factory = base.TriplesFactory.from_path(base.TRAIN_TRIPLES_PATH)
    hop1_relations = [rel for rel in base.HOP1_RELATIONS if rel in training_factory.relation_to_id]
    hop2_relations = [rel for rel in base.HOP2_RELATIONS if rel in training_factory.relation_to_id]
    get_tails = base.build_tail_predictor(kge_model, training_factory)
    dataset = base.load_dataset()
    label_classes = dataset["label_classes"]
    kg_classes = dataset["kg_classes"]
    class_labels_set = dataset["class_labels_set"]
    text_embeds = F.normalize(base.get_safe_text_embeddings(clap_model, label_classes, base.DEVICE), dim=-1)

    candidate_records = []
    sample_records = []
    for sample in tqdm(base.iter_samples(dataset), total=dataset["total"], desc=f"Collect-{dataset_name}"):
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]
        if not os.path.exists(audio_path) or not true_indices:
            continue
        try:
            audio_embed = F.normalize(get_audio_embedding(base, clap_model, audio_path, base.DEVICE), dim=-1)
        except Exception:
            continue
        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[: base.TOP_K]
        alpha_dynamic = compute_alpha_dynamic(base, cos_sim_orig)
        orig_score, orig_prompts, orig_extra = run_originalagg(
            base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map,
        )
        ours_score, _, ours_extra = run_ours_head(
            base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic
        )
        candidate_oracle = choose_candidate_oracle(base, orig_score, ours_score, top_indices, true_indices)
        mask_bits = candidate_oracle["mask_bits"]
        base_top1 = float(cos_sim_orig[top_indices[0]].item())
        base_margin = float(cos_sim_orig[top_indices[0]].item() - cos_sim_orig[top_indices[1]].item()) if len(top_indices) > 1 else 0.0
        ent = normalized_entropy(cos_sim_orig, top_indices)
        key_candidates = compute_key_candidates(
            top_indices=top_indices,
            baseline_scores=cos_sim_orig,
            orig_scores=orig_score,
            ours_scores=ours_score,
            true_indices=true_indices,
            include_true=True,
        )
        sample_records.append(
            {
                "dataset_name": dataset_name,
                "sample_id": audio_path,
                "true_label": label_classes[int(true_indices[0])],
                "baseline_top1": base_top1,
                "baseline_margin": base_margin,
                "entropy": ent,
                "hop2_activation": float(orig_extra["hop2_activated"]),
                "prompt_count_log1p": float(np.log1p(orig_prompts)),
                "oracle_choose_orig_ratio": float(candidate_oracle["choose_orig_ratio"]),
                "key_candidate_count": int(len(key_candidates)),
            }
        )
        for ci, bit in zip(top_indices, mask_bits):
            if int(ci) not in set(key_candidates):
                continue
            candidate_records.append(
                {
                    "dataset_name": dataset_name,
                    "sample_id": audio_path,
                    "true_label": label_classes[int(true_indices[0])],
                    "candidate_index": int(ci),
                    "candidate_label": label_classes[int(ci)],
                    "is_true_candidate": 1.0 if int(ci) in set(int(x) for x in true_indices) else 0.0,
                    "choose_orig": float(bit),
                    "baseline_top1": base_top1,
                    "baseline_margin": base_margin,
                    "entropy": ent,
                    "hop2_activation": float(orig_extra["hop2_activated"]),
                    "prompt_count_log1p": float(np.log1p(orig_prompts)),
                    "base_i": float(cos_sim_orig[ci].item()),
                    "orig_minus_base_i": float(orig_score[ci].item() - cos_sim_orig[ci].item()),
                    "ours_minus_base_i": float(ours_score[ci].item() - cos_sim_orig[ci].item()),
                    "orig_minus_ours_i": float(orig_score[ci].item() - ours_score[ci].item()),
                }
            )

    cand_df = pd.DataFrame(candidate_records)
    sample_df = pd.DataFrame(sample_records)
    cand_df.to_csv(cand_csv, index=False, encoding="utf-8-sig")
    sample_df.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    return cand_df, sample_df


def fit_router(train_df):
    x_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["choose_orig"].to_numpy(dtype=np.int32)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = LogisticRegression(
        random_state=42,
        max_iter=300,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(x_train_scaled, y_train)
    meta = {
        "class_balance": {str(k): int(v) for k, v in Counter(y_train.tolist()).items()},
        "coef_mean": model.coef_[0].tolist(),
        "coef_abs_mean": np.abs(model.coef_[0]).tolist(),
    }
    return scaler, model, meta


@torch.no_grad()
def evaluate_target(target_name, target_module_path, cache_root, scaler, model, launch_title, output_json):
    base = load_base(target_module_path)
    print(launch_title)
    print(f"Device: {base.DEVICE}")
    print("Oracle-guided candidate router: leave-one-dataset-out target evaluation")

    prompt_map = base.load_llm_prompts(base.LLM_PROMPTS_PATH)
    clap_model = base.CLAP(version="2023", use_cuda=torch.cuda.is_available())
    with base.warnings.catch_warnings():
        base.warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(base.KGE_MODEL_DIR, "trained_model.pkl"), map_location=base.DEVICE)
    kge_model.eval()
    training_factory = base.TriplesFactory.from_path(base.TRAIN_TRIPLES_PATH)
    hop1_relations = [rel for rel in base.HOP1_RELATIONS if rel in training_factory.relation_to_id]
    hop2_relations = [rel for rel in base.HOP2_RELATIONS if rel in training_factory.relation_to_id]
    get_tails = base.build_tail_predictor(kge_model, training_factory)
    dataset = base.load_dataset()
    label_classes = dataset["label_classes"]
    kg_classes = dataset["kg_classes"]
    class_labels_set = dataset["class_labels_set"]
    text_embeds = F.normalize(base.get_safe_text_embeddings(clap_model, label_classes, base.DEVICE), dim=-1)

    results = init_results()
    oracle_match = []
    oracle_match_true = []
    oracle_match_hard_negative = []
    prob_values = []
    true_gap_improved = []
    top1_corrected = []
    top1_changed = []
    trainable_ratio = []

    for sample in tqdm(base.iter_samples(dataset), total=dataset["total"], desc=f"OGC-{target_name}"):
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]
        if not os.path.exists(audio_path) or not true_indices:
            continue
        try:
            t_audio = time.time()
            audio_embed = F.normalize(get_audio_embedding(base, clap_model, audio_path, base.DEVICE), dim=-1)
            audio_ms = (time.time() - t_audio) * 1000.0
        except Exception:
            continue

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[: base.TOP_K]
        alpha_dynamic = compute_alpha_dynamic(base, cos_sim_orig)

        t_orig = time.time()
        orig_score, orig_prompts, orig_extra = run_originalagg(
            base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map
        )
        orig_ms = audio_ms + (time.time() - t_orig) * 1000.0

        t_ours = time.time()
        ours_score, ours_prompts, ours_extra = run_ours_head(
            base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic
        )
        ours_ms = audio_ms + (time.time() - t_ours) * 1000.0

        oracle = choose_candidate_oracle(base, orig_score, ours_score, top_indices, true_indices)
        key_candidates = compute_key_candidates(
            top_indices=top_indices,
            baseline_scores=cos_sim_orig,
            orig_scores=orig_score,
            ours_scores=ours_score,
            true_indices=None,
            include_true=False,
        )
        trainable_ratio.append(float(len(key_candidates) / max(1, len(top_indices))))

        final_score = ours_score.clone()
        route_bits = {}
        base_top1 = float(cos_sim_orig[top_indices[0]].item())
        base_margin = float(cos_sim_orig[top_indices[0]].item() - cos_sim_orig[top_indices[1]].item()) if len(top_indices) > 1 else 0.0
        ent = normalized_entropy(cos_sim_orig, top_indices)
        for ci in key_candidates:
            row = np.array([[
                base_top1,
                base_margin,
                ent,
                float(orig_extra["hop2_activated"]),
                float(np.log1p(orig_prompts)),
                float(cos_sim_orig[ci].item()),
                float(orig_score[ci].item() - cos_sim_orig[ci].item()),
                float(ours_score[ci].item() - cos_sim_orig[ci].item()),
                float(orig_score[ci].item() - ours_score[ci].item()),
            ]], dtype=np.float32)
            p = float(model.predict_proba(scaler.transform(row))[0, 1])
            prob_values.append(p)
            choose_orig = 1 if p > 0.5 else 0
            route_bits[int(ci)] = choose_orig
            if choose_orig == 1:
                final_score[ci] = orig_score[ci]
            else:
                final_score[ci] = ours_score[ci]

        oracle_mask = {int(ci): int(bit) for ci, bit in zip(top_indices, oracle["mask_bits"])}
        for ci in key_candidates:
            oracle_match.append(float(route_bits[int(ci)] == oracle_mask[int(ci)]))
            if int(ci) in set(int(x) for x in true_indices):
                oracle_match_true.append(float(route_bits[int(ci)] == oracle_mask[int(ci)]))
            else:
                if int(ci) in set(top_indices[:2]):
                    oracle_match_hard_negative.append(float(route_bits[int(ci)] == oracle_mask[int(ci)]))

        true_idx = int(true_indices[0])
        orig_gap = score_gap(orig_score, true_idx)
        router_gap = score_gap(final_score, true_idx)
        true_gap_improved.append(float(router_gap >= orig_gap))
        top1_corrected.append(float(base.rank_of_true(final_score, true_indices) < base.rank_of_true(orig_score, true_indices)))
        top1_changed.append(float(torch.argmax(final_score).item() != torch.argmax(orig_score).item()))

        results["Selective2Hop_OriginalAgg"]["ranks"].append(base.rank_of_true(orig_score, true_indices))
        results["Selective2Hop_OriginalAgg"]["times"].append(orig_ms)
        results["Selective2Hop_OriginalAgg"]["prompts"].append(orig_prompts)
        results["Selective2Hop_OriginalAgg"]["route_choose_orig"].append(1.0)

        results["Selective2Hop_Ours"]["ranks"].append(base.rank_of_true(ours_score, true_indices))
        results["Selective2Hop_Ours"]["times"].append(ours_ms)
        results["Selective2Hop_Ours"]["prompts"].append(ours_prompts)
        results["Selective2Hop_Ours"]["route_choose_orig"].append(0.0)

        results["OracleCandidate"]["ranks"].append(base.rank_of_true(oracle["score"], true_indices))
        results["OracleCandidate"]["times"].append(max(orig_ms, ours_ms))
        results["OracleCandidate"]["prompts"].append(orig_prompts)
        results["OracleCandidate"]["route_choose_orig"].append(float(oracle["choose_orig_ratio"]))

        results["OracleGuidedCandidate"]["ranks"].append(base.rank_of_true(final_score, true_indices))
        results["OracleGuidedCandidate"]["times"].append(max(orig_ms, ours_ms))
        results["OracleGuidedCandidate"]["prompts"].append(orig_prompts)
        results["OracleGuidedCandidate"]["route_choose_orig"].append(float(np.mean(list(route_bits.values()))) if route_bits else 0.0)

    metrics = {name: base.compute_metrics(results[name]["ranks"]) for name in results}
    print_tables(base, results, launch_title + " Final Tables")

    diagnostics = {
        "oracle_match_accuracy": float(np.mean(oracle_match)) if oracle_match else 0.0,
        "true_candidate_match_accuracy": float(np.mean(oracle_match_true)) if oracle_match_true else 0.0,
        "hard_negative_match_accuracy": float(np.mean(oracle_match_hard_negative)) if oracle_match_hard_negative else 0.0,
        "true_gap_improved_ratio": float(np.mean(true_gap_improved)) if true_gap_improved else 0.0,
        "top1_corrected_ratio": float(np.mean(top1_corrected)) if top1_corrected else 0.0,
        "top1_changed_ratio": float(np.mean(top1_changed)) if top1_changed else 0.0,
        "trainable_candidate_ratio": float(np.mean(trainable_ratio)) if trainable_ratio else 0.0,
        "prob_low_ratio": float(np.mean(np.array(prob_values) < 0.3)) if prob_values else 0.0,
        "prob_mid_ratio": float(np.mean((np.array(prob_values) >= 0.45) & (np.array(prob_values) <= 0.55))) if prob_values else 0.0,
        "prob_high_ratio": float(np.mean(np.array(prob_values) > 0.7)) if prob_values else 0.0,
    }
    print("\nDiagnostics")
    for k, v in diagnostics.items():
        print(f"{k}: {v}")

    payload_meta = {
        "router_type": "oracle_guided_candidate_leave_one_dataset_out",
        "target_dataset": target_name,
        "feature_cols": FEATURE_COLS,
        "diagnostics": diagnostics,
        "metrics": {name: {"Hit@1": metrics[name][0], "MRR": metrics[name][3]} for name in metrics},
    }
    save_results(base, results, output_json, payload_meta)


def run_oracle_guided_candidate(root_dir, target_name, output_dir, launch_title):
    bases_dir = os.path.join(root_dir, "bases")
    cache_dir = os.path.join(root_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    source_frames = []
    for ds_name, filename in BASE_FILENAMES.items():
        module_path = os.path.join(bases_dir, filename)
        if ds_name == target_name:
            continue
        ds_cache = os.path.join(cache_dir, ds_name)
        cand_df, _ = load_or_create_records(ds_name, module_path, ds_cache)
        if not cand_df.empty:
            source_frames.append(cand_df)

    if not source_frames:
        raise RuntimeError("No training records collected for oracle-guided candidate router.")

    train_df = pd.concat(source_frames, ignore_index=True)
    scaler, model, model_meta = fit_router(train_df)

    target_module_path = os.path.join(bases_dir, BASE_FILENAMES[target_name])
    evaluate_target(
        target_name=target_name,
        target_module_path=target_module_path,
        cache_root=cache_dir,
        scaler=scaler,
        model=model,
        launch_title=launch_title,
        output_json=os.path.join(output_dir, "results_oracle_guided_candidate.json"),
    )
