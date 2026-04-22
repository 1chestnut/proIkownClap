# -*- coding: utf-8 -*-
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


N_FOLDS = 5
RANK_MARGIN = 0.005
DEFAULT_CLOSE_MARGIN = 0.01
KEY_TOP = 3
THRESHOLD_GRID = np.linspace(0.2, 0.8, 13)


def load_base(module_path):
    spec = importlib.util.spec_from_file_location("router_base_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    h1_map = base.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map)
    s1 = base.score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
    max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
    if s1.numel() > 0 and max_h1 >= tau:
        logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * base.LOGIT_SCALE, s1 * base.LOGIT_SCALE])
        score = (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / base.LOGIT_SCALE
        return score, len(h1_map), False
    h2_prompts = base.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map)
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


def init_results():
    return {
        "Baseline": {"ranks": [], "times": [], "prompts": []},
        "iKnow": {"ranks": [], "times": [], "prompts": []},
        "Full2Hop": {"ranks": [], "times": [], "prompts": []},
        "Selective2Hop_OriginalAgg": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "alphas": [],
        },
        "Selective2Hop_Ours": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "alphas": [],
        },
        "Selective2Hop_RankAwareRouter": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "alphas": [],
            "route_choose_orig": [],
        },
    }


def print_tables(base, results, title):
    metrics = {name: base.compute_metrics(results[name]["ranks"]) for name in results}
    order = [
        "Baseline",
        "iKnow",
        "Full2Hop",
        "Selective2Hop_OriginalAgg",
        "Selective2Hop_Ours",
        "Selective2Hop_RankAwareRouter",
    ]
    print("\n" + "=" * 180)
    print(title)
    print("-" * 180)
    print(
        f"{'Metric':<8} | {'Baseline':<10} | {'iKnow':<10} | {'Full2Hop':<12} | "
        f"{'Sel2 OriginalAgg':<18} | {'Sel2 Ours':<18} | {'Sel2 RankAware':<18}"
    )
    print("-" * 180)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(
            f"{metric_name:<8} | {metrics['Baseline'][idx]:<10.2f} | {metrics['iKnow'][idx]:<10.2f} | "
            f"{metrics['Full2Hop'][idx]:<12.2f} | {metrics['Selective2Hop_OriginalAgg'][idx]:<18.2f} | "
            f"{metrics['Selective2Hop_Ours'][idx]:<18.2f} | {metrics['Selective2Hop_RankAwareRouter'][idx]:<18.2f}"
        )
    print("\n" + "=" * 190)
    print(
        f"{'Method':<26} | {'Hop2 activation(sample)':<24} | {'Hop2 activation(candidate)':<27} | "
        f"{'Avg prompts':<12} | {'Avg time (ms)':<14} | {'Hit@1':<8} | {'MRR':<8} | {'Route->Orig':<12}"
    )
    print("-" * 190)
    for name in order:
        sample_rate = "N/A"
        candidate_rate = "N/A"
        route_rate = "N/A"
        if name.startswith("Selective2Hop"):
            sample_rate = f"{np.mean(results[name]['hop2_activation_sample']) * 100:.1f}%"
            candidate_rate = f"{np.mean(results[name]['hop2_activation_candidate']) * 100:.1f}%"
        if name == "Selective2Hop_RankAwareRouter":
            route_rate = f"{np.mean(results[name]['route_choose_orig']) * 100:.1f}%"
        print(
            f"{name:<26} | {sample_rate:<24} | {candidate_rate:<27} | {np.mean(results[name]['prompts']):<12.1f} | "
            f"{np.mean(results[name]['times']):<14.1f} | {metrics[name][0]:<8.2f} | {metrics[name][3]:<8.2f} | {route_rate:<12}"
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
        }
        if name.startswith("Selective2Hop"):
            payload["stats"][name]["hop2_activation_sample"] = (
                float(np.mean(results[name]["hop2_activation_sample"])) if results[name]["hop2_activation_sample"] else 0.0
            )
            payload["stats"][name]["hop2_activation_candidate"] = (
                float(np.mean(results[name]["hop2_activation_candidate"])) if results[name]["hop2_activation_candidate"] else 0.0
            )
            payload["stats"][name]["alpha_mean"] = (
                float(np.mean(results[name]["alphas"])) if results[name]["alphas"] else 0.0
            )
        if name == "Selective2Hop_RankAwareRouter":
            payload["stats"][name]["route_choose_orig"] = (
                float(np.mean(results[name]["route_choose_orig"])) if results[name]["route_choose_orig"] else 0.0
            )
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def compute_entropy(top_scores):
    probs = torch.softmax(top_scores * 20.0, dim=0)
    denom = np.log(max(2, len(top_scores)))
    return float((-(probs * torch.log(probs + 1e-8)).sum() / denom).item())


def rankaware_utility(score_vec, true_idx):
    true_score = float(score_vec[true_idx].item())
    competitor = score_vec.clone()
    competitor[true_idx] = -1e9
    best_other = float(torch.max(competitor).item())
    return true_score - best_other


def key_candidate_set(top_indices, orig_score, ours_score, true_indices):
    top_indices = [int(x) for x in top_indices]
    orig_rank = [int(x) for x in torch.argsort(orig_score[top_indices], descending=True).detach().cpu().numpy()]
    ours_rank = [int(x) for x in torch.argsort(ours_score[top_indices], descending=True).detach().cpu().numpy()]
    key = set()
    for idx in true_indices:
        if int(idx) in top_indices:
            key.add(int(idx))
    for rank_idx in orig_rank[:KEY_TOP]:
        key.add(top_indices[rank_idx])
    for rank_idx in ours_rank[:KEY_TOP]:
        key.add(top_indices[rank_idx])
    return sorted(key)


def build_rankaware_label(ci, true_idx, ours_score, orig_score):
    ours_hybrid = ours_score.clone()
    orig_hybrid = ours_score.clone()
    orig_hybrid[ci] = orig_score[ci]
    util_ours = rankaware_utility(ours_hybrid, true_idx)
    util_orig = rankaware_utility(orig_hybrid, true_idx)
    if util_orig - util_ours > RANK_MARGIN:
        return 1, util_orig, util_ours
    if util_ours - util_orig > RANK_MARGIN:
        return 0, util_orig, util_ours
    return None, util_orig, util_ours


def train_model(x_train, y_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    clf = LogisticRegression(random_state=42, max_iter=200, class_weight="balanced", solver="lbfgs")
    clf.fit(x_train_scaled, y_train)
    return scaler, clf


def predict_proba(bundle, x):
    scaler, clf = bundle
    return clf.predict_proba(scaler.transform(x))[:, 1]


def select_threshold(y_true, probs):
    best_thr = 0.5
    best_score = -1.0
    for thr in THRESHOLD_GRID:
        pred = (probs > thr).astype(np.int32)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def train_rankaware_router(features, labels, sample_ids):
    n = len(features)
    sample_unique = np.unique(sample_ids)
    rng = np.random.default_rng(42)
    rng.shuffle(sample_unique)
    sample_folds = np.array_split(sample_unique, N_FOLDS)
    probs = np.zeros(n, dtype=np.float32)
    pred = np.zeros(n, dtype=np.int32)
    thresholds = []
    coef_list = []
    for fold_idx, val_samples in enumerate(sample_folds, start=1):
        train_samples = np.concatenate([sample_folds[j] for j in range(N_FOLDS) if j != fold_idx - 1])
        train_mask = np.isin(sample_ids, train_samples)
        val_mask = np.isin(sample_ids, val_samples)
        x_train = features[train_mask]
        y_train = labels[train_mask]
        x_val = features[val_mask]
        if len(np.unique(y_train)) < 2:
            probs[val_mask] = 0.0
            thresholds.append(0.5)
            print(f"[RankAware] Fold {fold_idx}: single-class training split, default to Ours")
            continue
        bundle = train_model(x_train, y_train)
        p_train = predict_proba(bundle, x_train)
        thr, bal_acc = select_threshold(y_train, p_train)
        p_val = predict_proba(bundle, x_val)
        probs[val_mask] = p_val.astype(np.float32)
        pred[val_mask] = (p_val > thr).astype(np.int32)
        thresholds.append(thr)
        coef_list.append(np.abs(bundle[1].coef_[0]))
        print(
            f"[RankAware] Fold {fold_idx}: train={len(y_train)}, val={int(np.sum(val_mask))}, "
            f"thr={thr:.2f}, train_bal_acc={bal_acc:.4f}"
        )
    diagnostics = {
        "candidate_disagreement_accuracy": float(np.mean(pred == labels)) if len(labels) else 0.0,
        "prob_low_ratio": float(np.mean(probs < 0.3)) if len(labels) else 0.0,
        "prob_mid_ratio": float(np.mean((probs >= 0.45) & (probs <= 0.55))) if len(labels) else 0.0,
        "prob_high_ratio": float(np.mean(probs > 0.7)) if len(labels) else 0.0,
        "feature_importance_mean": np.mean(np.stack(coef_list, axis=0), axis=0).tolist() if coef_list else [],
        "threshold_mean": float(np.mean(thresholds)) if thresholds else 0.5,
    }
    return probs, diagnostics


@torch.no_grad()
def collect_samples(
    base,
    clap_model,
    text_embeds,
    dataset,
    label_classes,
    kg_classes,
    class_labels_set,
    hop1_relations,
    hop2_relations,
    get_tails,
    prompt_map,
):
    samples = []
    candidate_rows = []
    results = init_results()
    skipped = 0
    sample_index = 0
    mixed_preference = 0
    disagreement_candidate_count = 0
    true_gap_improved_count = 0
    trainable_sample_count = 0

    for sample in tqdm(base.iter_samples(dataset), total=dataset["total"], desc="RankAwareCollect"):
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]
        if not os.path.exists(audio_path) or not true_indices:
            skipped += 1
            continue
        try:
            t_audio_start = time.time()
            audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = F.normalize(base.to_tensor(audio_emb_raw).to(base.DEVICE).float(), dim=-1)
            audio_ms = (time.time() - t_audio_start) * 1000.0
        except Exception:
            skipped += 1
            continue

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[:base.TOP_K]
        alpha_dynamic = base.static_alpha_from_maxsim(torch.max(cos_sim_orig).item())

        t = time.time()
        baseline_score, baseline_prompts = base.run_baseline(cos_sim_orig)
        baseline_ms = audio_ms + (time.time() - t) * 1000.0
        results["Baseline"]["ranks"].append(base.rank_of_true(baseline_score, true_indices))
        results["Baseline"]["times"].append(baseline_ms)
        results["Baseline"]["prompts"].append(baseline_prompts)

        t = time.time()
        iknow_score, iknow_prompts = base.run_iknow(
            clap_model,
            audio_embed,
            cos_sim_orig,
            top_indices,
            label_classes,
            kg_classes,
            class_labels_set,
            hop1_relations,
            get_tails,
        )
        iknow_ms = audio_ms + (time.time() - t) * 1000.0
        results["iKnow"]["ranks"].append(base.rank_of_true(iknow_score, true_indices))
        results["iKnow"]["times"].append(iknow_ms)
        results["iKnow"]["prompts"].append(iknow_prompts)

        t = time.time()
        full_score, full_prompts = base.run_full2hop(
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
        full_ms = audio_ms + (time.time() - t) * 1000.0
        results["Full2Hop"]["ranks"].append(base.rank_of_true(full_score, true_indices))
        results["Full2Hop"]["times"].append(full_ms)
        results["Full2Hop"]["prompts"].append(full_prompts)

        t = time.time()
        orig_score, orig_prompts, orig_extra = run_originalagg(
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
        )
        orig_ms = audio_ms + (time.time() - t) * 1000.0
        results["Selective2Hop_OriginalAgg"]["ranks"].append(base.rank_of_true(orig_score, true_indices))
        results["Selective2Hop_OriginalAgg"]["times"].append(orig_ms)
        results["Selective2Hop_OriginalAgg"]["prompts"].append(orig_prompts)
        results["Selective2Hop_OriginalAgg"]["hop2_activation_sample"].append(bool(orig_extra["hop2_activated"]))
        results["Selective2Hop_OriginalAgg"]["hop2_activation_candidate"].append(float(orig_extra["candidate_level_activation_rate"]))
        results["Selective2Hop_OriginalAgg"]["alphas"].append(alpha_dynamic)

        t = time.time()
        ours_score, ours_prompts, ours_extra = base.run_selective2hop(
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
        ours_ms = audio_ms + (time.time() - t) * 1000.0
        results["Selective2Hop_Ours"]["ranks"].append(base.rank_of_true(ours_score, true_indices))
        results["Selective2Hop_Ours"]["times"].append(ours_ms)
        results["Selective2Hop_Ours"]["prompts"].append(ours_prompts)
        results["Selective2Hop_Ours"]["hop2_activation_sample"].append(bool(ours_extra["hop2_activated"]))
        results["Selective2Hop_Ours"]["hop2_activation_candidate"].append(float(ours_extra["candidate_level_activation_rate"]))
        results["Selective2Hop_Ours"]["alphas"].append(alpha_dynamic)

        base_top = cos_sim_orig[top_indices]
        base_top_sorted, _ = torch.sort(base_top, descending=True)
        base_top1 = float(base_top_sorted[0].item())
        base_top2 = float(base_top_sorted[1].item()) if len(base_top_sorted) > 1 else base_top1
        entropy = compute_entropy(base_top)
        prompt_count = max(orig_prompts, ours_prompts)
        hop2_rate = max(
            float(orig_extra["candidate_level_activation_rate"]),
            float(ours_extra["candidate_level_activation_rate"]),
        )

        true_idx = int(true_indices[0])
        key_candidates = key_candidate_set(top_indices, orig_score, ours_score, true_indices)
        local_labels = set()
        sample_has_trainable = False
        sample_improved = False

        for ci in key_candidates:
            base_i = float(cos_sim_orig[ci].item())
            orig_i = float(orig_score[ci].item())
            ours_i = float(ours_score[ci].item())
            label, util_orig, util_ours = build_rankaware_label(ci, true_idx, ours_score, orig_score)
            if label is None:
                continue
            sample_has_trainable = True
            disagreement_candidate_count += 1
            local_labels.add(int(label))
            if label == 1:
                sample_improved = True
            candidate_rows.append(
                {
                    "sample_id": sample_index,
                    "candidate_id": ci,
                    "feature": np.array(
                        [
                            base_top1,
                            base_top1 - base_top2,
                            entropy,
                            hop2_rate,
                            np.log1p(prompt_count),
                            base_i,
                            orig_i - base_i,
                            ours_i - base_i,
                        ],
                        dtype=np.float32,
                    ),
                    "label": int(label),
                    "utility_orig": float(util_orig),
                    "utility_ours": float(util_ours),
                }
            )

        if len(local_labels) > 1:
            mixed_preference += 1
        if sample_has_trainable:
            trainable_sample_count += 1
        if sample_improved:
            true_gap_improved_count += 1

        samples.append(
            {
                "true_indices": true_indices,
                "orig_score": orig_score.detach().cpu(),
                "ours_score": ours_score.detach().cpu(),
                "router_time_ms": float(orig_ms + ours_ms),
                "router_prompts": float(prompt_count),
                "hop2_sample": bool(orig_extra["hop2_activated"] or ours_extra["hop2_activated"]),
                "hop2_candidate": float(hop2_rate),
                "alpha": float(alpha_dynamic),
                "top_indices": [int(x) for x in top_indices],
                "key_candidates": key_candidates,
            }
        )
        sample_index += 1

    diagnostics = {
        "mixed_preference_ratio": float(mixed_preference / max(1, len(samples))),
        "candidate_disagreement_count": int(disagreement_candidate_count),
        "trainable_candidate_ratio": float(len(candidate_rows) / max(1, len(samples) * max(1, KEY_TOP))),
        "trainable_sample_ratio": float(trainable_sample_count / max(1, len(samples))),
        "true_gap_improved_ratio": float(true_gap_improved_count / max(1, len(samples))),
        "key_top": KEY_TOP,
    }
    return samples, candidate_rows, results, skipped, diagnostics


def apply_rankaware_router(base, samples, candidate_rows, probs, diagnostics, results):
    feature_map = {(row["sample_id"], row["candidate_id"]): float(prob) for row, prob in zip(candidate_rows, probs)}
    threshold = float(diagnostics["threshold_mean"])
    corrected_ratio = 0

    for sample_id, sample in enumerate(samples):
        final_score = sample["ours_score"].clone()
        choose_orig_count = 0
        for ci in sample["key_candidates"]:
            key = (sample_id, ci)
            if key not in feature_map:
                continue
            p = feature_map[key]
            choose_orig = p > threshold
            if choose_orig:
                final_score[ci] = sample["orig_score"][ci]
                choose_orig_count += 1
            else:
                final_score[ci] = sample["ours_score"][ci]
        orig_rank = base.rank_of_true(sample["orig_score"], sample["true_indices"])
        ours_rank = base.rank_of_true(sample["ours_score"], sample["true_indices"])
        final_rank = base.rank_of_true(final_score, sample["true_indices"])
        if final_rank < min(orig_rank, ours_rank):
            corrected_ratio += 1
        results["Selective2Hop_RankAwareRouter"]["ranks"].append(final_rank)
        results["Selective2Hop_RankAwareRouter"]["times"].append(sample["router_time_ms"])
        results["Selective2Hop_RankAwareRouter"]["prompts"].append(sample["router_prompts"])
        results["Selective2Hop_RankAwareRouter"]["hop2_activation_sample"].append(sample["hop2_sample"])
        results["Selective2Hop_RankAwareRouter"]["hop2_activation_candidate"].append(sample["hop2_candidate"])
        results["Selective2Hop_RankAwareRouter"]["alphas"].append(sample["alpha"])
        results["Selective2Hop_RankAwareRouter"]["route_choose_orig"].append(
            float(choose_orig_count / max(1, len(sample["key_candidates"])))
        )
    diagnostics["top1_corrected_ratio"] = float(corrected_ratio / max(1, len(samples)))


def run_rankaware_router_dataset(module_path, output_json, launch_title):
    base = load_base(module_path)
    print(launch_title)
    print(f"Device: {base.DEVICE}")
    print("RankAwareRouter model=lr")
    print("RankAwareRouter features: global(base_top1, margin, entropy, hop2_rate, prompt_count) + local(base_i, orig-base_i, ours-base_i)")
    print("RankAwareRouter labels: choose head that enlarges true-vs-best-competitor utility gap; only key candidates are trainable")

    prompt_map = base.load_llm_prompts(base.LLM_PROMPTS_PATH)
    clap_model = base.CLAP(version="2023", use_cuda=torch.cuda.is_available())
    with base.warnings.catch_warnings():
        base.warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(base.KGE_MODEL_DIR, "trained_model.pkl"), map_location=base.DEVICE)
    kge_model.eval()
    training_factory = base.TriplesFactory.from_path(base.TRAIN_TRIPLES_PATH)
    hop1_relations = [rel for rel in base.HOP1_RELATIONS if rel in training_factory.relation_to_id]
    hop2_relations = [rel for rel in base.HOP2_RELATIONS if rel in training_factory.relation_to_id]
    print(f"Valid hop1 relations: {hop1_relations}")
    print(f"Valid hop2 relations: {hop2_relations}")

    get_tails = base.build_tail_predictor(kge_model, training_factory)
    dataset = base.load_dataset()
    label_classes = dataset["label_classes"]
    kg_classes = dataset["kg_classes"]
    class_labels_set = dataset["class_labels_set"]
    text_embeds = F.normalize(base.get_safe_text_embeddings(clap_model, label_classes, base.DEVICE), dim=-1)

    samples, candidate_rows, results, skipped, base_diag = collect_samples(
        base,
        clap_model,
        text_embeds,
        dataset,
        label_classes,
        kg_classes,
        class_labels_set,
        hop1_relations,
        hop2_relations,
        get_tails,
        prompt_map,
    )
    print(
        f"[RankAware] collected_samples={len(samples)}, trainable_candidates={len(candidate_rows)}, skipped={skipped}, "
        f"mixed_preference_ratio={base_diag['mixed_preference_ratio']:.4f}, trainable_sample_ratio={base_diag['trainable_sample_ratio']:.4f}"
    )

    features = np.stack([row["feature"] for row in candidate_rows], axis=0)
    labels = np.array([row["label"] for row in candidate_rows], dtype=np.int32)
    sample_ids = np.array([row["sample_id"] for row in candidate_rows], dtype=np.int32)

    probs, diagnostics = train_rankaware_router(features, labels, sample_ids)
    apply_rankaware_router(base, samples, candidate_rows, probs, diagnostics, results)

    feature_names = [
        "baseline_top1",
        "baseline_margin",
        "entropy",
        "hop2_activation_rate",
        "prompt_count_log1p",
        "base_i",
        "orig_minus_base_i",
        "ours_minus_base_i",
    ]
    print(
        f"[RankAware] disagreement_accuracy={diagnostics['candidate_disagreement_accuracy']:.4f}, "
        f"prob_low={diagnostics['prob_low_ratio']:.4f}, "
        f"prob_mid={diagnostics['prob_mid_ratio']:.4f}, "
        f"prob_high={diagnostics['prob_high_ratio']:.4f}, "
        f"top1_corrected_ratio={diagnostics.get('top1_corrected_ratio', 0.0):.4f}"
    )
    if diagnostics["feature_importance_mean"]:
        print("[RankAware] feature_importance:")
        for name, val in zip(feature_names, diagnostics["feature_importance_mean"]):
            print(f"  - {name}: {val:.6f}")

    print_tables(base, results, "RankAware Candidate Router Ablation")
    save_results(
        base,
        results,
        output_json,
        meta={
            "router_type": "candidate_rankaware_lr",
            "head_a": "Selective2Hop_OriginalAgg",
            "head_b": "Selective2Hop_Ours",
            "features": feature_names,
            "n_folds": N_FOLDS,
            "rank_margin": RANK_MARGIN,
            "default_close_margin": DEFAULT_CLOSE_MARGIN,
            "diagnostics": diagnostics,
            **base_diag,
            "skipped_count": skipped,
        },
    )
