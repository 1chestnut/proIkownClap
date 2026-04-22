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


REPULSIVE_BETA = 0.35
REPULSIVE_MARGIN = 0.01
N_FOLDS = 5
TRUE_SCORE_MARGIN = 0.01
DEFAULT_SWITCH_MARGIN = 0.01
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

    h2_prompts = base.build_h2_prompts(
        h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map
    )
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
    candidate_scores = {}
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
        candidate_scores[int(ci)] = float(s_i.item())
        score[ci] = s_i
        prompt_count += used_prompts
        hop2_flags.append(hop2_flag)
    extras = {
        "hop2_activated": any(hop2_flags),
        "candidate_level_activation_rate": float(np.mean(hop2_flags)) if hop2_flags else 0.0,
        "candidate_scores": candidate_scores,
    }
    return score, prompt_count, extras


def run_repulsive(cos_sim_orig, top_indices, candidate_scores):
    score = cos_sim_orig.clone()
    for ci in top_indices:
        own = candidate_scores[int(ci)]
        competitors = [candidate_scores[int(cj)] for cj in top_indices if int(cj) != int(ci)]
        comp = max(competitors) if competitors else own
        penalty = max(0.0, comp - own + REPULSIVE_MARGIN)
        score[ci] = own - (REPULSIVE_BETA * penalty)
    return score


def init_results():
    return {
        "Baseline": {"ranks": [], "times": [], "prompts": []},
        "iKnow": {"ranks": [], "times": [], "prompts": []},
        "Full2Hop": {"ranks": [], "times": [], "prompts": []},
        "Selective2Hop_OriginalAgg": {
            "ranks": [], "times": [], "prompts": [],
            "hop2_activation_sample": [], "hop2_activation_candidate": [], "alphas": [],
        },
        "Selective2Hop_Ours": {
            "ranks": [], "times": [], "prompts": [],
            "hop2_activation_sample": [], "hop2_activation_candidate": [], "alphas": [],
        },
        "Selective2Hop_RouterV2": {
            "ranks": [], "times": [], "prompts": [],
            "hop2_activation_sample": [], "hop2_activation_candidate": [], "alphas": [],
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
        "Selective2Hop_RouterV2",
    ]
    print("\n" + "=" * 180)
    print(title)
    print("-" * 180)
    print(
        f"{'Metric':<8} | {'Baseline':<10} | {'iKnow':<10} | {'Full2Hop':<12} | "
        f"{'Sel2 OriginalAgg':<18} | {'Sel2 Ours':<18} | {'Sel2 RouterV2':<18}"
    )
    print("-" * 180)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(
            f"{metric_name:<8} | {metrics['Baseline'][idx]:<10.2f} | {metrics['iKnow'][idx]:<10.2f} | "
            f"{metrics['Full2Hop'][idx]:<12.2f} | {metrics['Selective2Hop_OriginalAgg'][idx]:<18.2f} | "
            f"{metrics['Selective2Hop_Ours'][idx]:<18.2f} | {metrics['Selective2Hop_RouterV2'][idx]:<18.2f}"
        )

    print("\n" + "=" * 190)
    print(
        f"{'Method':<24} | {'Hop2 activation(sample)':<24} | {'Hop2 activation(candidate)':<27} | "
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
        if name == "Selective2Hop_RouterV2":
            route_rate = f"{np.mean(results[name]['route_choose_orig']) * 100:.1f}%"
        print(
            f"{name:<24} | {sample_rate:<24} | {candidate_rate:<27} | "
            f"{np.mean(results[name]['prompts']):<12.1f} | {np.mean(results[name]['times']):<14.1f} | "
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
        }
        if name.startswith("Selective2Hop"):
            payload["stats"][name]["hop2_activation_sample"] = float(
                np.mean(results[name]["hop2_activation_sample"])
            ) if results[name]["hop2_activation_sample"] else 0.0
            payload["stats"][name]["hop2_activation_candidate"] = float(
                np.mean(results[name]["hop2_activation_candidate"])
            ) if results[name]["hop2_activation_candidate"] else 0.0
            payload["stats"][name]["alpha_mean"] = float(
                np.mean(results[name]["alphas"])
            ) if results[name]["alphas"] else 0.0
        if name == "Selective2Hop_RouterV2":
            payload["stats"][name]["route_choose_orig"] = float(
                np.mean(results[name]["route_choose_orig"])
            ) if results[name]["route_choose_orig"] else 0.0
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def compute_entropy(top_scores):
    probs = torch.softmax(top_scores * 20.0, dim=0)
    denom = np.log(max(2, len(top_scores)))
    return float((-(probs * torch.log(probs + 1e-8)).sum() / denom).item())


def assign_train_label(orig_rank, ours_rank, orig_true_score, ours_true_score):
    if orig_rank < ours_rank:
        return 1
    if ours_rank < orig_rank:
        return 0
    if orig_true_score - ours_true_score > TRUE_SCORE_MARGIN:
        return 1
    if ours_true_score - orig_true_score > TRUE_SCORE_MARGIN:
        return 0
    return None


def should_default_to_ours(orig_pred_idx, ours_pred_idx, orig_top1, ours_top1):
    return (orig_pred_idx == ours_pred_idx) and (abs(orig_top1 - ours_top1) < DEFAULT_SWITCH_MARGIN)


def build_feature_vector(base_top1, margin, entropy, orig_gain, ours_gain, hop2_rate, prompt_count):
    return np.array(
        [
            float(base_top1),
            float(margin),
            float(entropy),
            float(orig_gain),
            float(ours_gain),
            float(hop2_rate),
            float(np.log1p(prompt_count)),
        ],
        dtype=np.float32,
    )


def select_threshold(y_true, p_train):
    best_thr = 0.5
    best_score = -1.0
    for thr in THRESHOLD_GRID:
        pred = (p_train > thr).astype(np.int32)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def train_and_predict(features, labels, trainable_mask, default_mask):
    n = len(features)
    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, N_FOLDS)

    probs = np.zeros(n, dtype=np.float32)
    thresholds = []
    usable_counts = []
    coef_list = []
    pred_by_fold_threshold = np.zeros(n, dtype=np.int32)

    for fold_idx, val_idx in enumerate(folds, start=1):
        train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fold_idx - 1])
        fit_idx = train_idx[trainable_mask[train_idx]]

        if len(fit_idx) == 0 or len(np.unique(labels[fit_idx])) < 2:
            probs[val_idx] = 0.0
            thresholds.append(0.5)
            usable_counts.append(int(len(fit_idx)))
            print(f"[RouterV2] Fold {fold_idx}: no usable disagreement samples, default to Ours")
            continue

        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[fit_idx])
        y_train = labels[fit_idx]
        clf = LogisticRegression(
            random_state=42,
            max_iter=200,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(x_train, y_train)
        coef_list.append(clf.coef_[0].copy())

        p_train = clf.predict_proba(x_train)[:, 1]
        thr, bal_acc = select_threshold(y_train, p_train)
        thresholds.append(thr)
        usable_counts.append(int(len(fit_idx)))

        x_val = scaler.transform(features[val_idx])
        p_val = clf.predict_proba(x_val)[:, 1]
        probs[val_idx] = p_val.astype(np.float32)
        pred_by_fold_threshold[val_idx] = (p_val > thr).astype(np.int32)
        print(
            f"[RouterV2] Fold {fold_idx}: train_all={len(train_idx)}, train_use={len(fit_idx)}, "
            f"val={len(val_idx)}, thr={thr:.2f}, train_bal_acc={bal_acc:.4f}, default_mask_val={float(np.mean(default_mask[val_idx])):.4f}"
        )

    diagnostics = {
        "coef_mean": np.mean(np.stack(coef_list, axis=0), axis=0).tolist() if coef_list else [],
        "coef_abs_mean": np.mean(np.abs(np.stack(coef_list, axis=0)), axis=0).tolist() if coef_list else [],
        "disagreement_accuracy": float(
            np.mean(pred_by_fold_threshold[trainable_mask] == labels[trainable_mask])
        ) if np.any(trainable_mask) else 0.0,
        "prob_low_ratio": float(np.mean(probs[trainable_mask] < 0.3)) if np.any(trainable_mask) else 0.0,
        "prob_mid_ratio": float(np.mean((probs[trainable_mask] >= 0.45) & (probs[trainable_mask] <= 0.55))) if np.any(trainable_mask) else 0.0,
        "prob_high_ratio": float(np.mean(probs[trainable_mask] > 0.7)) if np.any(trainable_mask) else 0.0,
    }
    return probs, thresholds, usable_counts, diagnostics


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
    results = init_results()
    skipped = 0

    for sample in tqdm(base.iter_samples(dataset), total=dataset["total"], desc="RouterV2Collect"):
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
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, get_tails,
        )
        iknow_ms = audio_ms + (time.time() - t) * 1000.0
        results["iKnow"]["ranks"].append(base.rank_of_true(iknow_score, true_indices))
        results["iKnow"]["times"].append(iknow_ms)
        results["iKnow"]["prompts"].append(iknow_prompts)

        t = time.time()
        full_score, full_prompts = base.run_full2hop(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic,
        )
        full_ms = audio_ms + (time.time() - t) * 1000.0
        results["Full2Hop"]["ranks"].append(base.rank_of_true(full_score, true_indices))
        results["Full2Hop"]["times"].append(full_ms)
        results["Full2Hop"]["prompts"].append(full_prompts)

        t = time.time()
        orig_score, orig_prompts, orig_extra = run_originalagg(
            base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map,
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
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic,
        )
        ours_ms = audio_ms + (time.time() - t) * 1000.0
        results["Selective2Hop_Ours"]["ranks"].append(base.rank_of_true(ours_score, true_indices))
        results["Selective2Hop_Ours"]["times"].append(ours_ms)
        results["Selective2Hop_Ours"]["prompts"].append(ours_prompts)
        results["Selective2Hop_Ours"]["hop2_activation_sample"].append(bool(ours_extra["hop2_activated"]))
        results["Selective2Hop_Ours"]["hop2_activation_candidate"].append(float(ours_extra["candidate_level_activation_rate"]))
        results["Selective2Hop_Ours"]["alphas"].append(alpha_dynamic)

        rep_score = run_repulsive(cos_sim_orig, top_indices, orig_extra["candidate_scores"])

        base_top = cos_sim_orig[top_indices]
        base_top_sorted, _ = torch.sort(base_top, descending=True)
        base_top1 = float(base_top_sorted[0].item())
        base_top2 = float(base_top_sorted[1].item()) if len(base_top_sorted) > 1 else base_top1
        base_margin = base_top1 - base_top2
        entropy = compute_entropy(base_top)

        orig_top1 = float(torch.max(orig_score[top_indices]).item())
        ours_top1 = float(torch.max(ours_score[top_indices]).item())
        rep_top1 = float(torch.max(rep_score[top_indices]).item())
        orig_gain = orig_top1 - base_top1
        ours_gain = ours_top1 - base_top1
        hop2_rate = max(float(orig_extra["candidate_level_activation_rate"]), float(ours_extra["candidate_level_activation_rate"]))
        prompt_count = max(orig_prompts, ours_prompts)

        orig_pred_idx = int(torch.argmax(orig_score).item())
        ours_pred_idx = int(torch.argmax(ours_score).item())
        true_idx = int(true_indices[0])
        orig_rank = base.rank_of_true(orig_score, true_indices)
        ours_rank = base.rank_of_true(ours_score, true_indices)
        train_label = assign_train_label(
            orig_rank,
            ours_rank,
            float(orig_score[true_idx].item()),
            float(ours_score[true_idx].item()),
        )
        default_to_ours = should_default_to_ours(orig_pred_idx, ours_pred_idx, orig_top1, ours_top1)

        samples.append(
            {
                "feature": build_feature_vector(
                    base_top1, base_margin, entropy, orig_gain, ours_gain, hop2_rate, prompt_count
                ),
                "trainable": train_label is not None,
                "label": -1 if train_label is None else int(train_label),
                "default_to_ours": bool(default_to_ours),
                "orig_score": orig_score.detach().cpu(),
                "ours_score": ours_score.detach().cpu(),
                "true_indices": true_indices,
                "router_time_ms": float(orig_ms + ours_ms),
                "router_prompts": float(prompt_count),
                "hop2_sample": bool(orig_extra["hop2_activated"] or ours_extra["hop2_activated"]),
                "hop2_candidate": float(hop2_rate),
                "alpha": float(alpha_dynamic),
            }
        )
    return samples, results, skipped


def apply_router_results(base, samples, probs, thresholds, results):
    fold_thresholds = np.array(thresholds, dtype=np.float32)
    global_threshold = float(np.mean(fold_thresholds)) if len(fold_thresholds) else 0.5
    for sample, prob in zip(samples, probs):
        if sample["default_to_ours"]:
            choose_orig = False
        else:
            choose_orig = float(prob) > global_threshold
        score = sample["orig_score"].clone() if choose_orig else sample["ours_score"].clone()
        results["Selective2Hop_RouterV2"]["ranks"].append(base.rank_of_true(score, sample["true_indices"]))
        results["Selective2Hop_RouterV2"]["times"].append(float(sample["router_time_ms"]))
        results["Selective2Hop_RouterV2"]["prompts"].append(float(sample["router_prompts"]))
        results["Selective2Hop_RouterV2"]["hop2_activation_sample"].append(bool(sample["hop2_sample"]))
        results["Selective2Hop_RouterV2"]["hop2_activation_candidate"].append(float(sample["hop2_candidate"]))
        results["Selective2Hop_RouterV2"]["alphas"].append(float(sample["alpha"]))
        results["Selective2Hop_RouterV2"]["route_choose_orig"].append(1.0 if choose_orig else 0.0)
    return global_threshold


def run_router_dataset(module_path, output_json, launch_title):
    base = load_base(module_path)
    print(launch_title)
    print(f"Device: {base.DEVICE}")
    print("RouterV2 heads: OriginalAgg vs Ours")
    print("RouterV2 features: baseline top1, margin, entropy, orig-base, ours-base, hop2 activation, prompt count(log1p)")
    print("Training policy: disagreement-aware only, tie/near-tie samples excluded from training, default-to-Ours on near-equal heads")

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

    samples, results, skipped = collect_samples(
        base, clap_model, text_embeds, dataset, label_classes, kg_classes,
        class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map,
    )
    features = np.stack([sample["feature"] for sample in samples], axis=0)
    labels = np.array([sample["label"] for sample in samples], dtype=np.int32)
    trainable_mask = np.array([sample["trainable"] for sample in samples], dtype=bool)
    default_mask = np.array([sample["default_to_ours"] for sample in samples], dtype=bool)

    print(
        f"[RouterV2] collected={len(samples)}, skipped={skipped}, "
        f"trainable_ratio={float(np.mean(trainable_mask)):.4f}, default_to_ours_ratio={float(np.mean(default_mask)):.4f}"
    )

    probs, thresholds, usable_counts, diagnostics = train_and_predict(features, labels, trainable_mask, default_mask)
    global_threshold = apply_router_results(base, samples, probs, thresholds, results)

    feature_names = [
        "baseline_top1",
        "baseline_margin",
        "entropy",
        "originalagg_gain",
        "ours_gain",
        "hop2_activation_rate",
        "prompt_count_log1p",
    ]
    print(
        "[RouterV2] disagreement_accuracy="
        f"{diagnostics['disagreement_accuracy']:.4f}, prob_low={diagnostics['prob_low_ratio']:.4f}, "
        f"prob_mid={diagnostics['prob_mid_ratio']:.4f}, prob_high={diagnostics['prob_high_ratio']:.4f}"
    )
    if diagnostics["coef_abs_mean"]:
        print("[RouterV2] feature_importance(abs coef mean):")
        for name, coef in zip(feature_names, diagnostics["coef_abs_mean"]):
            print(f"  - {name}: {coef:.6f}")

    print_tables(base, results, "RouterV2 Ablation")
    save_results(
        base,
        results,
        output_json,
        meta={
            "router_type": "disagreement_aware_logreg",
            "head_a": "Selective2Hop_OriginalAgg",
            "head_b": "Selective2Hop_Ours",
            "features": [
                *feature_names
            ],
            "n_folds": N_FOLDS,
            "threshold_grid": [float(x) for x in THRESHOLD_GRID],
            "global_threshold": global_threshold,
            "usable_train_counts": usable_counts,
            "trainable_ratio": float(np.mean(trainable_mask)),
            "default_to_ours_ratio": float(np.mean(default_mask)),
            "true_score_margin": TRUE_SCORE_MARGIN,
            "default_switch_margin": DEFAULT_SWITCH_MARGIN,
            "skipped_count": skipped,
            "diagnostics": diagnostics,
        },
    )
