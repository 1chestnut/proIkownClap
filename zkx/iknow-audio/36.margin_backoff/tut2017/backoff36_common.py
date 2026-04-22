# -*- coding: utf-8 -*-
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


N_FOLDS = 5
THRESHOLD_GRID = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20], dtype=np.float32)


def load_base(module_path):
    spec = importlib.util.spec_from_file_location("backoff36_base_module", module_path)
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
            "ranks": [], "times": [], "prompts": [], "hop2_activation_sample": [], "hop2_activation_candidate": [], "alphas": []
        },
        "Selective2Hop_Ours": {
            "ranks": [], "times": [], "prompts": [], "hop2_activation_sample": [], "hop2_activation_candidate": [], "alphas": []
        },
        "Selective2Hop_MarginBackoff": {
            "ranks": [], "times": [], "prompts": [], "hop2_activation_sample": [], "hop2_activation_candidate": [], "alphas": [], "route_choose_orig": []
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
        "Selective2Hop_MarginBackoff",
    ]
    print("\n" + "=" * 210)
    print(title)
    print("-" * 210)
    print(
        f"{'Metric':<10} | {'Baseline':<10} | {'iKnow':<10} | {'Full2Hop':<12} | "
        f"{'Sel2 OriginalAgg':<18} | {'Sel2 Ours':<18} | {'MarginBackoff':<18}"
    )
    print("-" * 210)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(
            f"{metric_name:<10} | "
            f"{metrics['Baseline'][idx]:<10.2f} | {metrics['iKnow'][idx]:<10.2f} | {metrics['Full2Hop'][idx]:<12.2f} | "
            f"{metrics['Selective2Hop_OriginalAgg'][idx]:<18.2f} | {metrics['Selective2Hop_Ours'][idx]:<18.2f} | "
            f"{metrics['Selective2Hop_MarginBackoff'][idx]:<18.2f}"
        )
    print("\n" + "=" * 220)
    print(
        f"{'Method':<28} | {'Hop2 activation(sample)':<24} | {'Hop2 activation(candidate)':<27} | "
        f"{'Avg prompts':<12} | {'Avg time (ms)':<14} | {'Hit@1':<8} | {'MRR':<8} | {'Route->Orig':<12}"
    )
    print("-" * 220)
    for name in order:
        sample_rate = "N/A"
        candidate_rate = "N/A"
        route_rate = "N/A"
        if name.startswith("Selective2Hop"):
            if "hop2_activation_sample" in results[name]:
                sample_rate = f"{np.mean(results[name]['hop2_activation_sample']) * 100:.1f}%"
                candidate_rate = f"{np.mean(results[name]['hop2_activation_candidate']) * 100:.1f}%"
        if name == "Selective2Hop_MarginBackoff":
            route_rate = f"{np.mean(results[name]['route_choose_orig']) * 100:.1f}%"
        print(
            f"{name:<28} | {sample_rate:<24} | {candidate_rate:<27} | "
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
        if "hop2_activation_sample" in results[name]:
            payload["stats"][name]["hop2_activation_sample"] = float(np.mean(results[name]["hop2_activation_sample"])) if results[name]["hop2_activation_sample"] else 0.0
            payload["stats"][name]["hop2_activation_candidate"] = float(np.mean(results[name]["hop2_activation_candidate"])) if results[name]["hop2_activation_candidate"] else 0.0
        if "route_choose_orig" in results[name]:
            payload["stats"][name]["route_choose_orig"] = float(np.mean(results[name]["route_choose_orig"])) if results[name]["route_choose_orig"] else 0.0
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def choose_threshold(base, samples):
    indices = np.arange(len(samples))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, N_FOLDS)
    chosen = []
    for fold_idx in range(N_FOLDS):
        train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fold_idx]) if N_FOLDS > 1 else folds[fold_idx]
        best_thr = float(THRESHOLD_GRID[0])
        best_key = None
        for thr in THRESHOLD_GRID:
            ranks = []
            for idx in train_idx:
                s = samples[int(idx)]
                final_score = s["orig_score"] if s["margin"] <= thr else s["ours_score"]
                ranks.append(base.rank_of_true(final_score, s["true_indices"]))
            hit1 = float(np.mean(np.array(ranks) <= 1))
            mrr = float(np.mean(1.0 / np.array(ranks)))
            key = (hit1, mrr)
            if best_key is None or key > best_key:
                best_key = key
                best_thr = float(thr)
        chosen.append(best_thr)
        print(f"[MarginBackoff] fold={fold_idx + 1} best_threshold={best_thr:.3f} train_hit1={best_key[0]:.4f} train_mrr={best_key[1]:.4f}")
    return float(np.mean(chosen)), [float(x) for x in chosen]


@torch.no_grad()
def run_backoff_dataset(module_path, output_json, launch_title):
    base = load_base(module_path)
    print(launch_title)
    print(f"Device: {base.DEVICE}")
    print("MarginBackoff rule: if baseline margin <= threshold use OriginalAgg else use Ours")
    print(f"Threshold grid: {THRESHOLD_GRID.tolist()}")

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
    results = init_results()
    skipped_count = 0
    samples = []

    for sample in tqdm(base.iter_samples(dataset), total=dataset["total"], desc="BackoffCollect"):
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]
        if not os.path.exists(audio_path) or not true_indices:
            skipped_count += 1
            continue
        try:
            t_audio_start = time.time()
            audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = F.normalize(base.to_tensor(audio_emb_raw).to(base.DEVICE).float(), dim=-1)
            audio_ms = (time.time() - t_audio_start) * 1000.0
        except Exception:
            skipped_count += 1
            continue

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = [int(x) for x in sorted_indices[: base.TOP_K]]
        baseline_score, baseline_prompts = base.run_baseline(cos_sim_orig)
        alpha_dynamic = base.static_alpha_from_maxsim(float(torch.max(cos_sim_orig).item()))

        results["Baseline"]["ranks"].append(base.rank_of_true(baseline_score, true_indices))
        results["Baseline"]["times"].append(audio_ms)
        results["Baseline"]["prompts"].append(baseline_prompts)

        top_scores = cos_sim_orig[top_indices]
        top_sorted, _ = torch.sort(top_scores, descending=True)
        margin = float((top_sorted[0] - top_sorted[1]).item()) if top_sorted.numel() > 1 else 0.0

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

        samples.append(
            {
                "true_indices": true_indices,
                "margin": margin,
                "orig_score": orig_score.detach().cpu(),
                "ours_score": ours_score.detach().cpu(),
                "router_time_ms": max(orig_ms, ours_ms),
                "router_prompts": max(orig_prompts, ours_prompts),
                "hop2_sample": bool(orig_extra["hop2_activated"] or ours_extra["hop2_activated"]),
                "hop2_candidate": max(float(orig_extra["candidate_level_activation_rate"]), float(ours_extra["candidate_level_activation_rate"])),
                "alpha": float(alpha_dynamic),
            }
        )

    threshold_mean, thresholds = choose_threshold(base, samples)
    choose_orig = 0
    for sample in samples:
        use_orig = sample["margin"] <= threshold_mean
        final_score = sample["orig_score"] if use_orig else sample["ours_score"]
        choose_orig += int(use_orig)
        results["Selective2Hop_MarginBackoff"]["ranks"].append(base.rank_of_true(final_score, sample["true_indices"]))
        results["Selective2Hop_MarginBackoff"]["times"].append(sample["router_time_ms"])
        results["Selective2Hop_MarginBackoff"]["prompts"].append(sample["router_prompts"])
        results["Selective2Hop_MarginBackoff"]["hop2_activation_sample"].append(sample["hop2_sample"])
        results["Selective2Hop_MarginBackoff"]["hop2_activation_candidate"].append(sample["hop2_candidate"])
        results["Selective2Hop_MarginBackoff"]["alphas"].append(sample["alpha"])
        results["Selective2Hop_MarginBackoff"]["route_choose_orig"].append(float(use_orig))

    print(f"[MarginBackoff] threshold_mean={threshold_mean:.4f}, choose_orig_ratio={choose_orig / max(1, len(samples)):.4f}")
    print_tables(base, results, "Confidence-Gated Margin Backoff")
    save_results(
        base,
        results,
        output_json,
        meta={
            "router_type": "confidence_gated_backoff",
            "head_a": "Selective2Hop_OriginalAgg",
            "head_b": "Selective2Hop_Ours",
            "threshold_grid": THRESHOLD_GRID.tolist(),
            "threshold_mean": threshold_mean,
            "thresholds": thresholds,
            "skipped_count": skipped_count,
        },
    )
