# -*- coding: utf-8 -*-
import importlib.util
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_base(module_path):
    spec = importlib.util.spec_from_file_location("oracle35_base_module", module_path)
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
        "OracleSample": {
            "ranks": [], "times": [], "prompts": [], "hop2_activation_sample": [], "hop2_activation_candidate": [], "alphas": [], "route_choose_orig": []
        },
        "OracleCandidate": {
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
        "OracleSample",
        "OracleCandidate",
    ]
    print("\n" + "=" * 210)
    print(title)
    print("-" * 210)
    print(
        f"{'Metric':<10} | {'Baseline':<10} | {'iKnow':<10} | {'Full2Hop':<12} | "
        f"{'Sel2 OriginalAgg':<18} | {'Sel2 Ours':<18} | {'OracleSample':<16} | {'OracleCandidate':<18}"
    )
    print("-" * 210)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(
            f"{metric_name:<10} | "
            f"{metrics['Baseline'][idx]:<10.2f} | {metrics['iKnow'][idx]:<10.2f} | {metrics['Full2Hop'][idx]:<12.2f} | "
            f"{metrics['Selective2Hop_OriginalAgg'][idx]:<18.2f} | {metrics['Selective2Hop_Ours'][idx]:<18.2f} | "
            f"{metrics['OracleSample'][idx]:<16.2f} | {metrics['OracleCandidate'][idx]:<18.2f}"
        )
    print("\n" + "=" * 220)
    print(
        f"{'Method':<24} | {'Hop2 activation(sample)':<24} | {'Hop2 activation(candidate)':<27} | "
        f"{'Avg prompts':<12} | {'Avg time (ms)':<14} | {'Hit@1':<8} | {'MRR':<8} | {'Route->Orig':<12}"
    )
    print("-" * 220)
    for name in order:
        sample_rate = "N/A"
        candidate_rate = "N/A"
        route_rate = "N/A"
        if name.startswith("Selective2Hop") or name.startswith("Oracle"):
            if "hop2_activation_sample" in results[name]:
                sample_rate = f"{np.mean(results[name]['hop2_activation_sample']) * 100:.1f}%"
                candidate_rate = f"{np.mean(results[name]['hop2_activation_candidate']) * 100:.1f}%"
        if name.startswith("Oracle"):
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
        if "hop2_activation_sample" in results[name]:
            payload["stats"][name]["hop2_activation_sample"] = float(np.mean(results[name]["hop2_activation_sample"])) if results[name]["hop2_activation_sample"] else 0.0
            payload["stats"][name]["hop2_activation_candidate"] = float(np.mean(results[name]["hop2_activation_candidate"])) if results[name]["hop2_activation_candidate"] else 0.0
        if "route_choose_orig" in results[name]:
            payload["stats"][name]["route_choose_orig"] = float(np.mean(results[name]["route_choose_orig"])) if results[name]["route_choose_orig"] else 0.0
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def score_gap(score_vec, true_idx):
    true_score = float(score_vec[true_idx].item())
    competitor = score_vec.clone()
    competitor[true_idx] = -1e9
    best_other = float(torch.max(competitor).item())
    return true_score - best_other


def choose_whole_head(base, orig_score, ours_score, true_indices):
    orig_rank = base.rank_of_true(orig_score, true_indices)
    ours_rank = base.rank_of_true(ours_score, true_indices)
    true_idx = int(true_indices[0])
    if orig_rank < ours_rank:
        return orig_score.clone(), 1.0
    if ours_rank < orig_rank:
        return ours_score.clone(), 0.0
    if score_gap(orig_score, true_idx) >= score_gap(ours_score, true_idx):
        return orig_score.clone(), 1.0
    return ours_score.clone(), 0.0


def choose_candidate_oracle(base, orig_score, ours_score, top_indices, true_indices):
    true_idx = int(true_indices[0])
    best_score = None
    best_choice = None
    for mask_bits in itertools.product([0, 1], repeat=len(top_indices)):
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
        if best_score is None or key > best_score:
            best_score = key
            best_choice = (final_score, float(choose_orig / max(1, len(top_indices))))
    return best_choice


@torch.no_grad()
def run_oracle_dataset(module_path, output_json, launch_title):
    base = load_base(module_path)
    print(launch_title)
    print(f"Device: {base.DEVICE}")
    print("Oracle diagnostics: sample-level oracle + candidate-level oracle over OriginalAgg and Ours")

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

    for sample in tqdm(base.iter_samples(dataset), total=dataset["total"], desc="OracleCollect"):
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

        oracle_sample_score, sample_choose_orig = choose_whole_head(base, orig_score, ours_score, true_indices)
        results["OracleSample"]["ranks"].append(base.rank_of_true(oracle_sample_score, true_indices))
        results["OracleSample"]["times"].append(max(orig_ms, ours_ms))
        results["OracleSample"]["prompts"].append(max(orig_prompts, ours_prompts))
        results["OracleSample"]["hop2_activation_sample"].append(bool(orig_extra["hop2_activated"] or ours_extra["hop2_activated"]))
        results["OracleSample"]["hop2_activation_candidate"].append(max(float(orig_extra["candidate_level_activation_rate"]), float(ours_extra["candidate_level_activation_rate"])))
        results["OracleSample"]["alphas"].append(alpha_dynamic)
        results["OracleSample"]["route_choose_orig"].append(sample_choose_orig)

        oracle_candidate_score, cand_choose_orig = choose_candidate_oracle(base, orig_score, ours_score, top_indices, true_indices)
        results["OracleCandidate"]["ranks"].append(base.rank_of_true(oracle_candidate_score, true_indices))
        results["OracleCandidate"]["times"].append(max(orig_ms, ours_ms))
        results["OracleCandidate"]["prompts"].append(max(orig_prompts, ours_prompts))
        results["OracleCandidate"]["hop2_activation_sample"].append(bool(orig_extra["hop2_activated"] or ours_extra["hop2_activated"]))
        results["OracleCandidate"]["hop2_activation_candidate"].append(max(float(orig_extra["candidate_level_activation_rate"]), float(ours_extra["candidate_level_activation_rate"])))
        results["OracleCandidate"]["alphas"].append(alpha_dynamic)
        results["OracleCandidate"]["route_choose_orig"].append(cand_choose_orig)

    print_tables(base, results, "Oracle Upper-Bound Analysis")
    save_results(
        base,
        results,
        output_json,
        meta={
            "oracle_type": "sample_and_candidate_upper_bound",
            "head_a": "Selective2Hop_OriginalAgg",
            "head_b": "Selective2Hop_Ours",
            "skipped_count": skipped_count,
            "top_k": base.TOP_K,
        },
    )
