# -*- coding: utf-8 -*-
import importlib.util
import itertools
import json
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_base(module_path):
    spec = importlib.util.spec_from_file_location("profile37_base_module", module_path)
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
            "ranks": [], "times": [], "prompts": [], "hop2_activation_sample": [], "hop2_activation_candidate": []
        },
        "Selective2Hop_Ours": {
            "ranks": [], "times": [], "prompts": [], "hop2_activation_sample": [], "hop2_activation_candidate": []
        },
        "OracleSample": {
            "ranks": [], "times": [], "prompts": [], "hop2_activation_sample": [], "hop2_activation_candidate": [], "route_choose_orig": []
        },
        "OracleCandidate": {
            "ranks": [], "times": [], "prompts": [], "hop2_activation_sample": [], "hop2_activation_candidate": [], "route_choose_orig": []
        },
    }


def print_main_tables(base, results, title):
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
            sample_rate = f"{np.mean(results[name]['hop2_activation_sample']) * 100:.1f}%"
            candidate_rate = f"{np.mean(results[name]['hop2_activation_candidate']) * 100:.1f}%"
        if name.startswith("Oracle"):
            route_rate = f"{np.mean(results[name]['route_choose_orig']) * 100:.1f}%"
        print(
            f"{name:<24} | {sample_rate:<24} | {candidate_rate:<27} | "
            f"{np.mean(results[name]['prompts']):<12.1f} | {np.mean(results[name]['times']):<14.1f} | "
            f"{metrics[name][0]:<8.2f} | {metrics[name][3]:<8.2f} | {route_rate:<12}"
        )


def choose_whole_head(base, orig_score, ours_score, true_indices):
    true_idx = int(true_indices[0])
    orig_rank = base.rank_of_true(orig_score, true_indices)
    ours_rank = base.rank_of_true(ours_score, true_indices)
    if orig_rank < ours_rank:
        return orig_score.clone(), 1.0
    if ours_rank < orig_rank:
        return ours_score.clone(), 0.0
    if score_gap(orig_score, true_idx) >= score_gap(ours_score, true_idx):
        return orig_score.clone(), 1.0
    return ours_score.clone(), 0.0


def choose_candidate_oracle(base, orig_score, ours_score, top_indices, true_indices):
    true_idx = int(true_indices[0])
    best_key = None
    best = None
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
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "score": final_score,
                "choose_orig_ratio": float(choose_orig / max(1, len(top_indices))),
                "mask_bits": list(mask_bits),
                "gap": gap,
            }
    return best


def bucket_summary(df, col, bins, labels=None):
    if df.empty:
        return []
    ser = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True, duplicates="drop")
    view = df.copy()
    view["bucket"] = ser.astype(str)
    out = []
    for bucket, group in view.groupby("bucket"):
        if bucket == "nan":
            continue
        out.append(
            {
                "bucket": bucket,
                "count": int(len(group)),
                "choose_orig_ratio": float(group["choose_orig"].mean()),
                "true_candidate_ratio": float(group["is_true_candidate"].mean()),
                "mean_flip_penalty": float(group["flip_penalty"].mean()),
            }
        )
    return out


def top_records(df, group_col, sort_cols, n=15):
    if df.empty:
        return []
    grouped = (
        df.groupby(group_col)
        .agg(
            count=("choose_orig", "size"),
            choose_orig_ratio=("choose_orig", "mean"),
            mean_flip_penalty=("flip_penalty", "mean"),
            true_candidate_ratio=("is_true_candidate", "mean"),
        )
        .reset_index()
        .sort_values(sort_cols, ascending=[False] * len(sort_cols))
        .head(n)
    )
    return grouped.to_dict(orient="records")


def print_profile_block(name, rows):
    print("\n" + "-" * 120)
    print(name)
    print("-" * 120)
    if not rows:
        print("No rows")
        return
    for row in rows:
        line = " | ".join([f"{k}={v}" for k, v in row.items()])
        print(line)


def save_payload(output_json, payload):
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def run_oracle_profile(module_path, output_dir, launch_title):
    base = load_base(module_path)
    os.makedirs(output_dir, exist_ok=True)
    print(launch_title)
    print(f"Device: {base.DEVICE}")
    print("Oracle profile diagnostics over OriginalAgg and Ours")

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
    candidate_records = []
    sample_records = []

    for sample in tqdm(base.iter_samples(dataset), total=dataset["total"], desc="OracleProfileCollect"):
        try:
            t_audio = time.time()
            audio_embed = F.normalize(get_audio_embedding(base, clap_model, sample["audio_path"], base.DEVICE), dim=-1)
            audio_ms = (time.time() - t_audio) * 1000.0
        except Exception:
            skipped_count += 1
            continue

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[: base.TOP_K]
        true_indices = sample["true_indices"]
        sample_id = sample["audio_path"]
        true_label = label_classes[int(true_indices[0])]
        alpha_dynamic = base.static_alpha_from_maxsim(float(torch.max(cos_sim_orig).item()))

        base_score, base_prompt_count = base.run_baseline(cos_sim_orig)
        results["Baseline"]["ranks"].append(base.rank_of_true(base_score, true_indices))
        results["Baseline"]["times"].append(audio_ms)
        results["Baseline"]["prompts"].append(base_prompt_count)

        t_ik = time.time()
        ik_score, ik_prompts = base.run_iknow(
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
        results["iKnow"]["ranks"].append(base.rank_of_true(ik_score, true_indices))
        results["iKnow"]["times"].append(audio_ms + (time.time() - t_ik) * 1000.0)
        results["iKnow"]["prompts"].append(ik_prompts)

        t_full = time.time()
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
        results["Full2Hop"]["ranks"].append(base.rank_of_true(full_score, true_indices))
        results["Full2Hop"]["times"].append(audio_ms + (time.time() - t_full) * 1000.0)
        results["Full2Hop"]["prompts"].append(full_prompts)

        t_orig = time.time()
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
        orig_ms = audio_ms + (time.time() - t_orig) * 1000.0
        results["Selective2Hop_OriginalAgg"]["ranks"].append(base.rank_of_true(orig_score, true_indices))
        results["Selective2Hop_OriginalAgg"]["times"].append(orig_ms)
        results["Selective2Hop_OriginalAgg"]["prompts"].append(orig_prompts)
        results["Selective2Hop_OriginalAgg"]["hop2_activation_sample"].append(float(orig_extra["hop2_activated"]))
        results["Selective2Hop_OriginalAgg"]["hop2_activation_candidate"].append(float(orig_extra["candidate_level_activation_rate"]))

        t_ours = time.time()
        ours_ret = base.run_selective2hop(
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
        if len(ours_ret) == 4:
            ours_score, ours_prompts, ours_extra, _ = ours_ret
        else:
            ours_score, ours_prompts, ours_extra = ours_ret
        ours_ms = audio_ms + (time.time() - t_ours) * 1000.0
        results["Selective2Hop_Ours"]["ranks"].append(base.rank_of_true(ours_score, true_indices))
        results["Selective2Hop_Ours"]["times"].append(ours_ms)
        results["Selective2Hop_Ours"]["prompts"].append(ours_prompts)
        results["Selective2Hop_Ours"]["hop2_activation_sample"].append(float(ours_extra["hop2_activated"]))
        results["Selective2Hop_Ours"]["hop2_activation_candidate"].append(float(ours_extra["candidate_level_activation_rate"]))

        sample_oracle_score, sample_choose_orig = choose_whole_head(base, orig_score, ours_score, true_indices)
        candidate_oracle = choose_candidate_oracle(base, orig_score, ours_score, top_indices, true_indices)
        candidate_oracle_score = candidate_oracle["score"]
        candidate_choose_ratio = candidate_oracle["choose_orig_ratio"]
        mask_bits = candidate_oracle["mask_bits"]

        results["OracleSample"]["ranks"].append(base.rank_of_true(sample_oracle_score, true_indices))
        results["OracleSample"]["times"].append(max(orig_ms, ours_ms))
        results["OracleSample"]["prompts"].append(orig_prompts)
        results["OracleSample"]["hop2_activation_sample"].append(float(orig_extra["hop2_activated"]))
        results["OracleSample"]["hop2_activation_candidate"].append(float(orig_extra["candidate_level_activation_rate"]))
        results["OracleSample"]["route_choose_orig"].append(sample_choose_orig)

        results["OracleCandidate"]["ranks"].append(base.rank_of_true(candidate_oracle_score, true_indices))
        results["OracleCandidate"]["times"].append(max(orig_ms, ours_ms))
        results["OracleCandidate"]["prompts"].append(orig_prompts)
        results["OracleCandidate"]["hop2_activation_sample"].append(float(orig_extra["hop2_activated"]))
        results["OracleCandidate"]["hop2_activation_candidate"].append(float(orig_extra["candidate_level_activation_rate"]))
        results["OracleCandidate"]["route_choose_orig"].append(candidate_choose_ratio)

        base_top1 = float(cos_sim_orig[top_indices[0]].item())
        base_margin = float(cos_sim_orig[top_indices[0]].item() - cos_sim_orig[top_indices[1]].item()) if len(top_indices) > 1 else 0.0
        ent = normalized_entropy(cos_sim_orig, top_indices)
        oracle_gap = score_gap(candidate_oracle_score, int(true_indices[0]))
        mixed_preference = 1.0 if 0.0 < candidate_choose_ratio < 1.0 else 0.0

        sample_records.append(
            {
                "sample_id": sample_id,
                "true_label": true_label,
                "baseline_top1": base_top1,
                "baseline_margin": base_margin,
                "entropy": ent,
                "orig_rank": base.rank_of_true(orig_score, true_indices),
                "ours_rank": base.rank_of_true(ours_score, true_indices),
                "oracle_sample_rank": base.rank_of_true(sample_oracle_score, true_indices),
                "oracle_candidate_rank": base.rank_of_true(candidate_oracle_score, true_indices),
                "oracle_sample_choose_orig": float(sample_choose_orig),
                "oracle_candidate_choose_orig_ratio": float(candidate_choose_ratio),
                "mixed_preference": mixed_preference,
                "hop2_activation": float(orig_extra["hop2_activated"]),
                "prompt_count_log1p": float(np.log1p(orig_prompts)),
                "oracle_gap": float(oracle_gap),
            }
        )

        for ci, bit in zip(top_indices, mask_bits):
            candidate_label = label_classes[int(ci)]
            is_true_candidate = 1.0 if int(ci) in set([int(x) for x in true_indices]) else 0.0
            base_i = float(cos_sim_orig[ci].item())
            orig_i = float(orig_score[ci].item())
            ours_i = float(ours_score[ci].item())
            choose_orig = 1.0 if bit == 1 else 0.0
            flipped = candidate_oracle_score.clone()
            flipped[ci] = ours_score[ci] if bit == 1 else orig_score[ci]
            flip_penalty = float(oracle_gap - score_gap(flipped, int(true_indices[0])))
            candidate_records.append(
                {
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "candidate_label": candidate_label,
                    "is_true_candidate": is_true_candidate,
                    "choose_orig": choose_orig,
                    "baseline_top1": base_top1,
                    "baseline_margin": base_margin,
                    "entropy": ent,
                    "base_i": base_i,
                    "orig_i": orig_i,
                    "ours_i": ours_i,
                    "orig_minus_base_i": float(orig_i - base_i),
                    "ours_minus_base_i": float(ours_i - base_i),
                    "orig_minus_ours_i": float(orig_i - ours_i),
                    "hop2_activation": float(orig_extra["hop2_activated"]),
                    "prompt_count_log1p": float(np.log1p(orig_prompts)),
                    "flip_penalty": flip_penalty,
                }
            )

    metrics = {name: base.compute_metrics(results[name]["ranks"]) for name in results}
    sample_df = pd.DataFrame(
        sample_records,
        columns=[
            "sample_id",
            "true_label",
            "baseline_top1",
            "baseline_margin",
            "entropy",
            "orig_rank",
            "ours_rank",
            "oracle_sample_rank",
            "oracle_candidate_rank",
            "oracle_sample_choose_orig",
            "oracle_candidate_choose_orig_ratio",
            "mixed_preference",
            "hop2_activation",
            "prompt_count_log1p",
            "oracle_gap",
        ],
    )
    cand_df = pd.DataFrame(
        candidate_records,
        columns=[
            "sample_id",
            "true_label",
            "candidate_label",
            "is_true_candidate",
            "choose_orig",
            "baseline_top1",
            "baseline_margin",
            "entropy",
            "base_i",
            "orig_i",
            "ours_i",
            "orig_minus_base_i",
            "ours_minus_base_i",
            "orig_minus_ours_i",
            "hop2_activation",
            "prompt_count_log1p",
            "flip_penalty",
        ],
    )

    feature_profiles = {
        "baseline_margin": bucket_summary(
            cand_df,
            "baseline_margin",
            bins=[-1e9, 0.02, 0.05, 0.10, 0.15, 1e9],
        ),
        "entropy": bucket_summary(
            cand_df,
            "entropy",
            bins=[-1e9, 0.2, 0.4, 0.6, 0.8, 1.1],
        ),
        "orig_minus_base_i": bucket_summary(
            cand_df,
            "orig_minus_base_i",
            bins=[-1e9, -0.05, -0.02, 0.0, 0.02, 0.05, 1e9],
        ),
        "ours_minus_base_i": bucket_summary(
            cand_df,
            "ours_minus_base_i",
            bins=[-1e9, -0.05, -0.02, 0.0, 0.02, 0.05, 1e9],
        ),
        "orig_minus_ours_i": bucket_summary(
            cand_df,
            "orig_minus_ours_i",
            bins=[-1e9, -0.05, -0.02, 0.0, 0.02, 0.05, 1e9],
        ),
    }

    semantic_profiles = {
        "true_label_profile": top_records(sample_df.assign(choose_orig=sample_df["oracle_sample_choose_orig"], is_true_candidate=1.0, flip_penalty=0.0), "true_label", ["choose_orig_ratio", "count"], n=50),
        "candidate_label_profile": top_records(cand_df, "candidate_label", ["choose_orig_ratio", "count"], n=50),
    }

    error_tolerance = {
        "by_choice": (
            cand_df.groupby("choose_orig")
            .agg(
                count=("choose_orig", "size"),
                mean_flip_penalty=("flip_penalty", "mean"),
                median_flip_penalty=("flip_penalty", "median"),
                p75_flip_penalty=("flip_penalty", lambda s: float(np.quantile(s, 0.75))),
                p90_flip_penalty=("flip_penalty", lambda s: float(np.quantile(s, 0.90))),
            )
            .reset_index()
            .to_dict(orient="records")
            if not cand_df.empty
            else []
        ),
        "high_risk_candidate_labels": top_records(cand_df, "candidate_label", ["mean_flip_penalty", "count"], n=20),
    }

    extra_summary = {
        "candidate_choose_orig_ratio": float(cand_df["choose_orig"].mean()) if not cand_df.empty else 0.0,
        "true_candidate_choose_orig_ratio": float(cand_df[cand_df["is_true_candidate"] > 0.5]["choose_orig"].mean()) if not cand_df[cand_df["is_true_candidate"] > 0.5].empty else 0.0,
        "mixed_preference_ratio": float(sample_df["mixed_preference"].mean()) if not sample_df.empty else 0.0,
        "sample_choose_orig_ratio": float(sample_df["oracle_sample_choose_orig"].mean()) if not sample_df.empty else 0.0,
    }

    print_main_tables(base, results, launch_title + " Final Tables")
    print_profile_block("Feature profile: baseline_margin", feature_profiles["baseline_margin"])
    print_profile_block("Feature profile: entropy", feature_profiles["entropy"])
    print_profile_block("Feature profile: orig_minus_base_i", feature_profiles["orig_minus_base_i"])
    print_profile_block("Feature profile: ours_minus_base_i", feature_profiles["ours_minus_base_i"])
    print_profile_block("Feature profile: orig_minus_ours_i", feature_profiles["orig_minus_ours_i"])
    print_profile_block("Semantic profile: true_label", semantic_profiles["true_label_profile"][:15])
    print_profile_block("Semantic profile: candidate_label", semantic_profiles["candidate_label_profile"][:15])
    print_profile_block("Error tolerance by choice", error_tolerance["by_choice"])
    print_profile_block("High-risk candidate labels", error_tolerance["high_risk_candidate_labels"][:15])
    print("\nExtra summary")
    for k, v in extra_summary.items():
        print(f"{k}: {v}")

    candidate_csv = os.path.join(output_dir, "candidate_records.csv")
    sample_csv = os.path.join(output_dir, "sample_records.csv")
    cand_df.to_csv(candidate_csv, index=False, encoding="utf-8-sig")
    sample_df.to_csv(sample_csv, index=False, encoding="utf-8-sig")

    payload = {
        "metrics": {
            name: {
                "Hit@1": metrics[name][0],
                "Hit@3": metrics[name][1],
                "Hit@5": metrics[name][2],
                "MRR": metrics[name][3],
            }
            for name in results
        },
        "stats": {},
        "profiles": {
            "feature_profiles": feature_profiles,
            "semantic_profiles": semantic_profiles,
            "error_tolerance": error_tolerance,
            "extra_summary": extra_summary,
        },
        "meta": {
            "oracle_type": "profile_only",
            "head_a": "Selective2Hop_OriginalAgg",
            "head_b": "Selective2Hop_Ours",
            "top_k": int(base.TOP_K),
            "skipped_count": int(skipped_count),
            "candidate_records_csv": candidate_csv,
            "sample_records_csv": sample_csv,
        },
    }
    for name in results:
        payload["stats"][name] = {
            "avg_prompts": float(np.mean(results[name]["prompts"])) if results[name]["prompts"] else 0.0,
            "avg_time_ms": float(np.mean(results[name]["times"])) if results[name]["times"] else 0.0,
        }
        if "hop2_activation_sample" in results[name]:
            payload["stats"][name]["hop2_activation_sample"] = float(np.mean(results[name]["hop2_activation_sample"])) if results[name]["hop2_activation_sample"] else 0.0
            payload["stats"][name]["hop2_activation_candidate"] = float(np.mean(results[name]["hop2_activation_candidate"])) if results[name]["hop2_activation_candidate"] else 0.0
        if "route_choose_orig" in results[name]:
            payload["stats"][name]["route_choose_orig"] = float(np.mean(results[name]["route_choose_orig"])) if results[name]["route_choose_orig"] else 0.0

    save_payload(os.path.join(output_dir, "results_oracle_profile.json"), payload)
