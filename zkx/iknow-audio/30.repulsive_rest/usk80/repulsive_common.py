# -*- coding: utf-8 -*-
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPULSIVE_BETA = 0.35
REPULSIVE_MARGIN = 0.01


def load_base(module_path):
    spec = importlib.util.spec_from_file_location("repulsive_base_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_originalagg_score(base, clap_model, audio_embed, cos_sim_orig, ci, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map):
    class_name = label_classes[ci]
    kg_ent = base.get_kg_entity(kg_classes[ci])
    tau = cos_sim_orig[ci].item() + base.RELATIVE_MARGIN
    try:
        h1_map = base.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map)
    except TypeError:
        h1_map = base.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map, True)
    s1 = base.score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
    max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
    if s1.numel() > 0 and max_h1 >= tau:
        logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * base.LOGIT_SCALE, s1 * base.LOGIT_SCALE])
        score = (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / base.LOGIT_SCALE
        return score, len(h1_map), False
    try:
        h2_prompts = base.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map)
    except TypeError:
        h2_prompts = base.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map, True)
    s2 = base.score_prompt_list(clap_model, audio_embed, h2_prompts)
    all_scores = torch.cat([s1, s2 * base.DECAY_GAMMA]) if s2.numel() > 0 else s1
    if all_scores.numel() == 0:
        return cos_sim_orig[ci], len(h1_map) + len(h2_prompts), True
    logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * base.LOGIT_SCALE, all_scores * base.LOGIT_SCALE])
    score = (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / base.LOGIT_SCALE
    return score, len(h1_map) + len(h2_prompts), True


def run_repulsive(base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map):
    score = cos_sim_orig.clone()
    prompt_count = 0
    hop2_flags = []
    candidate_scores = {}
    for ci in top_indices:
        s_i, used_prompts, hop2_flag = candidate_originalagg_score(
            base, clap_model, audio_embed, cos_sim_orig, ci, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map
        )
        candidate_scores[ci] = s_i
        prompt_count += used_prompts
        hop2_flags.append(hop2_flag)
    for ci in top_indices:
        own = candidate_scores[ci]
        competitors = [float(candidate_scores[cj].item()) for cj in top_indices if cj != ci]
        comp = max(competitors) if competitors else float(own.item())
        penalty = max(0.0, comp - float(own.item()) + REPULSIVE_MARGIN)
        score[ci] = own - (REPULSIVE_BETA * penalty)
    extras = {
        "hop2_activated": any(hop2_flags),
        "candidate_level_activation_rate": float(np.mean(hop2_flags)) if hop2_flags else 0.0,
        "alpha_mean": 0.0,
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
        "Selective2Hop_Repulsive": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "alphas": [],
        },
    }


def print_tables(base, results, title):
    metrics = {name: base.compute_metrics(results[name]["ranks"]) for name in results}
    order = ["Baseline", "iKnow", "Full2Hop", "Selective2Hop_OriginalAgg", "Selective2Hop_Repulsive"]
    print("\n" + "=" * 170)
    print(title)
    print("-" * 170)
    print(f"{'Metric':<8} | {'Baseline':<10} | {'iKnow':<10} | {'Full2Hop':<12} | {'Sel2 OriginalAgg':<18} | {'Sel2 Repulsive':<18}")
    print("-" * 170)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(
            f"{metric_name:<8} | {metrics['Baseline'][idx]:<10.2f} | {metrics['iKnow'][idx]:<10.2f} | "
            f"{metrics['Full2Hop'][idx]:<12.2f} | {metrics['Selective2Hop_OriginalAgg'][idx]:<18.2f} | {metrics['Selective2Hop_Repulsive'][idx]:<18.2f}"
        )

    print("\n" + "=" * 180)
    print(f"{'Method':<24} | {'Hop2 activation(sample)':<24} | {'Hop2 activation(candidate)':<27} | {'Avg prompts':<12} | {'Avg time (ms)':<14} | {'Hit@1':<8} | {'MRR':<8}")
    print("-" * 180)
    for name in order:
        sample_rate = "N/A"
        candidate_rate = "N/A"
        if name.startswith("Selective2Hop"):
            sample_rate = f"{np.mean(results[name]['hop2_activation_sample']) * 100:.1f}%"
            candidate_rate = f"{np.mean(results[name]['hop2_activation_candidate']) * 100:.1f}%"
        print(
            f"{name:<24} | {sample_rate:<24} | {candidate_rate:<27} | "
            f"{np.mean(results[name]['prompts']):<12.1f} | {np.mean(results[name]['times']):<14.1f} | "
            f"{metrics[name][0]:<8.2f} | {metrics[name][3]:<8.2f}"
        )


def save_results(base, results, output_json, extra_meta=None):
    metrics = {name: base.compute_metrics(results[name]["ranks"]) for name in results}
    payload = {"metrics": {}, "stats": {}}
    if extra_meta:
        payload["meta"] = extra_meta
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
            payload["stats"][name]["hop2_activation_sample"] = float(np.mean(results[name]["hop2_activation_sample"])) if results[name]["hop2_activation_sample"] else 0.0
            payload["stats"][name]["hop2_activation_candidate"] = float(np.mean(results[name]["hop2_activation_candidate"])) if results[name]["hop2_activation_candidate"] else 0.0
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_full2hop_adapter(base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic):
    if hasattr(base, "run_full2hop"):
        return base.run_full2hop(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic
        )
    return base.run_full2hop_variant(
        clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
        class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map,
        alpha_dynamic, True, "dynamic"
    )


def run_originalagg_adapter(base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic):
    if hasattr(base, "run_selective2hop_originalagg"):
        return base.run_selective2hop_originalagg(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map
        )
    return base.run_selective_variant(
        clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
        class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map,
        alpha_dynamic, True, "original"
    )


@torch.no_grad()
def run_repulsive_dataset(module_path, output_json, launch_title):
    base = load_base(module_path)
    print(launch_title)
    print(f"Device: {base.DEVICE}")
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

    for sample in tqdm(base.iter_samples(dataset), total=dataset["total"], desc="RepulsiveEval"):
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
        top_indices = sorted_indices[:base.TOP_K]
        alpha_dynamic = base.instance_alpha(torch.max(cos_sim_orig).item()) if hasattr(base, "instance_alpha") else base.static_alpha_from_maxsim(torch.max(cos_sim_orig).item())

        t = time.time()
        score, prompt_count = base.run_baseline(cos_sim_orig)
        results["Baseline"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Baseline"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Baseline"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count = base.run_iknow(clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, get_tails)
        results["iKnow"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["iKnow"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["iKnow"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count = run_full2hop_adapter(
            base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes,
            kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails,
            prompt_map, alpha_dynamic
        )
        results["Full2Hop"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Full2Hop"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Full2Hop"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count, extras = run_originalagg_adapter(
            base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes,
            kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails,
            prompt_map, alpha_dynamic
        )
        results["Selective2Hop_OriginalAgg"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Selective2Hop_OriginalAgg"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Selective2Hop_OriginalAgg"]["prompts"].append(prompt_count)
        results["Selective2Hop_OriginalAgg"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Selective2Hop_OriginalAgg"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Selective2Hop_OriginalAgg"]["alphas"].append(alpha_dynamic)

        t = time.time()
        score, prompt_count, extras = run_repulsive(base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map)
        results["Selective2Hop_Repulsive"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Selective2Hop_Repulsive"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Selective2Hop_Repulsive"]["prompts"].append(prompt_count)
        results["Selective2Hop_Repulsive"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Selective2Hop_Repulsive"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Selective2Hop_Repulsive"]["alphas"].append(float(extras["alpha_mean"]))

    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid samples.")
    print_tables(base, results, launch_title)
    save_results(base, results, output_json, extra_meta={"module_path": module_path, "repulsive_beta": REPULSIVE_BETA, "repulsive_margin": REPULSIVE_MARGIN})
