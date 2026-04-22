# -*- coding: utf-8 -*-
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

BASE_SCRIPT = "/data/zkx/zkx/iknow-audio/26.sigmoid_moe/tut2017/test_sigmoid_moe.py"


def load_base():
    spec = importlib.util.spec_from_file_location("tut28_base_module", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        "Experiment": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "alphas": [],
        },
    }


def print_tables(base, results, exp_label):
    metrics = {name: base.compute_metrics(results[name]["ranks"]) for name in results}
    order = ["Baseline", "iKnow", "Full2Hop", "Selective2Hop_OriginalAgg", "Experiment"]
    labels = {
        "Baseline": "Baseline",
        "iKnow": "iKnow",
        "Full2Hop": "Full2Hop",
        "Selective2Hop_OriginalAgg": "Sel2 OriginalAgg",
        "Experiment": exp_label,
    }
    print("\n" + "=" * 170)
    print(
        f"{'Metric':<8} | {'Baseline':<10} | {'iKnow':<10} | {'Full2Hop':<12} | "
        f"{'Sel2 OriginalAgg':<18} | {exp_label:<24}"
    )
    print("-" * 170)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(
            f"{metric_name:<8} | "
            f"{metrics['Baseline'][idx]:<10.2f} | "
            f"{metrics['iKnow'][idx]:<10.2f} | "
            f"{metrics['Full2Hop'][idx]:<12.2f} | "
            f"{metrics['Selective2Hop_OriginalAgg'][idx]:<18.2f} | "
            f"{metrics['Experiment'][idx]:<24.2f}"
        )

    print("\n" + "=" * 170)
    print(
        f"{'Method':<24} | {'Hop2 activation(sample)':<24} | {'Hop2 activation(candidate)':<27} | "
        f"{'Avg prompts':<12} | {'Avg time (ms)':<14} | {'Hit@1':<8} | {'MRR':<8}"
    )
    print("-" * 170)
    for name in order:
        sample_rate = "N/A"
        candidate_rate = "N/A"
        if name in ["Selective2Hop_OriginalAgg", "Experiment"]:
            sample_rate = f"{np.mean(results[name]['hop2_activation_sample']) * 100:.1f}%"
            candidate_rate = f"{np.mean(results[name]['hop2_activation_candidate']) * 100:.1f}%"
        print(
            f"{labels[name]:<24} | {sample_rate:<24} | {candidate_rate:<27} | "
            f"{np.mean(results[name]['prompts']):<12.1f} | {np.mean(results[name]['times']):<14.1f} | "
            f"{metrics[name][0]:<8.2f} | {metrics[name][3]:<8.2f}"
        )


def save_results(base, results, exp_label, output_json, extra_meta=None):
    metrics = {name: base.compute_metrics(results[name]["ranks"]) for name in results}
    payload = {"metrics": {}, "stats": {}, "experiment_label": exp_label}
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
        if name in ["Selective2Hop_OriginalAgg", "Experiment"]:
            payload["stats"][name]["hop2_activation_sample"] = float(np.mean(results[name]["hop2_activation_sample"])) if results[name]["hop2_activation_sample"] else 0.0
            payload["stats"][name]["hop2_activation_candidate"] = float(np.mean(results[name]["hop2_activation_candidate"])) if results[name]["hop2_activation_candidate"] else 0.0
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def run_tut_experiment(exp_title, exp_label, output_json, experiment_runner, extra_meta=None, setup_hook=None):
    base = load_base()
    print(exp_title)
    print(f"Device: {base.DEVICE}")
    prompt_map = base.load_llm_prompts(base.LLM_PROMPTS_PATH)
    context = {}
    if setup_hook is not None:
        context = setup_hook(base, prompt_map) or {}

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

    for sample in tqdm(base.iter_samples(dataset), total=dataset["total"], desc=exp_label):
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
        alpha_static = base.static_alpha_from_maxsim(torch.max(cos_sim_orig).item())

        t = time.time()
        score, prompt_count = base.run_baseline(cos_sim_orig)
        results["Baseline"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Baseline"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Baseline"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count = base.run_iknow(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes,
            kg_classes, class_labels_set, hop1_relations, get_tails
        )
        results["iKnow"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["iKnow"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["iKnow"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count = base.run_full2hop(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, alpha_static
        )
        results["Full2Hop"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Full2Hop"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Full2Hop"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count, extras = base.run_selective2hop_originalagg(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map
        )
        results["Selective2Hop_OriginalAgg"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Selective2Hop_OriginalAgg"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Selective2Hop_OriginalAgg"]["prompts"].append(prompt_count)
        results["Selective2Hop_OriginalAgg"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Selective2Hop_OriginalAgg"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Selective2Hop_OriginalAgg"]["alphas"].append(0.0)

        t = time.time()
        score, prompt_count, extras = experiment_runner(
            base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, context
        )
        results["Experiment"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Experiment"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Experiment"]["prompts"].append(prompt_count)
        results["Experiment"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Experiment"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Experiment"]["alphas"].append(float(extras.get("alpha_mean", 0.0)))

    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid samples.")
    print_tables(base, results, exp_label)
    save_results(base, results, exp_label, output_json, extra_meta=extra_meta)
