import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


DATASET_CONFIGS = {
    "esc50": {
        "module_path": "/data/zkx/zkx/iknow-audio/17.消融/esc50/test_ablation.py",
        "output_dir": "/data/zkx/zkx/iknow-audio/60.graphprop_ablation17/esc50",
        "device_id": 0,
    },
    "usk80": {
        "module_path": "/data/zkx/zkx/iknow-audio/17.消融/usk80/test_ablation.py",
        "output_dir": "/data/zkx/zkx/iknow-audio/60.graphprop_ablation17/usk80",
        "device_id": 1,
    },
    "dcase": {
        "module_path": "/data/zkx/zkx/iknow-audio/17.消融/dcase/test_ablation.py",
        "output_dir": "/data/zkx/zkx/iknow-audio/60.graphprop_ablation17/dcase",
        "device_id": 2,
    },
}


GRAPH_PROP_LAMBDA = 0.30


def load_module(module_path: str):
    spec = importlib.util.spec_from_file_location("ablation17_base", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_h2_records(module, h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map, use_llm):
    records = []
    for head_norm in h1_map.keys():
        for rel in hop2_relations:
            for tail in get_tails_fn(head_norm, rel):
                tail_norm = tail.lower().strip()
                if tail_norm == class_name.lower() or tail_norm in class_labels_set or tail_norm in h1_map:
                    continue
                records.append(
                    {
                        "parent_h1": head_norm,
                        "tail_norm": tail_norm,
                        "prompt": module.resolve_prompt(prompt_map, head_norm, rel, tail_norm, class_name, tail, use_llm),
                    }
                )
    return records


def build_normalized_adjacency(num_h1: int, h2_records):
    n = 1 + num_h1 + len(h2_records)
    a = torch.zeros((n, n), dtype=torch.float32)

    # Self-loop is mandatory; otherwise each node loses its own score entirely.
    a += torch.eye(n, dtype=torch.float32)

    # Connect the class/base node with all hop1 nodes.
    for h1_idx in range(num_h1):
        node_idx = 1 + h1_idx
        a[0, node_idx] = 1.0
        a[node_idx, 0] = 1.0

    # Connect each hop2 node with its parent hop1 node.
    for local_h2_idx, rec in enumerate(h2_records):
        parent_h1_idx = rec["h1_index"]
        h1_node_idx = 1 + parent_h1_idx
        h2_node_idx = 1 + num_h1 + local_h2_idx
        a[h1_node_idx, h2_node_idx] = 1.0
        a[h2_node_idx, h1_node_idx] = 1.0

    degree = torch.sum(a, dim=1)
    degree_inv_sqrt = torch.pow(torch.clamp(degree, min=1e-8), -0.5)
    d_inv_sqrt = torch.diag(degree_inv_sqrt)
    a_hat = d_inv_sqrt @ a @ d_inv_sqrt
    return a_hat


def propagate_prompt_scores(base_score, num_h1, h2_records, s1, s2):

    a_hat = build_normalized_adjacency(num_h1, h2_records).to(base_score.device)

    if s2.numel() == 0:
        s0 = torch.cat([base_score.unsqueeze(0), s1], dim=0)
    else:
        s0 = torch.cat([base_score.unsqueeze(0), s1, s2], dim=0)

    # Residual / PPR-style propagation to avoid oversmoothing strong event peaks.
    s_star = (1.0 - GRAPH_PROP_LAMBDA) * s0 + GRAPH_PROP_LAMBDA * torch.matmul(a_hat, s0)

    prompt_scores = s_star[1:]
    return prompt_scores


def run_selective_graphprop(
    module,
    clap_model,
    audio_embed,
    cos_sim_orig,
    top_indices,
    label_classes,
    kg_classes,
    class_labels_set,
    hop1_relations,
    hop2_relations,
    get_tails_fn,
    prompt_map,
    alpha_dynamic,
    use_llm,
):
    score = cos_sim_orig.clone()
    prompt_count = 0
    hop2_flags = []

    for ci in top_indices:
        class_name = label_classes[ci]
        kg_ent = module.get_kg_entity(kg_classes[ci])
        tau = cos_sim_orig[ci].item() + module.RELATIVE_MARGIN

        h1_map = module.build_h1_map(
            kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, prompt_map, use_llm
        )
        h1_items = list(h1_map.items())
        s1 = module.score_prompt_list(clap_model, audio_embed, [v for _, v in h1_items])
        max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0

        if s1.numel() > 0 and max_h1 >= tau:
            hop2_flags.append(False)
            best_scores = module.safe_topk(s1, module.TOP_P)
            score[ci] = module.aggregate_candidate(cos_sim_orig[ci], best_scores, alpha_dynamic, "dynamic")
            prompt_count += len(h1_map)
            continue

        hop2_flags.append(True)

        h2_records = build_h2_records(
            module, h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map, use_llm
        )
        for rec in h2_records:
            rec["h1_index"] = next(i for i, (head_norm, _) in enumerate(h1_items) if head_norm == rec["parent_h1"])

        s2 = module.score_prompt_list(clap_model, audio_embed, [rec["prompt"] for rec in h2_records])
        propagated_prompt_scores = propagate_prompt_scores(cos_sim_orig[ci], len(h1_items), h2_records, s1, s2)
        best_scores = module.safe_topk(propagated_prompt_scores, module.TOP_P)
        score[ci] = module.aggregate_candidate(cos_sim_orig[ci], best_scores, alpha_dynamic, "dynamic")
        prompt_count += len(h1_map) + len(h2_records)

    extras = {
        "hop2_activated": any(hop2_flags),
        "candidate_level_activation_rate": float(np.mean(hop2_flags)) if hop2_flags else 0.0,
        "graph_lambda": GRAPH_PROP_LAMBDA,
    }
    return score, prompt_count, extras


def init_results(module):
    results = module.init_results()
    results["Ours_GraphProp"] = {
        "ranks": [],
        "times": [],
        "prompts": [],
        "hop2_activation_sample": [],
        "hop2_activation_candidate": [],
        "alphas": [],
        "graph_lambda": [],
    }
    return results


def normalize_progress(module, progress, total_samples):
    fresh = {"next_index": 0, "total_samples": total_samples, "results": init_results(module)}
    if not progress:
        return fresh
    fresh["next_index"] = int(progress.get("next_index", 0))
    old_results = progress.get("results", {})
    for method, default_val in fresh["results"].items():
        if method in old_results and isinstance(old_results[method], dict):
            for key, default_list in default_val.items():
                fresh["results"][method][key] = old_results[method].get(key, default_list)
    return fresh


def compute_stats(module, results):
    stats = module.compute_stats(results)
    if "Ours_GraphProp" in results:
        vals = results["Ours_GraphProp"]["graph_lambda"]
        stats["Ours_GraphProp"]["graph_lambda"] = float(np.mean(vals)) if vals else GRAPH_PROP_LAMBDA
    return stats


def save_results(module, results, total_samples, completed):
    metrics = {name: module.compute_metrics(values["ranks"]) for name, values in results.items()}
    stats = compute_stats(module, results)
    payload = {"completed": completed, "total_samples": total_samples, "metrics": {}, "stats": stats}
    for method in results:
        payload["metrics"][method] = {
            "Hit@1": metrics[method][0],
            "Hit@3": metrics[method][1],
            "Hit@5": metrics[method][2],
            "MRR": metrics[method][3],
        }
    with open(module.OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_tables(module, results):
    metrics = {name: module.compute_metrics(results[name]["ranks"]) for name in results}
    stats = compute_stats(module, results)

    print("\n" + "=" * 156)
    print(f"{'Metric':<8} | {'Baseline':<12} | {'iKnow':<12} | {'Full2Hop':<14} | {'Ours':<12} | {'Ours_GraphProp':<16}")
    print("-" * 156)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(
            f"{metric_name:<8} | {metrics['Baseline'][idx]:<12.2f} | {metrics['iKnow'][idx]:<12.2f} | "
            f"{metrics['Full2Hop'][idx]:<14.2f} | {metrics['Ours'][idx]:<12.2f} | {metrics['Ours_GraphProp'][idx]:<16.2f}"
        )

    print("\n" + "=" * 156)
    print(f"{'Method Ablation':<18} | {'Ours':<12} | {'Ours_GraphProp':<16} | {'Delta':<10}")
    print("-" * 156)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        delta = metrics["Ours_GraphProp"][idx] - metrics["Ours"][idx]
        print(f"{metric_name:<18} | {metrics['Ours'][idx]:<12.2f} | {metrics['Ours_GraphProp'][idx]:<16.2f} | {delta:<10.2f}")

    print("\n" + "=" * 156)
    print(
        f"{'Method':<24} | {'Hop2 activation(sample)':<24} | {'Hop2 activation(candidate)':<27} | "
        f"{'Avg prompts':<12} | {'Avg time (ms)':<14} | {'Graph λ':<8}"
    )
    print("-" * 156)
    for name in ["Ours", "Ours_GraphProp"]:
        graph_lambda = "N/A" if name == "Ours" else f"{stats[name].get('graph_lambda', GRAPH_PROP_LAMBDA):.2f}"
        print(
            f"{name:<24} | {stats[name]['hop2_activation_sample'] * 100:<24.1f}% | "
            f"{stats[name]['hop2_activation_candidate'] * 100:<27.1f}% | "
            f"{stats[name]['avg_prompts']:<12.1f} | {stats[name]['avg_time_ms']:<14.1f} | {graph_lambda:<8}"
        )


@torch.no_grad()
def run_dataset(dataset_key: str):
    cfg = DATASET_CONFIGS[dataset_key]
    module = load_module(cfg["module_path"])

    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    module.OUTPUT_JSON = str(outdir / "results_graphprop.json")
    module.PROGRESS_JSON = str(outdir / "progress_graphprop.json")

    print(f"Starting {dataset_key} GraphProp ablation on top of 17.ablation logic")
    print(f"Device: {module.DEVICE}")
    print(f"Graph propagation lambda: {GRAPH_PROP_LAMBDA}")
    print("Method focus: Ours vs Ours_GraphProp")

    prompt_map = module.load_llm_prompts(module.LLM_PROMPTS_PATH)
    clap_model = module.CLAP(version="2023", use_cuda=torch.cuda.is_available())
    with module.warnings.catch_warnings():
        module.warnings.simplefilter("ignore")
        kge_model = module.torch.load(module.os.path.join(module.KGE_MODEL_DIR, "trained_model.pkl"), map_location=module.DEVICE)
    kge_model.eval()
    training_factory = module.TriplesFactory.from_path(module.TRAIN_TRIPLES_PATH)

    hop1_relations = [rel for rel in module.HOP1_RELATIONS if rel in training_factory.relation_to_id]
    hop2_relations = [rel for rel in module.HOP2_RELATIONS if rel in training_factory.relation_to_id]
    print(f"Valid hop1 relations: {hop1_relations}")
    print(f"Valid hop2 relations: {hop2_relations}")

    get_tails = module.build_tail_predictor(kge_model, training_factory)
    dataset = module.load_dataset()
    label_classes = dataset["label_classes"]
    kg_classes = dataset["kg_classes"]
    class_labels_set = dataset["class_labels_set"]
    samples = list(module.iter_samples(dataset))
    total_samples = len(samples)
    print(f"Total samples prepared: {total_samples}")

    if module.os.path.exists(module.OUTPUT_JSON):
        try:
            with open(module.OUTPUT_JSON, "r", encoding="utf-8") as f:
                done_payload = json.load(f)
            if done_payload.get("completed") and int(done_payload.get("total_samples", -1)) == total_samples:
                print(f"Found completed results: {module.OUTPUT_JSON}")
                return
        except Exception:
            pass

    if not module.os.path.exists(module.PROGRESS_JSON):
        progress = {"next_index": 0, "total_samples": total_samples, "results": init_results(module)}
    else:
        try:
            with open(module.PROGRESS_JSON, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = None
        progress = normalize_progress(module, payload, total_samples)
        if progress["next_index"] > total_samples:
            progress["next_index"] = 0

    start_idx = progress["next_index"]
    results = progress["results"]
    print(f"Resume from sample index: {start_idx}")
    text_embeds = module.F.normalize(module.get_safe_text_embeddings(clap_model, label_classes, module.DEVICE), dim=-1)

    def save_progress():
        tmp_path = module.PROGRESS_JSON + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False)
        module.os.replace(tmp_path, module.PROGRESS_JSON)

    for sample_idx in module.tqdm(range(start_idx, total_samples), total=total_samples, initial=start_idx, desc=f"{dataset_key} graphprop"):
        sample = samples[sample_idx]
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]

        if not module.os.path.exists(audio_path) or not true_indices:
            progress["next_index"] = sample_idx + 1
            save_progress()
            continue

        try:
            t_audio_start = time.time()
            audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = module.F.normalize(module.to_tensor(audio_emb_raw).to(module.DEVICE).float(), dim=-1)
            audio_ms = (time.time() - t_audio_start) * 1000.0
        except Exception:
            progress["next_index"] = sample_idx + 1
            save_progress()
            continue

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[: module.TOP_K]
        alpha_dynamic = module.instance_alpha(torch.max(cos_sim_orig).item())

        t = time.time()
        score, prompt_count = module.run_baseline(cos_sim_orig)
        results["Baseline"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["Baseline"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Baseline"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count = module.run_iknow(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, get_tails
        )
        results["iKnow"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["iKnow"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["iKnow"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count = module.run_full2hop_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, True, "dynamic"
        )
        results["Full2Hop"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["Full2Hop"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Full2Hop"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count, extras = module.run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, True, "dynamic"
        )
        results["Ours"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["Ours"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Ours"]["prompts"].append(prompt_count)
        results["Ours"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Ours"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Ours"]["alphas"].append(alpha_dynamic)

        t = time.time()
        score, prompt_count, extras = run_selective_graphprop(
            module, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, True
        )
        results["Ours_GraphProp"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["Ours_GraphProp"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Ours_GraphProp"]["prompts"].append(prompt_count)
        results["Ours_GraphProp"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Ours_GraphProp"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Ours_GraphProp"]["alphas"].append(alpha_dynamic)
        results["Ours_GraphProp"]["graph_lambda"].append(float(extras["graph_lambda"]))

        t = time.time()
        score, prompt_count, extras = module.run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, True, "original"
        )
        results["Selective2Hop_OriginalAgg"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["Selective2Hop_OriginalAgg"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Selective2Hop_OriginalAgg"]["prompts"].append(prompt_count)
        results["Selective2Hop_OriginalAgg"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Selective2Hop_OriginalAgg"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Selective2Hop_OriginalAgg"]["alphas"].append(alpha_dynamic)

        t = time.time()
        score, prompt_count, extras = module.run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, False, "dynamic"
        )
        results["Selective2Hop_NoLLM"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["Selective2Hop_NoLLM"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Selective2Hop_NoLLM"]["prompts"].append(prompt_count)
        results["Selective2Hop_NoLLM"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Selective2Hop_NoLLM"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Selective2Hop_NoLLM"]["alphas"].append(alpha_dynamic)

        progress["next_index"] = sample_idx + 1
        save_progress()

    progress["next_index"] = total_samples
    save_progress()
    save_results(module, results, total_samples, True)
    print_tables(module, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS.keys()), required=True)
    args = parser.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
