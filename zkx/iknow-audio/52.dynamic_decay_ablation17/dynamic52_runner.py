import argparse
import json
import os
from pathlib import Path
import re
import types

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


CONFIGS = {
    "dyn1": {
        "tau_high": 0.05,
        "gamma": 0.00,
        "lambda_low": 0.30,
        "lambda_mid": 0.50,
        "lambda_high": 0.85,
    },
    "dyn2": {
        "tau_high": 0.08,
        "gamma": 0.00,
        "lambda_low": 0.20,
        "lambda_mid": 0.40,
        "lambda_high": 0.85,
    },
    "dyn3": {
        "tau_high": 0.05,
        "gamma": 0.02,
        "lambda_low": 0.30,
        "lambda_mid": 0.50,
        "lambda_high": 1.00,
    },
}


def _replace_assignment(source: str, name: str, value: str) -> str:
    pattern = rf"^{name}\s*=\s*.*$"
    replacement = f'{name} = "{value}"'
    return re.sub(pattern, replacement, source, flags=re.MULTILINE)


def load_module(module_path: str, gpu_id: int, output_json: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    source = Path(module_path).read_text(encoding="utf-8")
    source = source.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "0"', f'os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"')
    source = source.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "1"', f'os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"')
    progress_json = str(Path(output_json).with_name(f"{Path(output_json).stem}_progress.json"))
    source = _replace_assignment(source, "OUTPUT_JSON", output_json.replace("\\", "/"))
    source = _replace_assignment(source, "PROGRESS_JSON", progress_json.replace("\\", "/"))
    module = types.ModuleType("dynamic52_base_module")
    module.__file__ = module_path
    exec(compile(source, module_path, "exec"), module.__dict__)
    return module


def init_results():
    return {
        "Baseline": {"ranks": [], "times": [], "prompts": []},
        "iKnow": {"ranks": [], "times": [], "prompts": []},
        "Full2Hop_Static085_NoLLM_OriginalAgg": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "avg_lambda": [],
            "gt_lambda": [],
            "positive_gain_rate": [],
        },
        "Full2Hop_DynamicDecay_NoLLM_OriginalAgg": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "avg_lambda": [],
            "gt_lambda": [],
            "positive_gain_rate": [],
        },
    }


def dynamic_lambda(margin: float, hop2_gain: float, cfg: dict) -> float:
    if margin >= cfg["tau_high"]:
        return cfg["lambda_low"]
    if hop2_gain > cfg["gamma"]:
        return cfg["lambda_high"]
    return cfg["lambda_mid"]


def run_full2hop_static_no_llm(module, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails_fn):
    score = cos_sim_orig.clone()
    prompt_count = 0
    lambdas = []
    gt_lambda = []
    gain_pos_flags = []
    for ci in top_indices:
        class_name = label_classes[ci]
        kg_ent = module.get_kg_entity(kg_classes[ci])
        h1_map = module.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, {}, False)
        h2_prompts = module.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, {}, False)
        s1 = module.score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
        s2 = module.score_prompt_list(clap_model, audio_embed, h2_prompts)
        lambda_i = float(module.DECAY_GAMMA)
        lambdas.append(lambda_i)
        gt_lambda.append((int(ci), lambda_i))
        h1_pool = module.soft_pool(s1) if s1.numel() > 0 else None
        h2_pool = module.soft_pool(s2) if s2.numel() > 0 else None
        gain_pos_flags.append(bool(h1_pool is not None and h2_pool is not None and (h2_pool - h1_pool).item() > 0))
        all_scores = torch.cat([s1, s2 * lambda_i]) if s2.numel() > 0 else s1
        best_scores = module.safe_topk(all_scores, module.TOP_P)
        score[ci] = module.aggregate_candidate(cos_sim_orig[ci], best_scores, 0.5, "original")
        prompt_count += len(h1_map) + len(h2_prompts)
    return score, prompt_count, lambdas, gt_lambda, gain_pos_flags


def run_full2hop_dynamic_no_llm(module, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails_fn, cfg):
    score = cos_sim_orig.clone()
    prompt_count = 0
    lambdas = []
    gt_lambda = []
    gain_pos_flags = []
    sorted_vals = torch.sort(cos_sim_orig, descending=True).values
    margin = float((sorted_vals[0] - sorted_vals[1]).item()) if sorted_vals.numel() >= 2 else 0.0
    for ci in top_indices:
        class_name = label_classes[ci]
        kg_ent = module.get_kg_entity(kg_classes[ci])
        h1_map = module.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, {}, False)
        h2_prompts = module.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, {}, False)
        s1 = module.score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
        s2 = module.score_prompt_list(clap_model, audio_embed, h2_prompts)
        h1_pool = module.soft_pool(s1) if s1.numel() > 0 else None
        h2_pool = module.soft_pool(s2) if s2.numel() > 0 else None
        hop2_gain = float((h2_pool - h1_pool).item()) if h1_pool is not None and h2_pool is not None else -999.0
        lambda_i = dynamic_lambda(margin, hop2_gain, cfg) if s2.numel() > 0 else 0.0
        lambdas.append(lambda_i)
        gt_lambda.append((int(ci), lambda_i))
        gain_pos_flags.append(bool(hop2_gain > 0))
        all_scores = torch.cat([s1, s2 * lambda_i]) if s2.numel() > 0 else s1
        best_scores = module.safe_topk(all_scores, module.TOP_P)
        score[ci] = module.aggregate_candidate(cos_sim_orig[ci], best_scores, 0.5, "original")
        prompt_count += len(h1_map) + len(h2_prompts)
    return score, prompt_count, lambdas, gt_lambda, gain_pos_flags


def build_payload(module, results, module_path: str, config_name: str):
    payload = {
        "module_path": module_path,
        "config": config_name,
        "config_values": CONFIGS[config_name],
        "results": {},
    }
    for name, values in results.items():
        metrics = module.compute_metrics(values["ranks"])
        entry = {
            "Hit@1": float(metrics[0]),
            "Hit@3": float(metrics[1]),
            "Hit@5": float(metrics[2]),
            "MRR": float(metrics[3]),
            "avg_prompts": float(np.mean(values["prompts"])) if values["prompts"] else 0.0,
            "avg_time_ms": float(np.mean(values["times"])) if values["times"] else 0.0,
        }
        if "avg_lambda" in values:
            entry["avg_lambda"] = float(np.mean(values["avg_lambda"])) if values["avg_lambda"] else 0.0
            entry["positive_gain_rate"] = float(np.mean(values["positive_gain_rate"])) if values["positive_gain_rate"] else 0.0
        payload["results"][name] = entry
    return payload


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    module = load_module(args.module_path, args.gpu_id, args.output_json)
    cfg = CONFIGS[args.config]
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting dynamic decay ablation from: {args.module_path}")
    print(f"Using config: {args.config} -> {cfg}")
    print("Locked setting: No LLM + Full 2-Hop + OriginalAgg")

    clap_model = module.CLAP(version="2023", use_cuda=torch.cuda.is_available())
    with torch.no_grad():
        kge_model = torch.load(os.path.join(module.KGE_MODEL_DIR, "trained_model.pkl"), map_location=module.DEVICE)
    kge_model.eval()
    training_factory = module.TriplesFactory.from_path(module.TRAIN_TRIPLES_PATH)

    hop1_relations = [rel for rel in module.HOP1_RELATIONS if rel in training_factory.relation_to_id]
    hop2_relations = [rel for rel in module.HOP2_RELATIONS if rel in training_factory.relation_to_id]
    get_tails = module.build_tail_predictor(kge_model, training_factory)

    dataset = module.load_dataset()
    label_classes = dataset["label_classes"]
    kg_classes = dataset["kg_classes"]
    class_labels_set = dataset["class_labels_set"]
    samples = list(module.iter_samples(dataset))
    text_embeds = F.normalize(module.get_safe_text_embeddings(clap_model, label_classes, module.DEVICE), dim=-1)
    results = init_results()

    desc = f"{Path(args.module_path).parent.name} dynamic decay"
    for sample in tqdm(samples, total=len(samples), desc=desc):
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]
        if not os.path.exists(audio_path) or not true_indices:
            continue

        t_audio = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        t_audio_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if t_audio is not None:
            t_audio.record()
        audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
        audio_embed = F.normalize(module.to_tensor(audio_emb_raw).to(module.DEVICE).float(), dim=-1)
        if t_audio_end is not None:
            t_audio_end.record()
            torch.cuda.synchronize()
            audio_ms = float(t_audio.elapsed_time(t_audio_end))
        else:
            audio_ms = 0.0

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[:module.TOP_K]

        t = module.time.time()
        score, prompt_count = module.run_baseline(cos_sim_orig)
        results["Baseline"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["Baseline"]["times"].append(audio_ms + (module.time.time() - t) * 1000.0)
        results["Baseline"]["prompts"].append(prompt_count)

        t = module.time.time()
        score, prompt_count = module.run_iknow(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, get_tails
        )
        results["iKnow"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["iKnow"]["times"].append(audio_ms + (module.time.time() - t) * 1000.0)
        results["iKnow"]["prompts"].append(prompt_count)

        t = module.time.time()
        score, prompt_count, lambdas, _, gain_flags = run_full2hop_static_no_llm(
            module, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails
        )
        results["Full2Hop_Static085_NoLLM_OriginalAgg"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["Full2Hop_Static085_NoLLM_OriginalAgg"]["times"].append(audio_ms + (module.time.time() - t) * 1000.0)
        results["Full2Hop_Static085_NoLLM_OriginalAgg"]["prompts"].append(prompt_count)
        results["Full2Hop_Static085_NoLLM_OriginalAgg"]["avg_lambda"].append(float(np.mean(lambdas)) if lambdas else 0.0)
        results["Full2Hop_Static085_NoLLM_OriginalAgg"]["positive_gain_rate"].append(float(np.mean(gain_flags)) if gain_flags else 0.0)

        t = module.time.time()
        score, prompt_count, lambdas, _, gain_flags = run_full2hop_dynamic_no_llm(
            module, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, cfg
        )
        results["Full2Hop_DynamicDecay_NoLLM_OriginalAgg"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["Full2Hop_DynamicDecay_NoLLM_OriginalAgg"]["times"].append(audio_ms + (module.time.time() - t) * 1000.0)
        results["Full2Hop_DynamicDecay_NoLLM_OriginalAgg"]["prompts"].append(prompt_count)
        results["Full2Hop_DynamicDecay_NoLLM_OriginalAgg"]["avg_lambda"].append(float(np.mean(lambdas)) if lambdas else 0.0)
        results["Full2Hop_DynamicDecay_NoLLM_OriginalAgg"]["positive_gain_rate"].append(float(np.mean(gain_flags)) if gain_flags else 0.0)

    payload = build_payload(module, results, args.module_path, args.config)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = payload["results"]
    print("\n" + "=" * 140)
    print(f"{'Metric':<8} | {'Baseline':<12} | {'iKnow':<12} | {'Static0.85':<18} | {'DynamicDecay':<18}")
    print("-" * 140)
    for metric_name in ["Hit@1", "Hit@3", "Hit@5", "MRR"]:
        print(
            f"{metric_name:<8} | {metrics['Baseline'][metric_name]:<12.2f} | {metrics['iKnow'][metric_name]:<12.2f} | "
            f"{metrics['Full2Hop_Static085_NoLLM_OriginalAgg'][metric_name]:<18.2f} | {metrics['Full2Hop_DynamicDecay_NoLLM_OriginalAgg'][metric_name]:<18.2f}"
        )

    print("\n" + "=" * 140)
    print(f"{'Method':<38} | {'Avg lambda':<12} | {'Positive hop2 gain rate':<24} | {'Avg prompts':<12} | {'Avg time (ms)':<14}")
    print("-" * 140)
    for name in ["Full2Hop_Static085_NoLLM_OriginalAgg", "Full2Hop_DynamicDecay_NoLLM_OriginalAgg"]:
        print(
            f"{name:<38} | {metrics[name].get('avg_lambda', 0.0):<12.3f} | {metrics[name].get('positive_gain_rate', 0.0):<24.3f} | "
            f"{metrics[name]['avg_prompts']:<12.1f} | {metrics[name]['avg_time_ms']:<14.1f}"
        )


if __name__ == "__main__":
    main()
