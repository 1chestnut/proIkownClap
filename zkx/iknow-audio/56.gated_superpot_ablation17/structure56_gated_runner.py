import argparse
import json
import os
from pathlib import Path
import re
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


METHOD_NAME = "Selective2Hop_GatedSuperPot"
KAPPA_MIN = 5.0
KAPPA_MAX = 100.0
MARGIN_TAU = 0.10
PEAKINESS_TAU = 0.08


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
    module = types.ModuleType("structure56_base_module")
    module.__file__ = module_path
    exec(compile(source, module_path, "exec"), module.__dict__)
    return module, progress_json


def dynamic_kappa(max_sim: float) -> float:
    max_sim = float(max(0.0, min(1.0, max_sim)))
    return KAPPA_MIN + (KAPPA_MAX - KAPPA_MIN) * max_sim


def global_superpot(base_score, best_scores, max_sim):
    if best_scores.numel() == 0:
        return base_score
    kappa = dynamic_kappa(max_sim)
    logits = torch.cat([base_score.unsqueeze(0) * kappa, best_scores * kappa])
    return (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / kappa


def init_results():
    return {
        "Baseline": {"ranks": [], "times": [], "prompts": []},
        "iKnow": {"ranks": [], "times": [], "prompts": []},
        "Selective2Hop_OriginalAgg": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
        },
        "Ours": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
        },
        METHOD_NAME: {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "superpot_triggered": [],
            "peakiness_values": [],
            "margin_values": [],
        },
    }


def init_progress(total_samples):
    return {"next_index": 0, "total_samples": total_samples, "results": init_results()}


def load_progress(progress_json, total_samples):
    if not os.path.exists(progress_json):
        return init_progress(total_samples)
    try:
        with open(progress_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return init_progress(total_samples)
    fresh = init_progress(total_samples)
    fresh["next_index"] = int(payload.get("next_index", 0))
    old = payload.get("results", {})
    for m, vals in fresh["results"].items():
        if m in old and isinstance(old[m], dict):
            for k in vals:
                fresh["results"][m][k] = old[m].get(k, vals[k])
    return fresh


def save_progress(progress_json, payload):
    tmp = progress_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, progress_json)


def save_results(output_json, module, results, total_samples):
    payload = {
        "completed": True,
        "total_samples": total_samples,
        "config": {
            "margin_tau": MARGIN_TAU,
            "peakiness_tau": PEAKINESS_TAU,
            "kappa_min": KAPPA_MIN,
            "kappa_max": KAPPA_MAX,
        },
        "metrics": {},
        "stats": {},
    }
    for name, values in results.items():
        hit1, hit3, hit5, mrr = module.compute_metrics(values["ranks"])
        payload["metrics"][name] = {
            "Hit@1": float(hit1),
            "Hit@3": float(hit3),
            "Hit@5": float(hit5),
            "MRR": float(mrr),
        }
        payload["stats"][name] = {
            "avg_prompts": float(np.mean(values["prompts"])) if values["prompts"] else 0.0,
            "avg_time_ms": float(np.mean(values["times"])) if values["times"] else 0.0,
        }
        if "hop2_activation_sample" in values:
            payload["stats"][name]["hop2_activation_sample"] = float(np.mean(values["hop2_activation_sample"])) if values["hop2_activation_sample"] else 0.0
            payload["stats"][name]["hop2_activation_candidate"] = float(np.mean(values["hop2_activation_candidate"])) if values["hop2_activation_candidate"] else 0.0
        if "superpot_triggered" in values:
            payload["stats"][name]["superpot_trigger_ratio"] = float(np.mean(values["superpot_triggered"])) if values["superpot_triggered"] else 0.0
            payload["stats"][name]["peakiness_q25"] = float(np.quantile(values["peakiness_values"], 0.25)) if values["peakiness_values"] else 0.0
            payload["stats"][name]["peakiness_q50"] = float(np.quantile(values["peakiness_values"], 0.50)) if values["peakiness_values"] else 0.0
            payload["stats"][name]["peakiness_q75"] = float(np.quantile(values["peakiness_values"], 0.75)) if values["peakiness_values"] else 0.0
            payload["stats"][name]["margin_q25"] = float(np.quantile(values["margin_values"], 0.25)) if values["margin_values"] else 0.0
            payload["stats"][name]["margin_q50"] = float(np.quantile(values["margin_values"], 0.50)) if values["margin_values"] else 0.0
            payload["stats"][name]["margin_q75"] = float(np.quantile(values["margin_values"], 0.75)) if values["margin_values"] else 0.0
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    metrics = payload["metrics"]
    print("\n" + "=" * 160)
    print(
        f"{'Metric':<12} | {'Baseline':<12} | {'iKnow':<12} | "
        f"{'Selective2Hop_OriginalAgg':<28} | {'Ours':<12} | {METHOD_NAME:<28}"
    )
    print("-" * 160)
    for metric_name in ["Hit@1", "Hit@3", "Hit@5", "MRR"]:
        print(
            f"{metric_name:<12} | {metrics['Baseline'][metric_name]:<12.2f} | {metrics['iKnow'][metric_name]:<12.2f} | "
            f"{metrics['Selective2Hop_OriginalAgg'][metric_name]:<28.2f} | {metrics['Ours'][metric_name]:<12.2f} | "
            f"{metrics[METHOD_NAME][metric_name]:<28.2f}"
        )
    print(
        f"\n[Gate] margin_tau={MARGIN_TAU:.2f}, peakiness_tau={PEAKINESS_TAU:.2f}, "
        f"trigger_ratio={payload['stats'][METHOD_NAME].get('superpot_trigger_ratio', 0.0):.4f}"
    )


def collect_candidate_scores(
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
    use_llm,
):
    max_sim = float(torch.max(cos_sim_orig).item())
    candidate_records = []
    hop2_flags = []
    total_prompts = 0
    for ci in top_indices:
        class_name = label_classes[ci]
        kg_ent = module.get_kg_entity(kg_classes[ci])
        tau = cos_sim_orig[ci].item() + module.RELATIVE_MARGIN
        h1_map = module.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, prompt_map, use_llm)
        s1 = module.score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
        max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
        if s1.numel() > 0 and max_h1 >= tau:
            hop2_flags.append(False)
            all_scores = s1
            total_prompts += len(h1_map)
        else:
            hop2_flags.append(True)
            h2_prompts = module.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map, use_llm)
            s2 = module.score_prompt_list(clap_model, audio_embed, h2_prompts)
            all_scores = torch.cat([s1, s2 * module.DECAY_GAMMA]) if s2.numel() > 0 else s1
            total_prompts += len(h1_map) + len(h2_prompts)
        best_scores = module.safe_topk(all_scores, module.TOP_P)
        knowledge_score = module.soft_pool(best_scores) if best_scores.numel() > 0 else cos_sim_orig[ci]
        superpot_score = global_superpot(cos_sim_orig[ci], best_scores, max_sim)
        peakiness = float(best_scores.max().item() - best_scores.mean().item()) if best_scores.numel() > 0 else 0.0
        candidate_records.append(
            {
                "index": int(ci),
                "ours_score": knowledge_score,
                "superpot_score": superpot_score,
                "peakiness": peakiness,
            }
        )
    extras = {
        "hop2_activated": any(hop2_flags),
        "candidate_level_activation_rate": float(np.mean(hop2_flags)) if hop2_flags else 0.0,
        "prompt_count": total_prompts,
        "sample_peakiness": float(np.mean([rec["peakiness"] for rec in candidate_records])) if candidate_records else 0.0,
        "max_sim": max_sim,
    }
    return candidate_records, extras


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    module, progress_json = load_module(args.module_path, args.gpu_id, args.output_json)
    print(f"Starting gated superpot from: {args.module_path}")
    print(
        f"Gate logic: if margin > {MARGIN_TAU:.2f} -> Ours; "
        f"else if peakiness < {PEAKINESS_TAU:.2f} -> SuperPot; else -> Ours"
    )

    prompt_map = module.load_llm_prompts(module.LLM_PROMPTS_PATH)
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
    total_samples = len(samples)
    text_embeds = F.normalize(module.get_safe_text_embeddings(clap_model, label_classes, module.DEVICE), dim=-1)

    progress = load_progress(progress_json, total_samples)
    results = progress["results"]
    start_idx = progress["next_index"]
    print(f"Resume from sample index: {start_idx}")

    for sample_idx in tqdm(
        range(start_idx, total_samples),
        total=total_samples,
        initial=start_idx,
        desc=f"{Path(args.module_path).parent.name} gated",
    ):
        sample = samples[sample_idx]
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]
        if not os.path.exists(audio_path) or not true_indices:
            progress["next_index"] = sample_idx + 1
            save_progress(progress_json, progress)
            continue

        try:
            t_audio = time.time()
            audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = F.normalize(module.to_tensor(audio_emb_raw).to(module.DEVICE).float(), dim=-1)
            audio_ms = (time.time() - t_audio) * 1000.0
        except Exception:
            progress["next_index"] = sample_idx + 1
            save_progress(progress_json, progress)
            continue

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[:module.TOP_K]
        margin = float(cos_sim_orig[sorted_indices[0]].item() - cos_sim_orig[sorted_indices[1]].item()) if len(sorted_indices) > 1 else 1.0

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

        alpha_dynamic = module.instance_alpha(torch.max(cos_sim_orig).item())
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

        t = time.time()
        score, prompt_count, extras = module.run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, True, "ours"
        )
        results["Ours"]["ranks"].append(module.rank_of_true(score, true_indices))
        results["Ours"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Ours"]["prompts"].append(prompt_count)
        results["Ours"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Ours"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))

        t = time.time()
        candidate_records, gate_extras = collect_candidate_scores(
            module, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, True
        )
        score = cos_sim_orig.clone()
        triggered = False
        sample_peakiness = gate_extras["sample_peakiness"]
        if margin <= MARGIN_TAU and sample_peakiness < PEAKINESS_TAU:
            triggered = True
        for rec in candidate_records:
            score[rec["index"]] = rec["superpot_score"] if triggered else rec["ours_score"]
        results[METHOD_NAME]["ranks"].append(module.rank_of_true(score, true_indices))
        results[METHOD_NAME]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results[METHOD_NAME]["prompts"].append(gate_extras["prompt_count"])
        results[METHOD_NAME]["hop2_activation_sample"].append(bool(gate_extras["hop2_activated"]))
        results[METHOD_NAME]["hop2_activation_candidate"].append(float(gate_extras["candidate_level_activation_rate"]))
        results[METHOD_NAME]["superpot_triggered"].append(bool(triggered))
        results[METHOD_NAME]["peakiness_values"].append(float(sample_peakiness))
        results[METHOD_NAME]["margin_values"].append(float(margin))

        progress["next_index"] = sample_idx + 1
        progress["results"] = results
        save_progress(progress_json, progress)

    save_results(args.output_json, module, results, total_samples)


if __name__ == "__main__":
    main()
