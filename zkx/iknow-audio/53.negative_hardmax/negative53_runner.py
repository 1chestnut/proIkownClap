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


NEG_LAMBDA = 0.35


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
    module = types.ModuleType("negative53_base_module")
    module.__file__ = module_path
    exec(compile(source, module_path, "exec"), module.__dict__)
    return module


def candidate_originalagg_details(module, clap_model, audio_embed, cos_sim_orig, ci, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map):
    class_name = label_classes[ci]
    kg_ent = module.get_kg_entity(kg_classes[ci])
    tau = cos_sim_orig[ci].item() + module.RELATIVE_MARGIN
    h1_map = module.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map, True)
    s1 = module.score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
    max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
    if s1.numel() > 0 and max_h1 >= tau:
        logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * module.LOGIT_SCALE, s1 * module.LOGIT_SCALE])
        score = (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / module.LOGIT_SCALE
        return {
            "score": score,
            "prompt_count": len(h1_map),
            "hop2": False,
            "prompts": list(h1_map.values()),
        }

    h2_prompts = module.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map, True)
    s2 = module.score_prompt_list(clap_model, audio_embed, h2_prompts)
    all_scores = torch.cat([s1, s2 * module.DECAY_GAMMA]) if s2.numel() > 0 else s1
    if all_scores.numel() == 0:
        return {
            "score": cos_sim_orig[ci],
            "prompt_count": len(h1_map) + len(h2_prompts),
            "hop2": True,
            "prompts": list(h1_map.values()) + list(h2_prompts),
        }
    logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * module.LOGIT_SCALE, all_scores * module.LOGIT_SCALE])
    score = (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / module.LOGIT_SCALE
    return {
        "score": score,
        "prompt_count": len(h1_map) + len(h2_prompts),
        "hop2": True,
        "prompts": list(h1_map.values()) + list(h2_prompts),
    }


def init_results():
    return {
        "Selective2Hop_OriginalAgg": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
        },
        "NegativePrompt_HardMax": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "avg_negative_prompts": [],
        },
    }


def build_payload(module, results, module_path: str):
    payload = {
        "module_path": module_path,
        "neg_lambda": NEG_LAMBDA,
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
            "hop2_activation_sample": float(np.mean(values["hop2_activation_sample"])) if values["hop2_activation_sample"] else 0.0,
            "hop2_activation_candidate": float(np.mean(values["hop2_activation_candidate"])) if values["hop2_activation_candidate"] else 0.0,
        }
        if "avg_negative_prompts" in values:
            entry["avg_negative_prompts"] = float(np.mean(values["avg_negative_prompts"])) if values["avg_negative_prompts"] else 0.0
        payload["results"][name] = entry
    return payload


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    module = load_module(args.module_path, args.gpu_id, args.output_json)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting negative hardmax test from: {args.module_path}")
    print(f"Negative lambda: {NEG_LAMBDA}")
    print("Output columns: Selective2Hop_OriginalAgg vs NegativePrompt_HardMax")

    prompt_map = module.load_llm_prompts(module.LLM_PROMPTS_PATH)
    clap_model = module.CLAP(version="2023", use_cuda=torch.cuda.is_available())
    with module.warnings.catch_warnings():
        module.warnings.simplefilter("ignore")
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

    desc = f"{Path(args.module_path).parent.name} neg hardmax"
    for sample in tqdm(samples, total=len(samples), desc=desc):
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]
        if not os.path.exists(audio_path) or not true_indices:
            continue

        t_audio_start = module.time.time()
        audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
        audio_embed = F.normalize(module.to_tensor(audio_emb_raw).to(module.DEVICE).float(), dim=-1)
        audio_ms = (module.time.time() - t_audio_start) * 1000.0

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[:module.TOP_K]
        alpha_dynamic = module.instance_alpha(torch.max(cos_sim_orig).item())

        t = module.time.time()
        orig_score_vec = cos_sim_orig.clone()
        prompt_count = 0
        hop2_flags = []
        details = {}
        for ci in top_indices:
            det = candidate_originalagg_details(
                module, clap_model, audio_embed, cos_sim_orig, ci, label_classes, kg_classes,
                class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map
            )
            details[int(ci)] = det
            orig_score_vec[ci] = det["score"]
            prompt_count += det["prompt_count"]
            hop2_flags.append(det["hop2"])
        results["Selective2Hop_OriginalAgg"]["ranks"].append(module.rank_of_true(orig_score_vec, true_indices))
        results["Selective2Hop_OriginalAgg"]["times"].append(audio_ms + (module.time.time() - t) * 1000.0)
        results["Selective2Hop_OriginalAgg"]["prompts"].append(prompt_count)
        results["Selective2Hop_OriginalAgg"]["hop2_activation_sample"].append(any(hop2_flags))
        results["Selective2Hop_OriginalAgg"]["hop2_activation_candidate"].append(float(np.mean(hop2_flags)) if hop2_flags else 0.0)

        t = module.time.time()
        neg_score_vec = orig_score_vec.clone()
        neg_prompt_total = 0
        baseline_top2 = list(top_indices[:2])
        for ci in baseline_top2:
            competitors = [cj for cj in baseline_top2 if cj != ci]
            if not competitors:
                continue
            competitor = int(competitors[0])
            own_prompts = set(details[int(ci)]["prompts"])
            competitor_prompts = [p for p in details[competitor]["prompts"] if p not in own_prompts]
            if not competitor_prompts:
                continue
            neg_scores = module.score_prompt_list(clap_model, audio_embed, competitor_prompts)
            if neg_scores.numel() == 0:
                continue
            neg_score = torch.max(neg_scores)
            neg_score_vec[ci] = details[int(ci)]["score"] - (NEG_LAMBDA * neg_score)
            neg_prompt_total += len(competitor_prompts)

        results["NegativePrompt_HardMax"]["ranks"].append(module.rank_of_true(neg_score_vec, true_indices))
        results["NegativePrompt_HardMax"]["times"].append(audio_ms + (module.time.time() - t) * 1000.0)
        results["NegativePrompt_HardMax"]["prompts"].append(prompt_count)
        results["NegativePrompt_HardMax"]["hop2_activation_sample"].append(any(hop2_flags))
        results["NegativePrompt_HardMax"]["hop2_activation_candidate"].append(float(np.mean(hop2_flags)) if hop2_flags else 0.0)
        results["NegativePrompt_HardMax"]["avg_negative_prompts"].append(float(neg_prompt_total))

    payload = build_payload(module, results, args.module_path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = payload["results"]
    print("\n" + "=" * 120)
    print(f"{'Metric':<8} | {'Selective2Hop_OriginalAgg':<28} | {'NegativePrompt_HardMax':<28}")
    print("-" * 120)
    for metric_name in ["Hit@1", "Hit@3", "Hit@5", "MRR"]:
        print(
            f"{metric_name:<8} | {metrics['Selective2Hop_OriginalAgg'][metric_name]:<28.2f} | "
            f"{metrics['NegativePrompt_HardMax'][metric_name]:<28.2f}"
        )


if __name__ == "__main__":
    main()
