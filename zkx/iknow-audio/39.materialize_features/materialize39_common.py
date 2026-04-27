# -*- coding: utf-8 -*-
import importlib.util
import inspect
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_base(module_path):
    spec = importlib.util.spec_from_file_location("materialize39_base_module", module_path)
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


def compute_alpha_dynamic(base, cos_sim_orig):
    max_sim = float(torch.max(cos_sim_orig).item())
    if hasattr(base, "static_alpha_from_maxsim"):
        return float(base.static_alpha_from_maxsim(max_sim))
    if hasattr(base, "instance_alpha"):
        return float(base.instance_alpha(max_sim))
    alpha_min = float(getattr(base, "ALPHA_MIN", 0.35))
    alpha_max = float(getattr(base, "ALPHA_MAX", 0.75))
    alpha = alpha_min + (alpha_max - alpha_min) * max_sim
    return float(max(alpha_min, min(alpha_max, alpha)))


def call_build_h1_map(base, kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map):
    sig = inspect.signature(base.build_h1_map)
    if "use_llm" in sig.parameters:
        return base.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map, True)
    return base.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map)


def call_build_h2_prompts(base, h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map):
    sig = inspect.signature(base.build_h2_prompts)
    if "use_llm" in sig.parameters:
        return base.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map, True)
    return base.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map)


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
    h1_map = call_build_h1_map(base, kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map)
    s1 = base.score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
    max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
    if s1.numel() > 0 and max_h1 >= tau:
        logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * base.LOGIT_SCALE, s1 * base.LOGIT_SCALE])
        score = (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / base.LOGIT_SCALE
        return score, len(h1_map), False
    h2_prompts = call_build_h2_prompts(base, h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map)
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


def run_ours_head(
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
    alpha_dynamic,
):
    if hasattr(base, "run_selective2hop"):
        ret = base.run_selective2hop(
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
        if len(ret) == 4:
            score, prompts, extra, _ = ret
        else:
            score, prompts, extra = ret
        return score, prompts, extra
    if hasattr(base, "run_selective_variant"):
        return base.run_selective_variant(
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
            True,
            "dynamic",
        )
    raise AttributeError("Unsupported base module: missing Ours implementation")


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


def choose_candidate_oracle(base, orig_score, ours_score, top_indices, true_indices):
    true_idx = int(true_indices[0])
    best_key = None
    best = None
    for mask_bits in np.ndindex(*(2 for _ in range(len(top_indices)))):
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
            best = {"score": final_score, "choose_orig_ratio": float(choose_orig / max(1, len(top_indices))), "mask_bits": list(mask_bits)}
    return best


def compute_key_candidates(top_indices, baseline_scores, orig_scores, ours_scores, true_indices):
    top_indices = list(top_indices)
    baseline_top = top_indices[: min(3, len(top_indices))]
    orig_top = sorted(top_indices, key=lambda i: float(orig_scores[i].item()), reverse=True)[: min(3, len(top_indices))]
    ours_top = sorted(top_indices, key=lambda i: float(ours_scores[i].item()), reverse=True)[: min(3, len(top_indices))]
    disagreement = sorted(top_indices, key=lambda i: abs(float(orig_scores[i].item()) - float(ours_scores[i].item())), reverse=True)[: min(2, len(top_indices))]
    chosen = set(baseline_top + orig_top + ours_top + disagreement + [int(x) for x in true_indices if int(x) in top_indices])
    return sorted(chosen)


@torch.no_grad()
def run_materialize(module_path, output_dir, launch_title, dataset_name):
    os.makedirs(output_dir, exist_ok=True)
    base = load_base(module_path)
    print(launch_title)
    print(f"Device: {base.DEVICE}")
    print("Materializing offline candidate feature table with Oracle labels")

    prompt_map = base.load_llm_prompts(base.LLM_PROMPTS_PATH)
    clap_model = base.CLAP(version="2023", use_cuda=torch.cuda.is_available())
    with base.warnings.catch_warnings():
        base.warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(base.KGE_MODEL_DIR, "trained_model.pkl"), map_location=base.DEVICE)
    kge_model.eval()
    training_factory = base.TriplesFactory.from_path(base.TRAIN_TRIPLES_PATH)
    hop1_relations = [rel for rel in base.HOP1_RELATIONS if rel in training_factory.relation_to_id]
    hop2_relations = [rel for rel in base.HOP2_RELATIONS if rel in training_factory.relation_to_id]
    get_tails = base.build_tail_predictor(kge_model, training_factory)
    dataset = base.load_dataset()
    label_classes = dataset["label_classes"]
    kg_classes = dataset["kg_classes"]
    class_labels_set = dataset["class_labels_set"]
    text_embeds = F.normalize(base.get_safe_text_embeddings(clap_model, label_classes, base.DEVICE), dim=-1)

    rows = []
    skipped_count = 0
    for sample in tqdm(base.iter_samples(dataset), total=dataset["total"], desc=f"Materialize-{dataset_name}"):
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]
        if not os.path.exists(audio_path) or not true_indices:
            skipped_count += 1
            continue
        try:
            audio_embed = F.normalize(get_audio_embedding(base, clap_model, audio_path, base.DEVICE), dim=-1)
        except Exception:
            skipped_count += 1
            continue

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[: base.TOP_K]
        alpha_dynamic = compute_alpha_dynamic(base, cos_sim_orig)
        orig_score, orig_prompts, orig_extra = run_originalagg(
            base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map
        )
        ours_score, ours_prompts, ours_extra = run_ours_head(
            base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic
        )
        oracle = choose_candidate_oracle(base, orig_score, ours_score, top_indices, true_indices)
        oracle_mask = {int(ci): int(bit) for ci, bit in zip(top_indices, oracle["mask_bits"])}
        key_candidates = set(compute_key_candidates(top_indices, cos_sim_orig, orig_score, ours_score, true_indices))

        baseline_top1 = float(cos_sim_orig[top_indices[0]].item())
        baseline_margin = float(cos_sim_orig[top_indices[0]].item() - cos_sim_orig[top_indices[1]].item()) if len(top_indices) > 1 else 0.0
        entropy = normalized_entropy(cos_sim_orig, top_indices)
        prompt_count = int(orig_prompts)
        true_idx_set = set(int(x) for x in true_indices)
        for ci in top_indices:
            ci = int(ci)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "sample_id": audio_path,
                    "candidate_index": ci,
                    "candidate_class": label_classes[ci],
                    "true_label": label_classes[int(true_indices[0])],
                    "is_ground_truth": 1 if ci in true_idx_set else 0,
                    "is_key_candidate": 1 if ci in key_candidates else 0,
                    "baseline_top1": baseline_top1,
                    "baseline_margin": baseline_margin,
                    "entropy": entropy,
                    "hop2_activation": float(orig_extra["hop2_activated"]),
                    "prompt_count": prompt_count,
                    "prompt_count_log1p": float(np.log1p(prompt_count)),
                    "base_score": float(cos_sim_orig[ci].item()),
                    "orig_score": float(orig_score[ci].item()),
                    "ours_score": float(ours_score[ci].item()),
                    "orig_minus_base": float(orig_score[ci].item() - cos_sim_orig[ci].item()),
                    "ours_minus_base": float(ours_score[ci].item() - cos_sim_orig[ci].item()),
                    "orig_minus_ours": float(orig_score[ci].item() - ours_score[ci].item()),
                    "oracle_label": oracle_mask[ci],
                }
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"{dataset_name}_features.csv")
    meta_path = os.path.join(output_dir, "results_materialize.json")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = {
        "dataset_name": dataset_name,
        "row_count": int(len(df)),
        "sample_count": int(df["sample_id"].nunique()) if not df.empty else 0,
        "key_candidate_ratio": float(df["is_key_candidate"].mean()) if not df.empty else 0.0,
        "oracle_choose_orig_ratio": float(df["oracle_label"].mean()) if not df.empty else 0.0,
        "skipped_count": int(skipped_count),
        "csv_path": csv_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nMaterialization summary")
    for k, v in summary.items():
        print(f"{k}: {v}")
