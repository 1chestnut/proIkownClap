# -*- coding: utf-8 -*-
import json
import os
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

LOCAL_MODEL_DIR = "/home/star/zkx/iknow-audio/data/model"
CLAP_WEIGHTS_PATH = os.path.join(LOCAL_MODEL_DIR, "CLAP_weights_2023.pth")
GPT2_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "gpt2")
ROBERTA_LOCAL_PATH = "/home/star/zkx/CLAP/model/roberta-base"

import msclap.CLAPWrapper


def offline_hf_hub_download(*args, **kwargs):
    if not os.path.exists(CLAP_WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing CLAP weights: {CLAP_WEIGHTS_PATH}")
    return CLAP_WEIGHTS_PATH


msclap.CLAPWrapper.hf_hub_download = offline_hf_hub_download

import transformers


def patch_transformers_offline(cls_name):
    cls = getattr(transformers, cls_name)
    orig_func = cls.from_pretrained

    @classmethod
    def my_func(cls_inner, pretrained_model_name_or_path, *args, **kwargs):
        target_path = GPT2_LOCAL_PATH if "gpt2" in str(pretrained_model_name_or_path).lower() else ROBERTA_LOCAL_PATH
        kwargs["local_files_only"] = True
        return orig_func.__func__(cls_inner, target_path, *args, **kwargs)

    setattr(cls, "from_pretrained", my_func)


for cls_name in ["AutoModel", "AutoConfig", "AutoTokenizer", "GPT2Tokenizer", "RobertaTokenizer"]:
    try:
        patch_transformers_offline(cls_name)
    except AttributeError:
        continue

from msclap import CLAP
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

ESC50_CSV = "/home/star/zkx/CLAP/data/ESC-50/esc50.csv"
ESC50_AUDIO_DIR = "/home/star/zkx/CLAP/data/ESC-50/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")
LLM_PROMPTS_PATH = "/data/zkx/zkx/iknow-audio/18adj/esc50_llm_filter/llm_prompts_filtered_from_acoustic.json"
OUTPUT_JSON = "/data/zkx/zkx/iknow-audio/28.next/esc_repulsive/results_repulsive.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 5
TOP_M = 3
TOP_P = 5
DECAY_GAMMA = 0.85
LOGIT_SCALE = 100.0
RELATIVE_MARGIN = -0.02
ALPHA_MIN = 0.35
ALPHA_MAX = 0.75

HOP1_RELATIONS = ["belongs to class", "has parent", "perceived as", "event composed of", "has children"]
HOP2_RELATIONS = ["belongs to class", "has parent", "event composed of", "has children"]


def to_tensor(emb):
    if isinstance(emb, torch.Tensor):
        return emb
    return torch.from_numpy(emb)


def compute_metrics(ranks):
    ranks = np.array(ranks)
    if len(ranks) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        np.mean(ranks <= 1) * 100.0,
        np.mean(ranks <= 3) * 100.0,
        np.mean(ranks <= 5) * 100.0,
        np.mean(1.0 / ranks) * 100.0,
    )


def load_llm_prompts(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompt_map = {}
    for key, text in data.items():
        parts = key.split("||")
        if len(parts) == 3:
            sub, rel, obj = [p.strip().lower() for p in parts]
            prompt_map[f"{sub}||{rel}||{obj}"] = text
    return prompt_map


def get_safe_text_embeddings(model, text_list, device):
    if not text_list:
        return torch.empty((0,), device=device)
    embs = []
    for text in text_list:
        emb = model.get_text_embeddings([text])
        embs.append(to_tensor(emb).to(device).float())
    return torch.cat(embs, dim=0)


def score_prompt_list(clap_model, audio_embed, prompts):
    if not prompts:
        return torch.empty((0,), device=audio_embed.device)
    prompt_embs = F.normalize(get_safe_text_embeddings(clap_model, prompts, audio_embed.device), dim=-1)
    scores = torch.matmul(audio_embed, prompt_embs.T).squeeze()
    if scores.dim() == 0:
        scores = scores.unsqueeze(0)
    return scores


def safe_topk(scores, k):
    if scores.numel() == 0:
        return scores
    values, _ = torch.topk(scores, min(k, scores.numel()))
    return values


def soft_pool(scores):
    if scores.numel() == 0:
        return None
    return (torch.logsumexp(scores * LOGIT_SCALE, dim=0) - np.log(scores.numel())) / LOGIT_SCALE


def static_alpha_from_maxsim(max_sim):
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim
    return float(max(ALPHA_MIN, min(ALPHA_MAX, alpha)))


def candidate_gain_alpha(gain_value, q25, q75):
    if q75 <= q25:
        return (ALPHA_MIN + ALPHA_MAX) / 2.0
    u = (gain_value - q25) / (q75 - q25)
    u = max(0.0, min(1.0, float(u)))
    return ALPHA_MAX - (ALPHA_MAX - ALPHA_MIN) * u


def rank_of_true(score_vec, true_indices):
    order = torch.argsort(score_vec, descending=True).detach().cpu().numpy()
    return min(int(np.where(order == target_idx)[0][0] + 1) for target_idx in true_indices)


def init_results():
    return {
        "Baseline": {"ranks": [], "times": [], "prompts": []},
        "iKnow": {"ranks": [], "times": [], "prompts": []},
        "Full2Hop": {"ranks": [], "times": [], "prompts": []},
        "Selective2Hop_StaticAlpha": {
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


def get_kg_entity(class_name):
    clean_name = class_name.replace("_", " ").strip()
    if "(" in clean_name:
        clean_name = clean_name.split("(")[0].strip()
    return clean_name.replace(" - ", " ")


def resolve_head_query(head, training_factory):
    return head


def build_tail_predictor(kge_model, training_factory):
    cache = {}

    def get_tails(head, rel):
        key = (head, rel)
        if key in cache:
            return cache[key]
        query_head = resolve_head_query(head, training_factory)
        if query_head not in training_factory.entity_to_id:
            cache[key] = []
            return []
        try:
            pred = predict_target(model=kge_model, head=query_head, relation=rel, triples_factory=training_factory)
            result = pred.df.sort_values(by="score", ascending=False).head(TOP_M)["tail_label"].tolist()
        except Exception:
            result = []
        cache[key] = result
        return result

    return get_tails


def build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, prompt_map):
    h1_map = OrderedDict()
    for rel in hop1_relations:
        for tail in get_tails_fn(kg_ent, rel):
            tail_norm = tail.lower().strip()
            if tail_norm == class_name.lower() or tail_norm in class_labels_set:
                continue
            if tail_norm not in h1_map:
                key = f"{kg_ent.lower()}||{rel.lower()}||{tail_norm}"
                h1_map[tail_norm] = prompt_map.get(key, f"{class_name}, {tail}")
    return h1_map


def build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map):
    h2_prompts = []
    for head_norm in h1_map.keys():
        for rel in hop2_relations:
            for tail in get_tails_fn(head_norm, rel):
                tail_norm = tail.lower().strip()
                if tail_norm == class_name.lower() or tail_norm in class_labels_set or tail_norm in h1_map:
                    continue
                key = f"{head_norm}||{rel.lower()}||{tail_norm}"
                h2_prompts.append(prompt_map.get(key, f"{class_name}, {tail}"))
    return h2_prompts


def run_baseline(cos_sim_orig):
    return cos_sim_orig.clone(), 1


def run_iknow(clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, get_tails_fn):
    score = cos_sim_orig.clone()
    prompt_count = 0
    for ci in top_indices:
        class_name = label_classes[ci]
        kg_ent = get_kg_entity(kg_classes[ci])
        tails = OrderedDict()
        for rel in hop1_relations:
            for tail in get_tails_fn(kg_ent, rel):
                tail_norm = tail.lower().strip()
                if tail_norm == class_name.lower() or tail_norm in class_labels_set:
                    continue
                if tail_norm not in tails:
                    tails[tail_norm] = f"{class_name}, {tail}"
        if not tails:
            continue
        scores = score_prompt_list(clap_model, audio_embed, list(tails.values()))
        logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * LOGIT_SCALE, scores * LOGIT_SCALE])
        score[ci] = (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / LOGIT_SCALE
        prompt_count += len(tails)
    return score, prompt_count


def run_full2hop(clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails_fn, prompt_map, alpha_dynamic):
    score = cos_sim_orig.clone()
    prompt_count = 0
    for ci in top_indices:
        class_name = label_classes[ci]
        kg_ent = get_kg_entity(kg_classes[ci])
        h1_map = build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, prompt_map)
        h2_prompts = build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map)
        s1 = score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
        s2 = score_prompt_list(clap_model, audio_embed, h2_prompts)
        all_scores = torch.cat([s1, s2 * DECAY_GAMMA]) if s2.numel() > 0 else s1
        best_scores = safe_topk(all_scores, TOP_P)
        if best_scores.numel() > 0:
            soft_s = soft_pool(best_scores)
            score[ci] = (alpha_dynamic * cos_sim_orig[ci]) + ((1.0 - alpha_dynamic) * soft_s)
        prompt_count += len(h1_map) + len(h2_prompts)
    return score, prompt_count


def run_selective2hop(clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails_fn, prompt_map, alpha_dynamic):
    score = cos_sim_orig.clone()
    prompt_count = 0
    hop2_flags = []
    for ci in top_indices:
        class_name = label_classes[ci]
        kg_ent = get_kg_entity(kg_classes[ci])
        tau = cos_sim_orig[ci].item() + RELATIVE_MARGIN
        h1_map = build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, prompt_map)
        s1 = score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
        max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
        if s1.numel() > 0 and max_h1 >= tau:
            hop2_flags.append(False)
            best_scores = safe_topk(s1, TOP_P)
            if best_scores.numel() > 0:
                soft_s = soft_pool(best_scores)
                score[ci] = (alpha_dynamic * cos_sim_orig[ci]) + ((1.0 - alpha_dynamic) * soft_s)
            prompt_count += len(h1_map)
            continue
        hop2_flags.append(True)
        h2_prompts = build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map)
        s2 = score_prompt_list(clap_model, audio_embed, h2_prompts)
        all_scores = torch.cat([s1, s2 * DECAY_GAMMA]) if s2.numel() > 0 else s1
        best_scores = safe_topk(all_scores, TOP_P)
        if best_scores.numel() > 0:
            soft_s = soft_pool(best_scores)
            score[ci] = (alpha_dynamic * cos_sim_orig[ci]) + ((1.0 - alpha_dynamic) * soft_s)
        prompt_count += len(h1_map) + len(h2_prompts)
    extras = {
        "hop2_activated": any(hop2_flags),
        "candidate_level_activation_rate": float(np.mean(hop2_flags)) if hop2_flags else 0.0,
    }
    return score, prompt_count, extras


def build_selective_kg_score(clap_model, audio_embed, cos_sim_orig, ci, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails_fn, prompt_map):
    class_name = label_classes[ci]
    kg_ent = get_kg_entity(kg_classes[ci])
    tau = cos_sim_orig[ci].item() + RELATIVE_MARGIN
    h1_map = build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, prompt_map)
    s1 = score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
    max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0

    if s1.numel() > 0 and max_h1 >= tau:
        best_scores = safe_topk(s1, TOP_P)
        soft_s = soft_pool(best_scores) if best_scores.numel() > 0 else None
        return soft_s, len(h1_map), False

    h2_prompts = build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map)
    s2 = score_prompt_list(clap_model, audio_embed, h2_prompts)
    all_scores = torch.cat([s1, s2 * DECAY_GAMMA]) if s2.numel() > 0 else s1
    best_scores = safe_topk(all_scores, TOP_P)
    soft_s = soft_pool(best_scores) if best_scores.numel() > 0 else None
    return soft_s, len(h1_map) + len(h2_prompts), True


def build_selective_branch_scores(clap_model, audio_embed, cos_sim_orig, ci, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails_fn, prompt_map):
    class_name = label_classes[ci]
    kg_ent = get_kg_entity(kg_classes[ci])
    tau = cos_sim_orig[ci].item() + RELATIVE_MARGIN
    h1_map = build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, prompt_map)
    s1 = score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
    max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
    hop1_score = soft_pool(safe_topk(s1, TOP_P)) if s1.numel() > 0 else None

    if s1.numel() > 0 and max_h1 >= tau:
        return hop1_score, None, len(h1_map), False

    h2_prompts = build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map)
    s2 = score_prompt_list(clap_model, audio_embed, h2_prompts)
    hop2_candidates = s2 * DECAY_GAMMA if s2.numel() > 0 else s2
    hop2_score = soft_pool(safe_topk(hop2_candidates, TOP_P)) if s2.numel() > 0 else None
    return hop1_score, hop2_score, len(h1_map) + len(h2_prompts), True


REPULSIVE_BETA = 0.35
REPULSIVE_MARGIN = 0.01


def candidate_originalagg_score(clap_model, audio_embed, cos_sim_orig, ci, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails_fn, prompt_map):
    class_name = label_classes[ci]
    kg_ent = get_kg_entity(kg_classes[ci])
    tau = cos_sim_orig[ci].item() + RELATIVE_MARGIN
    h1_map = build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, prompt_map)
    s1 = score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
    max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
    if s1.numel() > 0 and max_h1 >= tau:
        logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * LOGIT_SCALE, s1 * LOGIT_SCALE])
        score = (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / LOGIT_SCALE
        return score, len(h1_map), False
    h2_prompts = build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map)
    s2 = score_prompt_list(clap_model, audio_embed, h2_prompts)
    all_scores = torch.cat([s1, s2 * DECAY_GAMMA]) if s2.numel() > 0 else s1
    if all_scores.numel() == 0:
        return cos_sim_orig[ci], len(h1_map) + len(h2_prompts), True
    logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * LOGIT_SCALE, all_scores * LOGIT_SCALE])
    score = (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / LOGIT_SCALE
    return score, len(h1_map) + len(h2_prompts), True


def run_selective2hop_repulsive(clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails_fn, prompt_map):
    score = cos_sim_orig.clone()
    prompt_count = 0
    hop2_flags = []
    candidate_scores = {}
    for ci in top_indices:
        own_score, used_prompts, hop2_flag = candidate_originalagg_score(
            clap_model, audio_embed, cos_sim_orig, ci, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails_fn, prompt_map
        )
        candidate_scores[ci] = own_score
        hop2_flags.append(hop2_flag)
        prompt_count += used_prompts
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



def load_dataset():
    df = pd.read_csv(ESC50_CSV)
    kg_classes = sorted(df["category"].unique())
    label_classes = [cat.replace("_", " ") for cat in kg_classes]
    class_to_idx = {cat: idx for idx, cat in enumerate(kg_classes)}
    return {
        "df": df,
        "label_classes": label_classes,
        "kg_classes": kg_classes,
        "class_labels_set": {name.lower() for name in label_classes},
        "class_to_idx": class_to_idx,
        "total": len(df),
    }


def iter_samples(dataset):
    class_to_idx = dataset["class_to_idx"]
    for _, row in dataset["df"].iterrows():
        true_indices = [class_to_idx[row["category"]]] if row["category"] in class_to_idx else []
        yield {"audio_path": os.path.join(ESC50_AUDIO_DIR, row["filename"]), "true_indices": true_indices}


def print_final_tables(results):
    metrics = {name: compute_metrics(results[name]["ranks"]) for name in results}
    order = ["Baseline", "iKnow", "Full2Hop", "Selective2Hop_StaticAlpha", "Selective2Hop_Repulsive"]
    print("\n" + "=" * 160)
    print(f"{'Metric':<8} | {'Baseline':<10} | {'iKnow':<10} | {'Full2Hop':<12} | {'Sel2 StaticAlpha':<18} | {'Sel2 Repulsive':<18}")
    print("-" * 160)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        row = [f"{metric_name:<8}"]
        for name in order:
            width = 10 if name in ["Baseline", "iKnow"] else 18
            if name == "Full2Hop":
                width = 12
            row.append(f"{metrics[name][idx]:<{width}.2f}")
        print(" | ".join(row))

    print("\n" + "=" * 170)
    print(
        f"{'Method':<24} | {'Hop2 activation(sample)':<24} | {'Hop2 activation(candidate)':<27} | {'Avg prompts':<12} | {'Avg time (ms)':<14} | {'Hit@1':<8} | {'MRR':<8}"
    )
    print("-" * 170)
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


def save_results(results):
    metrics = {name: compute_metrics(results[name]["ranks"]) for name in results}
    payload = {"metrics": {}, "stats": {}}
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
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def main():
    print("Starting ESC50 repulsive fusion pilot")
    print(f"Device: {DEVICE}")
    print("Prompt policy: baseline label-only, iKnow uses pure single-hop concat without Top-P, selective hop1/hop2 use filtered acoustic LLM prompts with concat fallback; final fusion adds competitor-aware repulsive penalty over original aggregated scores")
    print(f"Alpha bounds: min={ALPHA_MIN}, max={ALPHA_MAX}")
    prompt_map = load_llm_prompts(LLM_PROMPTS_PATH)
    clap_model = CLAP(version="2023", use_cuda=torch.cuda.is_available())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = torch.load(os.path.join(KGE_MODEL_DIR, "trained_model.pkl"), map_location=DEVICE)
    kge_model.eval()
    training_factory = TriplesFactory.from_path(TRAIN_TRIPLES_PATH)

    hop1_relations = [rel for rel in HOP1_RELATIONS if rel in training_factory.relation_to_id]
    hop2_relations = [rel for rel in HOP2_RELATIONS if rel in training_factory.relation_to_id]
    print(f"Valid hop1 relations: {hop1_relations}")
    print(f"Valid hop2 relations: {hop2_relations}")

    get_tails = build_tail_predictor(kge_model, training_factory)
    dataset = load_dataset()
    label_classes = dataset["label_classes"]
    kg_classes = dataset["kg_classes"]
    class_labels_set = dataset["class_labels_set"]
    text_embeds = F.normalize(get_safe_text_embeddings(clap_model, label_classes, DEVICE), dim=-1)
    results = init_results()
    skipped_count = 0

    for sample in tqdm(iter_samples(dataset), total=dataset["total"], desc="ESC50 repulsive"):
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]
        if not os.path.exists(audio_path) or not true_indices:
            skipped_count += 1
            continue
        try:
            t_audio_start = time.time()
            audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = F.normalize(to_tensor(audio_emb_raw).to(DEVICE).float(), dim=-1)
            audio_ms = (time.time() - t_audio_start) * 1000.0
        except Exception:
            skipped_count += 1
            continue

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[:TOP_K]
        alpha_static = static_alpha_from_maxsim(torch.max(cos_sim_orig).item())
        alpha_static = static_alpha_from_maxsim(torch.max(cos_sim_orig).item())

        t = time.time()
        score, prompt_count = run_baseline(cos_sim_orig)
        results["Baseline"]["ranks"].append(rank_of_true(score, true_indices))
        results["Baseline"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Baseline"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count = run_iknow(clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, get_tails)
        results["iKnow"]["ranks"].append(rank_of_true(score, true_indices))
        results["iKnow"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["iKnow"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count = run_full2hop(clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, alpha_static)
        results["Full2Hop"]["ranks"].append(rank_of_true(score, true_indices))
        results["Full2Hop"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Full2Hop"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count, extras = run_selective2hop(clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, alpha_static)
        results["Selective2Hop_StaticAlpha"]["ranks"].append(rank_of_true(score, true_indices))
        results["Selective2Hop_StaticAlpha"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Selective2Hop_StaticAlpha"]["prompts"].append(prompt_count)
        results["Selective2Hop_StaticAlpha"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Selective2Hop_StaticAlpha"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Selective2Hop_StaticAlpha"]["alphas"].append(alpha_static)

        t = time.time()
        score, prompt_count, extras = run_selective2hop_repulsive(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map
        )
        results["Selective2Hop_Repulsive"]["ranks"].append(rank_of_true(score, true_indices))
        results["Selective2Hop_Repulsive"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Selective2Hop_Repulsive"]["prompts"].append(prompt_count)
        results["Selective2Hop_Repulsive"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Selective2Hop_Repulsive"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Selective2Hop_Repulsive"]["alphas"].append(float(extras["alpha_mean"]))

    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid samples.")
    print_final_tables(results)
    save_results(results)


if __name__ == "__main__":
    main()
