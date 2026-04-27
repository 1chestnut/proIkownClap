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

DCASE_CSV = "/home/star/zkx/iknow-audio/data/DCASE17-T4/my_evaluation_dataset.csv"
DCASE_AUDIO_DIR = "/home/star/zkx/iknow-audio/data/DCASE17-T4/audio"
KGE_MODEL_DIR = "/home/star/zkx/iknow-audio/KGE_models/001/TFVpYwo2_RotatE_False"
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, "AKG_train_triples.tsv")
LLM_PROMPTS_PATH = "/home/star/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/dcase17/llm_prompts.json"
OUTPUT_JSON = "/data/zkx/zkx/iknow-audio/17.消融/dcase/results_ablation.json"
PROGRESS_JSON = "/data/zkx/zkx/iknow-audio/17.消融/dcase/progress.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 5
TOP_M = 3
TOP_P = 5
DECAY_GAMMA = 0.85
LOGIT_SCALE = 100.0
RELATIVE_MARGIN = -0.02
ALPHA_MIN = 0.4
ALPHA_MAX = 0.8

HOP1_RELATIONS = ["indicates", "described by", "used for", "associated with environment", "has parent", "is instance of"]
HOP2_RELATIONS = ["indicates", "described by", "used for", "associated with environment", "has parent", "is instance of"]


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


def original_lse(base_score, scores):
    logits = torch.cat([base_score.unsqueeze(0) * LOGIT_SCALE, scores * LOGIT_SCALE])
    return (torch.logsumexp(logits, dim=0) - np.log(logits.numel())) / LOGIT_SCALE


def instance_alpha(max_sim):
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * max_sim
    return float(max(ALPHA_MIN, min(ALPHA_MAX, alpha)))


def rank_of_true(score_vec, true_indices):
    order = torch.argsort(score_vec, descending=True).detach().cpu().numpy()
    return min(int(np.where(order == target_idx)[0][0] + 1) for target_idx in true_indices)


def init_results():
    return {
        "Baseline": {"ranks": [], "times": [], "prompts": []},
        "iKnow": {"ranks": [], "times": [], "prompts": []},
        "Full2Hop": {"ranks": [], "times": [], "prompts": []},
        "Ours": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "alphas": [],
        },
        "Selective2Hop_OriginalAgg": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "alphas": [],
        },
        "Selective2Hop_NoLLM": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "alphas": [],
        },
    }


def get_kg_entity(class_name):
    mapping = {
        "ambulance (siren)": "ambulance",
        "fire engine, fire truck (siren)": "fire engine",
        "police car (siren)": "police car",
        "air horn, truck horn": "horn",
        "civil defense siren": "siren",
        "reversing beeps": "beep",
        "car passing by": "car",
    }
    return mapping.get(class_name, class_name)


def resolve_head_query(head, training_factory):
    return head if head in training_factory.entity_to_id else ""


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


def resolve_prompt(prompt_map, head_norm, rel, tail_norm, class_name, tail_text, use_llm):
    if use_llm:
        key = f"{head_norm}||{rel.lower()}||{tail_norm}"
        return prompt_map.get(key, f"{class_name}, {tail_text}")
    return f"{class_name}, {tail_text}"


def build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, prompt_map, use_llm):
    h1_map = OrderedDict()
    head_norm = kg_ent.lower()
    for rel in hop1_relations:
        for tail in get_tails_fn(kg_ent, rel):
            tail_norm = tail.lower().strip()
            if tail_norm == class_name.lower() or tail_norm in class_labels_set:
                continue
            if tail_norm not in h1_map:
                h1_map[tail_norm] = resolve_prompt(prompt_map, head_norm, rel, tail_norm, class_name, tail, use_llm)
    return h1_map


def build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map, use_llm):
    h2_prompts = []
    for head_norm in h1_map.keys():
        for rel in hop2_relations:
            for tail in get_tails_fn(head_norm, rel):
                tail_norm = tail.lower().strip()
                if tail_norm == class_name.lower() or tail_norm in class_labels_set or tail_norm in h1_map:
                    continue
                h2_prompts.append(resolve_prompt(prompt_map, head_norm, rel, tail_norm, class_name, tail, use_llm))
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
        score[ci] = original_lse(cos_sim_orig[ci], scores)
        prompt_count += len(tails)
    return score, prompt_count


def aggregate_candidate(base_score, best_scores, alpha_dynamic, aggregation_mode):
    if best_scores.numel() == 0:
        return base_score
    if aggregation_mode == "original":
        return original_lse(base_score, best_scores)
    knowledge_score = soft_pool(best_scores)
    return (alpha_dynamic * base_score) + ((1.0 - alpha_dynamic) * knowledge_score)


def run_full2hop_variant(
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
    aggregation_mode,
):
    score = cos_sim_orig.clone()
    prompt_count = 0
    for ci in top_indices:
        class_name = label_classes[ci]
        kg_ent = get_kg_entity(kg_classes[ci])
        h1_map = build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, prompt_map, use_llm)
        h2_prompts = build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map, use_llm)
        s1 = score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
        s2 = score_prompt_list(clap_model, audio_embed, h2_prompts)
        all_scores = torch.cat([s1, s2 * DECAY_GAMMA]) if s2.numel() > 0 else s1
        best_scores = safe_topk(all_scores, TOP_P)
        score[ci] = aggregate_candidate(cos_sim_orig[ci], best_scores, alpha_dynamic, aggregation_mode)
        prompt_count += len(h1_map) + len(h2_prompts)
    return score, prompt_count


def run_selective_variant(
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
    aggregation_mode,
):
    score = cos_sim_orig.clone()
    prompt_count = 0
    hop2_flags = []
    for ci in top_indices:
        class_name = label_classes[ci]
        kg_ent = get_kg_entity(kg_classes[ci])
        tau = cos_sim_orig[ci].item() + RELATIVE_MARGIN
        h1_map = build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails_fn, prompt_map, use_llm)
        s1 = score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
        max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
        if s1.numel() > 0 and max_h1 >= tau:
            hop2_flags.append(False)
            best_scores = safe_topk(s1, TOP_P)
            score[ci] = aggregate_candidate(cos_sim_orig[ci], best_scores, alpha_dynamic, aggregation_mode)
            prompt_count += len(h1_map)
            continue
        hop2_flags.append(True)
        h2_prompts = build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails_fn, prompt_map, use_llm)
        s2 = score_prompt_list(clap_model, audio_embed, h2_prompts)
        all_scores = torch.cat([s1, s2 * DECAY_GAMMA]) if s2.numel() > 0 else s1
        best_scores = safe_topk(all_scores, TOP_P)
        score[ci] = aggregate_candidate(cos_sim_orig[ci], best_scores, alpha_dynamic, aggregation_mode)
        prompt_count += len(h1_map) + len(h2_prompts)
    extras = {
        "hop2_activated": any(hop2_flags),
        "candidate_level_activation_rate": float(np.mean(hop2_flags)) if hop2_flags else 0.0,
    }
    return score, prompt_count, extras


def init_progress_payload(total_samples):
    return {"next_index": 0, "total_samples": total_samples, "results": init_results()}


def normalize_progress(progress, total_samples):
    fresh = init_progress_payload(total_samples)
    if not progress:
        return fresh
    fresh["next_index"] = int(progress.get("next_index", 0))
    old_results = progress.get("results", {})
    for method, default_val in fresh["results"].items():
        if method in old_results and isinstance(old_results[method], dict):
            for key, default_list in default_val.items():
                fresh["results"][method][key] = old_results[method].get(key, default_list)
    return fresh


def load_progress(total_samples):
    if not os.path.exists(PROGRESS_JSON):
        return init_progress_payload(total_samples)
    try:
        with open(PROGRESS_JSON, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return init_progress_payload(total_samples)
    payload = normalize_progress(payload, total_samples)
    if payload["next_index"] > total_samples:
        payload["next_index"] = 0
    return payload


def save_progress(progress):
    tmp_path = PROGRESS_JSON + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)
    os.replace(tmp_path, PROGRESS_JSON)


def compute_stats(results):
    stats = {}
    for method, values in results.items():
        stats[method] = {
            "avg_prompts": float(np.mean(values["prompts"])) if values["prompts"] else 0.0,
            "avg_time_ms": float(np.mean(values["times"])) if values["times"] else 0.0,
        }
        if "hop2_activation_sample" in values:
            stats[method]["hop2_activation_sample"] = float(np.mean(values["hop2_activation_sample"])) if values["hop2_activation_sample"] else 0.0
            stats[method]["hop2_activation_candidate"] = float(np.mean(values["hop2_activation_candidate"])) if values["hop2_activation_candidate"] else 0.0
    return stats


def save_results(results, total_samples, completed):
    metrics = {name: compute_metrics(values["ranks"]) for name, values in results.items()}
    stats = compute_stats(results)
    payload = {
        "completed": completed,
        "total_samples": total_samples,
        "metrics": {},
        "stats": stats,
    }
    for method in results:
        payload["metrics"][method] = {
            "Hit@1": metrics[method][0],
            "Hit@3": metrics[method][1],
            "Hit@5": metrics[method][2],
            "MRR": metrics[method][3],
        }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_main_tables(results):
    metrics = {name: compute_metrics(results[name]["ranks"]) for name in results}
    stats = compute_stats(results)
    print("\n" + "=" * 132)
    print(f"{'Metric':<8} | {'Baseline':<12} | {'iKnow':<12} | {'Full2Hop':<14} | {'Ours':<12}")
    print("-" * 132)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(
            f"{metric_name:<8} | {metrics['Baseline'][idx]:<12.2f} | {metrics['iKnow'][idx]:<12.2f} | "
            f"{metrics['Full2Hop'][idx]:<14.2f} | {metrics['Ours'][idx]:<12.2f}"
        )

    print("\n" + "=" * 180)
    print(
        f"{'Method':<28} | {'Hop2 activation(sample)':<24} | {'Hop2 activation(candidate)':<27} | "
        f"{'Avg prompts':<12} | {'Avg time (ms)':<14} | {'Hit@1':<8} | {'MRR':<8}"
    )
    print("-" * 180)
    for name in ["Baseline", "iKnow", "Full2Hop", "Ours"]:
        sample_rate = "N/A"
        candidate_rate = "N/A"
        if name == "Ours":
            sample_rate = f"{stats[name]['hop2_activation_sample'] * 100:.1f}%"
            candidate_rate = f"{stats[name]['hop2_activation_candidate'] * 100:.1f}%"
        print(
            f"{name:<28} | {sample_rate:<24} | {candidate_rate:<27} | {stats[name]['avg_prompts']:<12.1f} | "
            f"{stats[name]['avg_time_ms']:<14.1f} | {metrics[name][0]:<8.2f} | {metrics[name][3]:<8.2f}"
        )

    print("\n" + "=" * 132)
    print(f"{'Fusion Ablation':<18} | {'Selective2Hop_OriginalAgg':<28} | {'Ours':<12}")
    print("-" * 132)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(f"{metric_name:<18} | {metrics['Selective2Hop_OriginalAgg'][idx]:<28.2f} | {metrics['Ours'][idx]:<12.2f}")

    print("\n" + "=" * 132)
    print(f"{'LLM Ablation':<18} | {'Selective2Hop_NoLLM':<24} | {'Ours':<12}")
    print("-" * 132)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(f"{metric_name:<18} | {metrics['Selective2Hop_NoLLM'][idx]:<24.2f} | {metrics['Ours'][idx]:<12.2f}")



def load_dataset():
    label_classes = [
        "ambulance (siren)", "bicycle", "bus", "car", "car alarm", "car passing by",
        "civil defense siren", "fire engine, fire truck (siren)", "motorcycle",
        "police car (siren)", "reversing beeps", "screaming", "skateboard",
        "train", "train horn", "truck", "air horn, truck horn",
    ]
    class_to_idx = {cat.lower(): idx for idx, cat in enumerate(label_classes)}
    df = pd.read_csv(DCASE_CSV)
    return {
        "df": df,
        "label_classes": label_classes,
        "kg_classes": label_classes,
        "class_labels_set": {name.lower() for name in label_classes},
        "class_to_idx": class_to_idx,
        "total": len(df),
    }


def iter_samples(dataset):
    class_to_idx = dataset["class_to_idx"]
    for _, row in dataset["df"].iterrows():
        normalized = str(row["paper_formatted_labels"]).lower().strip().replace("fire engine, fire truck (siren)", "C_FIRE").replace("air horn, truck horn", "C_AIR")
        true_indices = []
        for part in normalized.split(","):
            key = part.strip().replace("C_FIRE", "fire engine, fire truck (siren)").replace("C_AIR", "air horn, truck horn")
            if key in class_to_idx:
                true_indices.append(class_to_idx[key])
        yield {"audio_path": os.path.join(DCASE_AUDIO_DIR, row["audio_filename"]), "true_indices": true_indices}


@torch.no_grad()
def main():
    print("Starting DCASE ablation: Baseline / iKnow / Full2Hop / Ours / Selective2Hop_OriginalAgg / Selective2Hop_NoLLM")
    print(f"Device: {DEVICE}")
    print("Reference baseline: 16.综合/test2 as the fixed best code line.")
    print("Ablations: Full2Hop vs Ours; Selective2Hop_OriginalAgg vs Ours; Selective2Hop_NoLLM vs Ours")

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
    samples = list(iter_samples(dataset))
    total_samples = len(samples)
    print(f"Total samples prepared: {total_samples}")

    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                done_payload = json.load(f)
            if done_payload.get("completed") and int(done_payload.get("total_samples", -1)) == total_samples:
                print(f"Found completed results: {OUTPUT_JSON}")
                return
        except Exception:
            pass

    progress = load_progress(total_samples)
    start_idx = progress["next_index"]
    results = progress["results"]
    print(f"Resume from sample index: {start_idx}")
    text_embeds = F.normalize(get_safe_text_embeddings(clap_model, label_classes, DEVICE), dim=-1)
    skipped_count = 0

    for sample_idx in tqdm(range(start_idx, total_samples), total=total_samples, initial=start_idx, desc="DCASE ablation"):
        sample = samples[sample_idx]
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]

        if not os.path.exists(audio_path) or not true_indices:
            skipped_count += 1
            progress["next_index"] = sample_idx + 1
            save_progress(progress)
            continue

        try:
            t_audio_start = time.time()
            audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = F.normalize(to_tensor(audio_emb_raw).to(DEVICE).float(), dim=-1)
            audio_ms = (time.time() - t_audio_start) * 1000.0
        except Exception:
            skipped_count += 1
            progress["next_index"] = sample_idx + 1
            save_progress(progress)
            continue

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[:TOP_K]
        alpha_dynamic = instance_alpha(torch.max(cos_sim_orig).item())

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
        score, prompt_count = run_full2hop_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, True, "dynamic"
        )
        results["Full2Hop"]["ranks"].append(rank_of_true(score, true_indices))
        results["Full2Hop"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Full2Hop"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count, extras = run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, True, "dynamic"
        )
        results["Ours"]["ranks"].append(rank_of_true(score, true_indices))
        results["Ours"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Ours"]["prompts"].append(prompt_count)
        results["Ours"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Ours"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Ours"]["alphas"].append(alpha_dynamic)

        t = time.time()
        score, prompt_count, extras = run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, True, "original"
        )
        results["Selective2Hop_OriginalAgg"]["ranks"].append(rank_of_true(score, true_indices))
        results["Selective2Hop_OriginalAgg"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Selective2Hop_OriginalAgg"]["prompts"].append(prompt_count)
        results["Selective2Hop_OriginalAgg"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Selective2Hop_OriginalAgg"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Selective2Hop_OriginalAgg"]["alphas"].append(alpha_dynamic)

        t = time.time()
        score, prompt_count, extras = run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, False, "dynamic"
        )
        results["Selective2Hop_NoLLM"]["ranks"].append(rank_of_true(score, true_indices))
        results["Selective2Hop_NoLLM"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Selective2Hop_NoLLM"]["prompts"].append(prompt_count)
        results["Selective2Hop_NoLLM"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Selective2Hop_NoLLM"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Selective2Hop_NoLLM"]["alphas"].append(alpha_dynamic)

        progress["next_index"] = sample_idx + 1
        progress["results"] = results
        save_progress(progress)

    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid samples.")
    save_results(results, total_samples, completed=True)
    print_main_tables(results)


if __name__ == "__main__":
    main()
