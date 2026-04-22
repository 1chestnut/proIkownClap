# -*- coding: utf-8 -*-
import importlib.util
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


SEED = 42
N_FOLDS = 5
HIDDEN_DIM = 8
DROPOUT = 0.2
EPOCHS = 80
LEARNING_RATE = 0.01


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base(module_path):
    spec = importlib.util.spec_from_file_location("router_base_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyHardRouter(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
    h2_prompts = base.build_h2_prompts(
        h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map
    )
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
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "alphas": [],
        },
        "Selective2Hop_Ours": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "alphas": [],
        },
        "Selective2Hop_HardRouter": {
            "ranks": [],
            "times": [],
            "prompts": [],
            "hop2_activation_sample": [],
            "hop2_activation_candidate": [],
            "alphas": [],
            "route_choose_orig": [],
        },
    }


def print_tables(base, results, title):
    metrics = {name: base.compute_metrics(results[name]["ranks"]) for name in results}
    order = [
        "Baseline",
        "iKnow",
        "Full2Hop",
        "Selective2Hop_OriginalAgg",
        "Selective2Hop_Ours",
        "Selective2Hop_HardRouter",
    ]
    print("\n" + "=" * 180)
    print(title)
    print("-" * 180)
    print(
        f"{'Metric':<8} | {'Baseline':<10} | {'iKnow':<10} | {'Full2Hop':<12} | "
        f"{'Sel2 OriginalAgg':<18} | {'Sel2 Ours':<18} | {'Sel2 HardRouter':<18}"
    )
    print("-" * 180)
    for idx, metric_name in enumerate(["Hit@1", "Hit@3", "Hit@5", "MRR"]):
        print(
            f"{metric_name:<8} | {metrics['Baseline'][idx]:<10.2f} | {metrics['iKnow'][idx]:<10.2f} | "
            f"{metrics['Full2Hop'][idx]:<12.2f} | {metrics['Selective2Hop_OriginalAgg'][idx]:<18.2f} | "
            f"{metrics['Selective2Hop_Ours'][idx]:<18.2f} | {metrics['Selective2Hop_HardRouter'][idx]:<18.2f}"
        )

    print("\n" + "=" * 190)
    print(
        f"{'Method':<24} | {'Hop2 activation(sample)':<24} | {'Hop2 activation(candidate)':<27} | "
        f"{'Avg prompts':<12} | {'Avg time (ms)':<14} | {'Hit@1':<8} | {'MRR':<8} | {'Route->Orig':<12}"
    )
    print("-" * 190)
    for name in order:
        sample_rate = "N/A"
        candidate_rate = "N/A"
        route_rate = "N/A"
        if name.startswith("Selective2Hop"):
            sample_rate = f"{np.mean(results[name]['hop2_activation_sample']) * 100:.1f}%"
            candidate_rate = f"{np.mean(results[name]['hop2_activation_candidate']) * 100:.1f}%"
        if name == "Selective2Hop_HardRouter":
            route_rate = f"{np.mean(results[name]['route_choose_orig']) * 100:.1f}%"
        print(
            f"{name:<24} | {sample_rate:<24} | {candidate_rate:<27} | "
            f"{np.mean(results[name]['prompts']):<12.1f} | {np.mean(results[name]['times']):<14.1f} | "
            f"{metrics[name][0]:<8.2f} | {metrics[name][3]:<8.2f} | {route_rate:<12}"
        )


def save_results(base, results, output_json, meta=None):
    metrics = {name: base.compute_metrics(results[name]["ranks"]) for name in results}
    payload = {"metrics": {}, "stats": {}, "meta": meta or {}}
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
            payload["stats"][name]["hop2_activation_sample"] = float(
                np.mean(results[name]["hop2_activation_sample"])
            ) if results[name]["hop2_activation_sample"] else 0.0
            payload["stats"][name]["hop2_activation_candidate"] = float(
                np.mean(results[name]["hop2_activation_candidate"])
            ) if results[name]["hop2_activation_candidate"] else 0.0
            payload["stats"][name]["alpha_mean"] = float(
                np.mean(results[name]["alphas"])
            ) if results[name]["alphas"] else 0.0
        if name == "Selective2Hop_HardRouter":
            payload["stats"][name]["route_choose_orig"] = float(
                np.mean(results[name]["route_choose_orig"])
            ) if results[name]["route_choose_orig"] else 0.0
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def choose_label(orig_rank, ours_rank, orig_true_score, ours_true_score):
    if orig_rank < ours_rank:
        return 1.0
    if ours_rank < orig_rank:
        return 0.0
    return 1.0 if orig_true_score >= ours_true_score else 0.0


def train_and_predict(features, labels):
    set_seed(SEED)
    n = len(features)
    indices = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)
    folds = np.array_split(indices, N_FOLDS)
    probs = np.zeros(n, dtype=np.float32)

    for fold_idx, val_idx in enumerate(folds, start=1):
        train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fold_idx - 1])
        x_train = features[train_idx]
        y_train = labels[train_idx]
        x_val = features[val_idx]

        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std

        if len(np.unique(y_train)) < 2:
            probs[val_idx] = float(np.mean(y_train))
            print(f"[RouterCV] Fold {fold_idx}: single-class train fold, use constant p={float(np.mean(y_train)):.4f}")
            continue

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TinyHardRouter().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.BCEWithLogitsLoss()

        x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)

        model.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            logits = model(x_train_t)
            loss = loss_fn(logits, y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(x_val_t)).detach().cpu().numpy()
        probs[val_idx] = val_probs.astype(np.float32)
        print(
            f"[RouterCV] Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, "
            f"mean_p={float(np.mean(val_probs)):.4f}"
        )
    return probs


@torch.no_grad()
def collect_samples(base, clap_model, text_embeds, dataset, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map):
    samples = []
    results = init_results()
    skipped = 0

    for sample in tqdm(base.iter_samples(dataset), total=dataset["total"], desc="RouterHardCollect"):
        audio_path = sample["audio_path"]
        true_indices = sample["true_indices"]
        if not os.path.exists(audio_path) or not true_indices:
            skipped += 1
            continue
        try:
            t_audio_start = time.time()
            audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = F.normalize(base.to_tensor(audio_emb_raw).to(base.DEVICE).float(), dim=-1)
            audio_ms = (time.time() - t_audio_start) * 1000.0
        except Exception:
            skipped += 1
            continue

        cos_sim_orig = torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[:base.TOP_K]
        alpha_dynamic = base.static_alpha_from_maxsim(torch.max(cos_sim_orig).item())

        t = time.time()
        baseline_score, baseline_prompts = base.run_baseline(cos_sim_orig)
        baseline_ms = audio_ms + (time.time() - t) * 1000.0
        results["Baseline"]["ranks"].append(base.rank_of_true(baseline_score, true_indices))
        results["Baseline"]["times"].append(baseline_ms)
        results["Baseline"]["prompts"].append(baseline_prompts)

        t = time.time()
        iknow_score, iknow_prompts = base.run_iknow(
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
        iknow_ms = audio_ms + (time.time() - t) * 1000.0
        results["iKnow"]["ranks"].append(base.rank_of_true(iknow_score, true_indices))
        results["iKnow"]["times"].append(iknow_ms)
        results["iKnow"]["prompts"].append(iknow_prompts)

        t = time.time()
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
        full_ms = audio_ms + (time.time() - t) * 1000.0
        results["Full2Hop"]["ranks"].append(base.rank_of_true(full_score, true_indices))
        results["Full2Hop"]["times"].append(full_ms)
        results["Full2Hop"]["prompts"].append(full_prompts)

        t = time.time()
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
        orig_ms = audio_ms + (time.time() - t) * 1000.0
        results["Selective2Hop_OriginalAgg"]["ranks"].append(base.rank_of_true(orig_score, true_indices))
        results["Selective2Hop_OriginalAgg"]["times"].append(orig_ms)
        results["Selective2Hop_OriginalAgg"]["prompts"].append(orig_prompts)
        results["Selective2Hop_OriginalAgg"]["hop2_activation_sample"].append(bool(orig_extra["hop2_activated"]))
        results["Selective2Hop_OriginalAgg"]["hop2_activation_candidate"].append(float(orig_extra["candidate_level_activation_rate"]))
        results["Selective2Hop_OriginalAgg"]["alphas"].append(alpha_dynamic)

        t = time.time()
        ours_score, ours_prompts, ours_extra = base.run_selective2hop(
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
        ours_ms = audio_ms + (time.time() - t) * 1000.0
        results["Selective2Hop_Ours"]["ranks"].append(base.rank_of_true(ours_score, true_indices))
        results["Selective2Hop_Ours"]["times"].append(ours_ms)
        results["Selective2Hop_Ours"]["prompts"].append(ours_prompts)
        results["Selective2Hop_Ours"]["hop2_activation_sample"].append(bool(ours_extra["hop2_activated"]))
        results["Selective2Hop_Ours"]["hop2_activation_candidate"].append(float(ours_extra["candidate_level_activation_rate"]))
        results["Selective2Hop_Ours"]["alphas"].append(alpha_dynamic)

        base_top = cos_sim_orig[top_indices]
        base_top_sorted, _ = torch.sort(base_top, descending=True)
        base_top1 = float(base_top_sorted[0].item())
        base_top2 = float(base_top_sorted[1].item()) if len(base_top_sorted) > 1 else base_top1
        base_margin = base_top1 - base_top2
        orig_top1 = float(torch.max(orig_score[top_indices]).item())
        ours_top1 = float(torch.max(ours_score[top_indices]).item())
        orig_gain = orig_top1 - base_top1
        ours_gain = ours_top1 - base_top1
        feature_vec = np.array([base_top1, base_margin, orig_gain, ours_gain], dtype=np.float32)

        true_idx = int(true_indices[0])
        orig_rank = base.rank_of_true(orig_score, true_indices)
        ours_rank = base.rank_of_true(ours_score, true_indices)
        route_label = choose_label(
            orig_rank,
            ours_rank,
            float(orig_score[true_idx].item()),
            float(ours_score[true_idx].item()),
        )

        samples.append(
            {
                "feature": feature_vec,
                "label": route_label,
                "true_indices": true_indices,
                "orig_score": orig_score.detach().cpu(),
                "ours_score": ours_score.detach().cpu(),
                "router_time_ms": (orig_ms + ours_ms - baseline_ms),
                "router_prompts": max(orig_prompts, ours_prompts),
                "hop2_sample": bool(orig_extra["hop2_activated"] or ours_extra["hop2_activated"]),
                "hop2_candidate": max(
                    float(orig_extra["candidate_level_activation_rate"]),
                    float(ours_extra["candidate_level_activation_rate"]),
                ),
                "alpha": alpha_dynamic,
            }
        )

    return samples, results, skipped


def apply_router_results(base, samples, probs, results):
    for sample, prob in zip(samples, probs):
        choose_orig = float(prob) > 0.5
        score = sample["orig_score"].clone() if choose_orig else sample["ours_score"].clone()
        results["Selective2Hop_HardRouter"]["ranks"].append(base.rank_of_true(score, sample["true_indices"]))
        results["Selective2Hop_HardRouter"]["times"].append(float(sample["router_time_ms"]))
        results["Selective2Hop_HardRouter"]["prompts"].append(float(sample["router_prompts"]))
        results["Selective2Hop_HardRouter"]["hop2_activation_sample"].append(bool(sample["hop2_sample"]))
        results["Selective2Hop_HardRouter"]["hop2_activation_candidate"].append(float(sample["hop2_candidate"]))
        results["Selective2Hop_HardRouter"]["alphas"].append(float(sample["alpha"]))
        results["Selective2Hop_HardRouter"]["route_choose_orig"].append(1.0 if choose_orig else 0.0)


def run_router_dataset(module_path, output_json, launch_title):
    set_seed(SEED)
    base = load_base(module_path)
    print(launch_title)
    print(f"Device: {base.DEVICE}")
    print("Hard-router heads: OriginalAgg vs Ours (Selective2Hop_StaticAlpha)")
    print("Router features: baseline top1, baseline top1-top2 margin, OriginalAgg gain, Ours gain")
    print(f"Cross-validation: {N_FOLDS}-fold, 2-layer MLP, hidden={HIDDEN_DIM}, dropout={DROPOUT}, epochs={EPOCHS}")

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

    samples, results, skipped = collect_samples(
        base,
        clap_model,
        text_embeds,
        dataset,
        label_classes,
        kg_classes,
        class_labels_set,
        hop1_relations,
        hop2_relations,
        get_tails,
        prompt_map,
    )
    print(f"[RouterCV] collected samples: {len(samples)}, skipped: {skipped}")

    features = np.stack([sample["feature"] for sample in samples], axis=0)
    labels = np.array([sample["label"] for sample in samples], dtype=np.float32)
    probs = train_and_predict(features, labels)
    apply_router_results(base, samples, probs, results)

    title = "Hard Router Ablation"
    print_tables(base, results, title)
    save_results(
        base,
        results,
        output_json,
        meta={
            "router_type": "hard_switch_mlp",
            "head_a": "Selective2Hop_OriginalAgg",
            "head_b": "Selective2Hop_Ours",
            "features": ["baseline_top1", "baseline_margin", "originalagg_gain", "ours_gain"],
            "n_folds": N_FOLDS,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "skipped_count": skipped,
        },
    )
