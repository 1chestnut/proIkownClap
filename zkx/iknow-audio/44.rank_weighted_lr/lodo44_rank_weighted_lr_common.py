import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "baseline_top1",
    "baseline_margin",
    "entropy",
    "hop2_activation",
    "prompt_count_log1p",
    "base_score",
    "orig_minus_base",
    "ours_minus_base",
    "orig_minus_ours",
    "base_rank",
    "orig_rank",
    "ours_rank",
    "min_rank",
    "rank_gap",
]


@dataclass
class DatasetSpec:
    name: str
    csv_path: str


def load_feature_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "\ufeffdataset_name" in df.columns:
        df = df.rename(columns={"\ufeffdataset_name": "dataset_name"})
    return add_rank_features(df)


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["base_rank"] = (
        out.groupby("sample_id")["base_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    out["orig_rank"] = (
        out.groupby("sample_id")["orig_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    out["ours_rank"] = (
        out.groupby("sample_id")["ours_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    out["min_rank"] = out[["base_rank", "orig_rank", "ours_rank"]].min(axis=1)
    out["rank_gap"] = out["orig_rank"] - out["ours_rank"]
    return out


def _rank_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    hit1 = hit3 = hit5 = 0
    mrr = 0.0
    sample_count = 0
    for _, group in df.groupby("sample_id", sort=False):
        sample_count += 1
        ordered = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        gt_rows = ordered.index[ordered["is_ground_truth"].astype(int) == 1].tolist()
        if not gt_rows:
            continue
        rank = gt_rows[0] + 1
        hit1 += int(rank <= 1)
        hit3 += int(rank <= 3)
        hit5 += int(rank <= 5)
        mrr += 1.0 / rank
    if sample_count == 0:
        return {"Hit@1": 0.0, "Hit@3": 0.0, "Hit@5": 0.0, "MRR": 0.0}
    return {
        "Hit@1": round(100.0 * hit1 / sample_count, 2),
        "Hit@3": round(100.0 * hit3 / sample_count, 2),
        "Hit@5": round(100.0 * hit5 / sample_count, 2),
        "MRR": round(100.0 * mrr / sample_count, 2),
    }


def _safe_float_dict(arr: np.ndarray, cols: List[str]) -> Dict[str, float]:
    return {col: float(val) for col, val in zip(cols, arr.tolist())}


def _prob_stats(probs: np.ndarray) -> Dict[str, float]:
    if probs.size == 0:
        return {"prob_low_ratio": 0.0, "prob_mid_ratio": 0.0, "prob_high_ratio": 0.0}
    return {
        "prob_low_ratio": float(np.mean(probs < 0.3)),
        "prob_mid_ratio": float(np.mean((probs >= 0.3) & (probs <= 0.7))),
        "prob_high_ratio": float(np.mean(probs > 0.7)),
    }


def _compute_gap_series(df: pd.DataFrame, score_col: str) -> pd.Series:
    values = {}
    for sample_id, group in df.groupby("sample_id", sort=False):
        gt_scores = group.loc[group["is_ground_truth"].astype(int) == 1, score_col].tolist()
        neg_scores = group.loc[group["is_ground_truth"].astype(int) == 0, score_col].tolist()
        if not gt_scores or not neg_scores:
            continue
        values[sample_id] = float(gt_scores[0] - max(neg_scores))
    return pd.Series(values, dtype=float)


def _top1_labels(df: pd.DataFrame, score_col: str) -> Dict[str, str]:
    labels = {}
    for sample_id, group in df.groupby("sample_id", sort=False):
        ordered = group.sort_values(score_col, ascending=False)
        labels[sample_id] = str(ordered.iloc[0]["candidate_class"])
    return labels


def _true_labels(df: pd.DataFrame) -> Dict[str, str]:
    labels = {}
    gt_df = df[df["is_ground_truth"].astype(int) == 1]
    for sample_id, group in gt_df.groupby("sample_id", sort=False):
        labels[sample_id] = str(group.iloc[0]["candidate_class"])
    return labels


def _bucket_profile(df: pd.DataFrame, feature_col: str, label_col: str, bins: int = 4) -> Dict[str, float]:
    key_df = df[df["is_router_candidate"].astype(int) == 1].copy()
    if key_df.empty:
        return {}
    try:
        bucket_ids = pd.qcut(key_df[feature_col], q=bins, duplicates="drop")
    except ValueError:
        return {}
    out = {}
    for bucket, group in key_df.groupby(bucket_ids, observed=True):
        out[str(bucket)] = float(group[label_col].mean())
    return out


def select_router_candidates(df: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for _, group in df.groupby("sample_id", sort=False):
        idx = set()
        idx.update(group.index[group["is_ground_truth"].astype(int) == 1].tolist())
        idx.update(group.nsmallest(2, "base_rank").index.tolist())
        idx.update(group.nsmallest(2, "orig_rank").index.tolist())
        idx.update(group.nsmallest(2, "ours_rank").index.tolist())
        div = (group["orig_score"] - group["ours_score"]).abs()
        idx.update(group.loc[div.nlargest(1).index].index.tolist())
        selected.extend(sorted(idx))
    out = df.loc[selected].copy()
    out["is_router_candidate"] = 1
    return out


def add_conditional_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    strong_comp = (
        (out["is_ground_truth"].astype(int) == 0)
        & (
            (out["base_rank"] <= 2)
            | (out["orig_rank"] <= 2)
            | (out["ours_rank"] <= 2)
        )
    )
    weights = np.ones(len(out), dtype=float)
    gt_orig = (out["is_ground_truth"].astype(int) == 1) & (out["oracle_label"].astype(int) == 1)
    gt_ours = (out["is_ground_truth"].astype(int) == 1) & (out["oracle_label"].astype(int) == 0)
    weights[gt_orig.to_numpy()] = 5.0
    weights[gt_ours.to_numpy()] = 2.0
    weights[strong_comp.to_numpy()] = np.maximum(weights[strong_comp.to_numpy()], 2.0)
    out["sample_weight"] = weights
    out["is_strong_competitor"] = strong_comp.astype(int)
    return out


def prepare_target_table(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["router_prob"] = probs
    use_orig = (out["is_router_candidate"].astype(int) == 1) & (out["router_prob"] > 0.5)
    out["router_choose_orig"] = use_orig.astype(int)
    out["router_score"] = np.where(use_orig, out["orig_score"], out["ours_score"])
    out["oracle_score"] = np.where(out["oracle_label"].astype(int) == 1, out["orig_score"], out["ours_score"])
    return out


def run_lodo_rank_weighted_lr(target_name: str, dataset_specs: List[DatasetSpec], out_dir: str) -> None:
    tables = {spec.name: load_feature_table(spec.csv_path) for spec in dataset_specs}
    target_df = tables[target_name]
    train_df_full = pd.concat([df for name, df in tables.items() if name != target_name], ignore_index=True)
    train_df_full = train_df_full[train_df_full["is_key_candidate"].astype(int) == 1].copy()
    train_df_full = train_df_full.dropna(subset=FEATURE_COLUMNS + ["oracle_label"])

    train_df = select_router_candidates(train_df_full)
    train_df = add_conditional_weights(train_df)

    y_train = train_df["oracle_label"].astype(int).to_numpy()
    X_train = train_df[FEATURE_COLUMNS].astype(float).to_numpy()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train, sample_weight=train_df["sample_weight"].to_numpy())

    target_eval = target_df.dropna(subset=FEATURE_COLUMNS).copy()
    target_eval = select_router_candidates(target_eval)
    target_eval = target_df.merge(
        target_eval[["sample_id", "candidate_index", "is_router_candidate"]],
        on=["sample_id", "candidate_index"],
        how="left",
    )
    target_eval["is_router_candidate"] = target_eval["is_router_candidate"].fillna(0).astype(int)

    X_test = target_eval[FEATURE_COLUMNS].astype(float).to_numpy()
    X_test_scaled = scaler.transform(X_test)
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    target_eval = prepare_target_table(target_eval, probs)

    router_mask = target_eval["is_router_candidate"].astype(int) == 1
    gt_mask = target_eval["is_ground_truth"].astype(int) == 1
    neg_mask = ~gt_mask
    oracle_match = (
        target_eval.loc[router_mask, "router_choose_orig"].astype(int).to_numpy()
        == target_eval.loc[router_mask, "oracle_label"].astype(int).to_numpy()
    )

    ours_gap = _compute_gap_series(target_eval, "ours_score")
    orig_gap = _compute_gap_series(target_eval, "orig_score")
    router_gap = _compute_gap_series(target_eval, "router_score")
    oracle_gap = _compute_gap_series(target_eval, "oracle_score")

    true_labels = _true_labels(target_eval)
    top1_ours = _top1_labels(target_eval, "ours_score")
    top1_orig = _top1_labels(target_eval, "orig_score")
    top1_router = _top1_labels(target_eval, "router_score")
    common_ids = sorted(set(true_labels) & set(top1_ours) & set(top1_orig) & set(top1_router))
    top1_changed = [top1_router[sid] != top1_ours[sid] for sid in common_ids]
    top1_corrected_vs_ours = [
        (top1_router[sid] == true_labels[sid]) and (top1_ours[sid] != true_labels[sid])
        for sid in common_ids
    ]
    top1_corrected_vs_orig = [
        (top1_router[sid] == true_labels[sid]) and (top1_orig[sid] != true_labels[sid])
        for sid in common_ids
    ]

    metrics = {
        "OriginalAgg": _rank_metrics(target_eval, "orig_score"),
        "Ours": _rank_metrics(target_eval, "ours_score"),
        "OracleCandidate": _rank_metrics(target_eval, "oracle_score"),
        "rank_weighted_lr": _rank_metrics(target_eval, "router_score"),
    }

    diagnostics = {
        "variant_name": "rank_weighted_lr",
        "train_rows_before": int(len(train_df_full)),
        "train_rows_after": int(len(train_df)),
        "router_candidate_keep_ratio": float(len(train_df) / max(len(train_df_full), 1)),
        "train_choose_orig_ratio": float(np.mean(y_train)),
        "gt_weight_mean": float(train_df["sample_weight"].mean()),
        "strong_competitor_ratio_train": float(train_df["is_strong_competitor"].mean()),
        "route_choose_orig_ratio": float(target_eval["router_choose_orig"].mean()),
        "route_choose_orig_gt_ratio": float(target_eval.loc[gt_mask, "router_choose_orig"].mean()),
        "route_choose_orig_non_gt_ratio": float(target_eval.loc[neg_mask, "router_choose_orig"].mean()),
        "oracle_choose_orig_gt_ratio": float(target_eval.loc[gt_mask, "oracle_label"].astype(int).mean()),
        "oracle_choose_orig_non_gt_ratio": float(target_eval.loc[neg_mask, "oracle_label"].astype(int).mean()),
        "oracle_match_accuracy": float(np.mean(oracle_match)) if oracle_match.size else 0.0,
        "true_candidate_match_accuracy": float(
            np.mean(
                target_eval.loc[gt_mask & router_mask, "router_choose_orig"].astype(int).to_numpy()
                == target_eval.loc[gt_mask & router_mask, "oracle_label"].astype(int).to_numpy()
            )
        ) if np.any(gt_mask & router_mask) else 0.0,
        "hard_negative_match_accuracy": float(
            np.mean(
                target_eval.loc[neg_mask & router_mask, "router_choose_orig"].astype(int).to_numpy()
                == target_eval.loc[neg_mask & router_mask, "oracle_label"].astype(int).to_numpy()
            )
        ) if np.any(neg_mask & router_mask) else 0.0,
        "router_candidate_ratio_test": float(np.mean(router_mask)),
        "true_gap_improved_vs_ours_ratio": float(np.mean((router_gap - ours_gap).fillna(0.0) > 0)),
        "true_gap_improved_vs_orig_ratio": float(np.mean((router_gap - orig_gap).fillna(0.0) > 0)),
        "oracle_gap_above_router_ratio": float(np.mean((oracle_gap - router_gap).fillna(0.0) > 0)),
        "top1_changed_ratio": float(np.mean(top1_changed)) if top1_changed else 0.0,
        "top1_corrected_vs_ours_ratio": float(np.mean(top1_corrected_vs_ours)) if top1_corrected_vs_ours else 0.0,
        "top1_corrected_vs_orig_ratio": float(np.mean(top1_corrected_vs_orig)) if top1_corrected_vs_orig else 0.0,
        **_prob_stats(probs),
        "coef_mean": _safe_float_dict(clf.coef_[0], FEATURE_COLUMNS),
        "coef_abs_mean": _safe_float_dict(np.abs(clf.coef_[0]), FEATURE_COLUMNS),
        "scaler_mean": _safe_float_dict(scaler.mean_, FEATURE_COLUMNS),
        "scaler_scale": _safe_float_dict(scaler.scale_, FEATURE_COLUMNS),
        "router_margin_bucket_profile": _bucket_profile(target_eval, "baseline_margin", "router_choose_orig"),
        "oracle_margin_bucket_profile": _bucket_profile(target_eval, "baseline_margin", "oracle_label"),
        "router_entropy_bucket_profile": _bucket_profile(target_eval, "entropy", "router_choose_orig"),
        "oracle_entropy_bucket_profile": _bucket_profile(target_eval, "entropy", "oracle_label"),
    }

    os.makedirs(out_dir, exist_ok=True)
    target_eval.to_csv(os.path.join(out_dir, f"{target_name}_rank_weighted_lr_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "results_rank_weighted_lr.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_name": target_name,
                "feature_columns": FEATURE_COLUMNS,
                "metrics": metrics,
                "diagnostics": diagnostics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Target: {target_name}")
    print("Variant: rank_weighted_lr")
    print()
    print("Method                 Hit@1   Hit@3   Hit@5    MRR")
    print("----------------------------------------------------")
    for name in ["OriginalAgg", "Ours", "OracleCandidate", "rank_weighted_lr"]:
        row = metrics[name]
        print(f"{name:<22} {row['Hit@1']:>6.2f} {row['Hit@3']:>7.2f} {row['Hit@5']:>7.2f} {row['MRR']:>7.2f}")
    print()
    print("Diagnostics")
    for key, value in diagnostics.items():
        if isinstance(value, dict):
            print(f"- {key}:")
            for sub_k, sub_v in value.items():
                print(f"  - {sub_k}: {sub_v}")
        else:
            print(f"- {key}: {value}")
