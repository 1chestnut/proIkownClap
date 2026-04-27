import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
]


@dataclass
class DatasetSpec:
    name: str
    csv_path: str


def load_feature_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "\ufeffdataset_name" in df.columns:
        df = df.rename(columns={"\ufeffdataset_name": "dataset_name"})
    return df


def prepare_target_table(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["router_prob"] = probs
    use_orig = (out["is_key_candidate"].astype(int) == 1) & (out["router_prob"] > 0.5)
    out["router_choose_orig"] = use_orig.astype(int)
    out["router_score"] = np.where(use_orig, out["orig_score"], out["ours_score"])
    out["oracle_score"] = np.where(out["oracle_label"].astype(int) == 1, out["orig_score"], out["ours_score"])
    return out


def _rank_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    hit1 = 0
    hit3 = 0
    hit5 = 0
    mrr = 0.0
    sample_count = 0

    for _, group in df.groupby("sample_id", sort=False):
        sample_count += 1
        ordered = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        gt_rows = ordered.index[ordered["is_ground_truth"].astype(int) == 1].tolist()
        if not gt_rows:
            continue
        rank = gt_rows[0] + 1
        if rank <= 1:
            hit1 += 1
        if rank <= 3:
            hit3 += 1
        if rank <= 5:
            hit5 += 1
        mrr += 1.0 / rank

    if sample_count == 0:
        return {"Hit@1": 0.0, "Hit@3": 0.0, "Hit@5": 0.0, "MRR": 0.0}

    return {
        "Hit@1": round(100.0 * hit1 / sample_count, 2),
        "Hit@3": round(100.0 * hit3 / sample_count, 2),
        "Hit@5": round(100.0 * hit5 / sample_count, 2),
        "MRR": round(100.0 * mrr / sample_count, 2),
    }


def _prob_stats(probs: np.ndarray) -> Dict[str, float]:
    if probs.size == 0:
        return {"prob_low_ratio": 0.0, "prob_mid_ratio": 0.0, "prob_high_ratio": 0.0}
    return {
        "prob_low_ratio": float(np.mean(probs < 0.3)),
        "prob_mid_ratio": float(np.mean((probs >= 0.3) & (probs <= 0.7))),
        "prob_high_ratio": float(np.mean(probs > 0.7)),
    }


def _safe_float_dict(arr: np.ndarray, cols: List[str]) -> Dict[str, float]:
    return {col: float(val) for col, val in zip(cols, arr.tolist())}


def run_lodo_pilot(target_name: str, dataset_specs: List[DatasetSpec], out_dir: str) -> None:
    tables = {spec.name: load_feature_table(spec.csv_path) for spec in dataset_specs}
    target_df = tables[target_name]
    train_dfs = [df for name, df in tables.items() if name != target_name]
    train_df = pd.concat(train_dfs, ignore_index=True)

    train_df = train_df[train_df["is_key_candidate"].astype(int) == 1].copy()
    train_df = train_df.dropna(subset=FEATURE_COLUMNS + ["oracle_label"])
    y_train = train_df["oracle_label"].astype(int).to_numpy()
    X_train = train_df[FEATURE_COLUMNS].astype(float).to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    target_eval = target_df.dropna(subset=FEATURE_COLUMNS).copy()
    X_test = target_eval[FEATURE_COLUMNS].astype(float).to_numpy()
    X_test_scaled = scaler.transform(X_test)
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    target_eval = prepare_target_table(target_eval, probs)

    key_mask = target_eval["is_key_candidate"].astype(int) == 1
    oracle_match = (
        target_eval.loc[key_mask, "router_choose_orig"].astype(int).to_numpy()
        == target_eval.loc[key_mask, "oracle_label"].astype(int).to_numpy()
    )

    metrics = {
        "OriginalAgg": _rank_metrics(target_eval, "orig_score"),
        "Ours": _rank_metrics(target_eval, "ours_score"),
        "OracleCandidate": _rank_metrics(target_eval, "oracle_score"),
        "LODOCandidateRouter": _rank_metrics(target_eval, "router_score"),
    }

    diagnostics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(target_eval)),
        "train_choose_orig_ratio": float(np.mean(y_train)),
        "route_choose_orig_ratio": float(target_eval["router_choose_orig"].mean()),
        "oracle_match_accuracy": float(np.mean(oracle_match)) if oracle_match.size else 0.0,
        "key_candidate_ratio_test": float(np.mean(key_mask)),
        **_prob_stats(probs),
        "coef_mean": _safe_float_dict(clf.coef_[0], FEATURE_COLUMNS),
        "coef_abs_mean": _safe_float_dict(np.abs(clf.coef_[0]), FEATURE_COLUMNS),
        "scaler_mean": _safe_float_dict(scaler.mean_, FEATURE_COLUMNS),
        "scaler_scale": _safe_float_dict(scaler.scale_, FEATURE_COLUMNS),
    }

    os.makedirs(out_dir, exist_ok=True)
    target_eval.to_csv(os.path.join(out_dir, f"{target_name}_pilot_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "results_lodo_router_pilot.json"), "w", encoding="utf-8") as f:
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
    print()
    print("Method                 Hit@1   Hit@3   Hit@5    MRR")
    print("----------------------------------------------------")
    for name in ["OriginalAgg", "Ours", "OracleCandidate", "LODOCandidateRouter"]:
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
