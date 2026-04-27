import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


def select_hard_candidates(df: pd.DataFrame, max_div_count: int = 2) -> pd.DataFrame:
    selected = []
    for _, group in df.groupby("sample_id", sort=False):
        idx = set()
        gt_rows = group.index[group["is_ground_truth"].astype(int) == 1].tolist()
        idx.update(gt_rows)
        idx.update(group.nlargest(2, "orig_score").index.tolist())
        idx.update(group.nlargest(2, "ours_score").index.tolist())
        div = (group["orig_score"] - group["ours_score"]).abs()
        idx.update(group.loc[div.nlargest(max_div_count).index].index.tolist())
        selected.extend(sorted(idx))
    out = df.loc[selected].copy()
    out["is_hard_selected"] = 1
    return out


def add_sample_weights(df: pd.DataFrame, gt_weight: float = 5.0) -> pd.DataFrame:
    out = df.copy()
    out["sample_weight"] = np.where(out["is_ground_truth"].astype(int) == 1, gt_weight, 1.0)
    return out


def prepare_target_table(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["router_prob"] = probs
    use_orig = (out["is_key_candidate"].astype(int) == 1) & (out["router_prob"] > 0.5)
    out["router_choose_orig"] = use_orig.astype(int)
    out["router_score"] = np.where(use_orig, out["orig_score"], out["ours_score"])
    out["oracle_score"] = np.where(out["oracle_label"].astype(int) == 1, out["orig_score"], out["ours_score"])
    return out


def build_model(model_kind: str):
    if model_kind == "lr":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
    if model_kind == "rf":
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model kind: {model_kind}")


def run_lodo_variant(
    target_name: str,
    dataset_specs: List[DatasetSpec],
    out_dir: str,
    variant_name: str,
    model_kind: str,
    use_sample_weight: bool,
) -> None:
    tables = {spec.name: load_feature_table(spec.csv_path) for spec in dataset_specs}
    target_df = tables[target_name]
    train_df_full = pd.concat([df for name, df in tables.items() if name != target_name], ignore_index=True)
    train_df_full = train_df_full[train_df_full["is_key_candidate"].astype(int) == 1].copy()
    train_df_full = train_df_full.dropna(subset=FEATURE_COLUMNS + ["oracle_label"])

    train_df = select_hard_candidates(train_df_full)
    if use_sample_weight:
        train_df = add_sample_weights(train_df, gt_weight=5.0)
    else:
        train_df["sample_weight"] = 1.0

    y_train = train_df["oracle_label"].astype(int).to_numpy()
    X_train = train_df[FEATURE_COLUMNS].astype(float).to_numpy()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = build_model(model_kind)
    fit_kwargs = {}
    if use_sample_weight:
        fit_kwargs["sample_weight"] = train_df["sample_weight"].to_numpy()
    clf.fit(X_train_scaled, y_train, **fit_kwargs)

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
        variant_name: _rank_metrics(target_eval, "router_score"),
    }

    diagnostics = {
        "variant_name": variant_name,
        "model_kind": model_kind,
        "train_rows_before": int(len(train_df_full)),
        "train_rows_after": int(len(train_df)),
        "hard_mining_keep_ratio": float(len(train_df) / max(len(train_df_full), 1)),
        "train_choose_orig_ratio": float(np.mean(y_train)),
        "gt_weight_mean": float(train_df["sample_weight"].mean()),
        "route_choose_orig_ratio": float(target_eval["router_choose_orig"].mean()),
        "oracle_match_accuracy": float(np.mean(oracle_match)) if oracle_match.size else 0.0,
        "key_candidate_ratio_test": float(np.mean(key_mask)),
        **_prob_stats(probs),
        "scaler_mean": _safe_float_dict(scaler.mean_, FEATURE_COLUMNS),
        "scaler_scale": _safe_float_dict(scaler.scale_, FEATURE_COLUMNS),
    }

    if model_kind == "lr":
        diagnostics["coef_mean"] = _safe_float_dict(clf.coef_[0], FEATURE_COLUMNS)
        diagnostics["coef_abs_mean"] = _safe_float_dict(np.abs(clf.coef_[0]), FEATURE_COLUMNS)
    else:
        diagnostics["feature_importance_mean"] = _safe_float_dict(clf.feature_importances_, FEATURE_COLUMNS)

    os.makedirs(out_dir, exist_ok=True)
    target_eval.to_csv(os.path.join(out_dir, f"{target_name}_{variant_name}_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "results_lodo_variant.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_name": target_name,
                "variant_name": variant_name,
                "feature_columns": FEATURE_COLUMNS,
                "metrics": metrics,
                "diagnostics": diagnostics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Target: {target_name}")
    print(f"Variant: {variant_name}")
    print()
    print("Method                 Hit@1   Hit@3   Hit@5    MRR")
    print("----------------------------------------------------")
    for name in ["OriginalAgg", "Ours", "OracleCandidate", variant_name]:
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
