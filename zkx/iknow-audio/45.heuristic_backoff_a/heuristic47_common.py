import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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


def _sample_level_unique(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["sample_id", "baseline_top1", "baseline_margin", "entropy", "hop2_activation", "prompt_count"]
    return df[cols].drop_duplicates(subset=["sample_id"]).copy()


def _quantile_or_default(series: pd.Series, q: float, default: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return default
    return float(clean.quantile(q))


def compute_thresholds(train_df: pd.DataFrame, variant: str) -> Dict[str, float]:
    sample_df = _sample_level_unique(train_df)
    tau = _quantile_or_default(sample_df["baseline_margin"], 0.25, 0.05)

    key_df = train_df[train_df["is_key_candidate"].astype(int) == 1].copy()
    pos_gain = key_df.loc[key_df["orig_minus_base"] > 0, "orig_minus_base"]
    pos_margin = key_df.loc[key_df["orig_minus_ours"] > 0, "orig_minus_ours"]
    gamma = _quantile_or_default(pos_gain, 0.25, 0.0)
    delta = _quantile_or_default(pos_margin, 0.25, 0.0)

    if variant == "heuristic_backoff_a":
        gamma = 0.0
        delta = None
    elif variant == "heuristic_backoff_b":
        delta = None
    elif variant == "heuristic_backoff_c":
        pass
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    return {
        "tau": tau,
        "gamma": gamma,
        "delta": delta,
    }


def apply_heuristic(df: pd.DataFrame, thresholds: Dict[str, float], variant: str) -> pd.DataFrame:
    out = df.copy()
    margin_gate = out["baseline_margin"] < thresholds["tau"]
    gain_gate = out["orig_minus_base"] > thresholds["gamma"]
    if variant == "heuristic_backoff_c":
        delta_gate = out["orig_minus_ours"] > thresholds["delta"]
    else:
        delta_gate = True

    use_orig = (
        (out["is_key_candidate"].astype(int) == 1)
        & margin_gate
        & gain_gate
        & delta_gate
    )
    out["heuristic_choose_orig"] = use_orig.astype(int)
    out["heuristic_score"] = np.where(use_orig, out["orig_score"], out["ours_score"])
    out["oracle_score"] = np.where(out["oracle_label"].astype(int) == 1, out["orig_score"], out["ours_score"])
    return out


def _bucket_profile(df: pd.DataFrame, feature_col: str, label_col: str, bins: int = 4) -> Dict[str, float]:
    key_df = df[df["is_key_candidate"].astype(int) == 1].copy()
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


def run_heuristic_variant(target_name: str, dataset_specs: List[DatasetSpec], out_dir: str, variant: str) -> None:
    tables = {spec.name: load_feature_table(spec.csv_path) for spec in dataset_specs}
    target_df = tables[target_name]
    train_df = pd.concat([df for name, df in tables.items() if name != target_name], ignore_index=True)

    thresholds = compute_thresholds(train_df, variant)
    eval_df = apply_heuristic(target_df, thresholds, variant)

    metrics = {
        "OriginalAgg": _rank_metrics(eval_df, "orig_score"),
        "Ours": _rank_metrics(eval_df, "ours_score"),
        "OracleCandidate": _rank_metrics(eval_df, "oracle_score"),
        variant: _rank_metrics(eval_df, "heuristic_score"),
    }

    key_mask = eval_df["is_key_candidate"].astype(int) == 1
    gt_mask = eval_df["is_ground_truth"].astype(int) == 1
    neg_mask = ~gt_mask

    diagnostics = {
        "variant_name": variant,
        "thresholds": thresholds,
        "route_choose_orig_ratio": float(eval_df["heuristic_choose_orig"].mean()),
        "route_choose_orig_gt_ratio": float(eval_df.loc[gt_mask, "heuristic_choose_orig"].mean()),
        "route_choose_orig_non_gt_ratio": float(eval_df.loc[neg_mask, "heuristic_choose_orig"].mean()),
        "oracle_choose_orig_gt_ratio": float(eval_df.loc[gt_mask, "oracle_label"].astype(int).mean()),
        "oracle_choose_orig_non_gt_ratio": float(eval_df.loc[neg_mask, "oracle_label"].astype(int).mean()),
        "oracle_match_accuracy": float(
            np.mean(
                eval_df.loc[key_mask, "heuristic_choose_orig"].astype(int).to_numpy()
                == eval_df.loc[key_mask, "oracle_label"].astype(int).to_numpy()
            )
        ) if np.any(key_mask) else 0.0,
        "margin_gate_ratio": float((eval_df["baseline_margin"] < thresholds["tau"]).mean()),
        "positive_gain_ratio": float((eval_df["orig_minus_base"] > thresholds["gamma"]).mean()),
        "heuristic_margin_bucket_profile": _bucket_profile(eval_df, "baseline_margin", "heuristic_choose_orig"),
        "oracle_margin_bucket_profile": _bucket_profile(eval_df, "baseline_margin", "oracle_label"),
        "heuristic_entropy_bucket_profile": _bucket_profile(eval_df, "entropy", "heuristic_choose_orig"),
        "oracle_entropy_bucket_profile": _bucket_profile(eval_df, "entropy", "oracle_label"),
    }

    os.makedirs(out_dir, exist_ok=True)
    eval_df.to_csv(os.path.join(out_dir, f"{target_name}_{variant}_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "results_heuristic_variant.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_name": target_name,
                "variant_name": variant,
                "metrics": metrics,
                "diagnostics": diagnostics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Target: {target_name}")
    print(f"Variant: {variant}")
    print()
    print("Method                 Hit@1   Hit@3   Hit@5    MRR")
    print("----------------------------------------------------")
    for name in ["OriginalAgg", "Ours", "OracleCandidate", variant]:
        row = metrics[name]
        print(f"{name:<22} {row['Hit@1']:>6.2f} {row['Hit@3']:>7.2f} {row['Hit@5']:>7.2f} {row['MRR']:>7.2f}")
    print()
    print("Diagnostics")
    for key, value in diagnostics.items():
        print(f"- {key}: {value}")
