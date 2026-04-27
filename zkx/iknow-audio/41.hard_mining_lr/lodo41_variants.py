import os

from lodo41_variants_common import DatasetSpec, run_lodo_variant


ROOT = "/data/zkx/zkx/iknow-audio/39.materialize_features"
OUT_ROOT = "/data/zkx/zkx/iknow-audio"

DATASETS = [
    DatasetSpec("esc50", f"{ROOT}/esc50/esc50_features.csv"),
    DatasetSpec("tut2017", f"{ROOT}/tut2017/tut2017_features.csv"),
    DatasetSpec("dcase", f"{ROOT}/dcase/dcase_features.csv"),
    DatasetSpec("usk80", f"{ROOT}/usk80/usk80_features.csv"),
]

VARIANT_CONFIG = {
    "hard_lr": {
        "out_root": f"{OUT_ROOT}/41.hard_mining_lr",
        "model_kind": "lr",
        "use_sample_weight": False,
    },
    "weighted_lr": {
        "out_root": f"{OUT_ROOT}/42.weighted_lr",
        "model_kind": "lr",
        "use_sample_weight": True,
    },
    "weighted_rf": {
        "out_root": f"{OUT_ROOT}/43.weighted_rf",
        "model_kind": "rf",
        "use_sample_weight": True,
    },
}


def main() -> None:
    target_name = os.environ["LODO_TARGET"]
    variant_name = os.environ["LODO_VARIANT"]
    cfg = VARIANT_CONFIG[variant_name]
    out_dir = os.path.join(cfg["out_root"], target_name)
    run_lodo_variant(
        target_name=target_name,
        dataset_specs=DATASETS,
        out_dir=out_dir,
        variant_name=variant_name,
        model_kind=cfg["model_kind"],
        use_sample_weight=cfg["use_sample_weight"],
    )


if __name__ == "__main__":
    main()
