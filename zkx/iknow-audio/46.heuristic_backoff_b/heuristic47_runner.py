import os

from heuristic47_common import DatasetSpec, run_heuristic_variant


ROOT = "/data/zkx/zkx/iknow-audio/39.materialize_features"
OUT_ROOTS = {
    "heuristic_backoff_a": "/data/zkx/zkx/iknow-audio/45.heuristic_backoff_a",
    "heuristic_backoff_b": "/data/zkx/zkx/iknow-audio/46.heuristic_backoff_b",
    "heuristic_backoff_c": "/data/zkx/zkx/iknow-audio/47.heuristic_backoff_c",
}

DATASETS = [
    DatasetSpec("esc50", f"{ROOT}/esc50/esc50_features.csv"),
    DatasetSpec("tut2017", f"{ROOT}/tut2017/tut2017_features.csv"),
    DatasetSpec("dcase", f"{ROOT}/dcase/dcase_features.csv"),
    DatasetSpec("usk80", f"{ROOT}/usk80/usk80_features.csv"),
]


def main() -> None:
    target_name = os.environ["LODO_TARGET"]
    variant = os.environ["HEUR_VARIANT"]
    out_dir = os.path.join(OUT_ROOTS[variant], target_name)
    run_heuristic_variant(target_name, DATASETS, out_dir, variant)


if __name__ == "__main__":
    main()
