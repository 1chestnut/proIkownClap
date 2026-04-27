import os

from lodo49_gate_rf_common import DatasetSpec, run_lodo_gate_rf


ROOT = "/data/zkx/zkx/iknow-audio/39.materialize_features"
OUT_ROOT = "/data/zkx/zkx/iknow-audio/49.gate_rf"

DATASETS = [
    DatasetSpec("esc50", f"{ROOT}/esc50/esc50_features.csv"),
    DatasetSpec("tut2017", f"{ROOT}/tut2017/tut2017_features.csv"),
    DatasetSpec("dcase", f"{ROOT}/dcase/dcase_features.csv"),
    DatasetSpec("usk80", f"{ROOT}/usk80/usk80_features.csv"),
]


def main() -> None:
    target_name = os.environ["LODO_TARGET"]
    out_dir = os.path.join(OUT_ROOT, target_name)
    run_lodo_gate_rf(target_name, DATASETS, out_dir)


if __name__ == "__main__":
    main()
