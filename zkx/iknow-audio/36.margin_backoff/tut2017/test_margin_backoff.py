# -*- coding: utf-8 -*-
from backoff36_common import run_backoff_dataset


if __name__ == "__main__":
    run_backoff_dataset(
        module_path="/data/zkx/zkx/iknow-audio/26.sigmoid_moe/tut2017/test_sigmoid_moe.py",
        output_json="/data/zkx/zkx/iknow-audio/36.margin_backoff/tut2017/results_margin_backoff.json",
        launch_title="Starting TUT2017 confidence-gated margin backoff over OriginalAgg and Ours",
    )
