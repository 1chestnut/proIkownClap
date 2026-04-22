# -*- coding: utf-8 -*-
import os

from backoff36_common import run_backoff_dataset


if __name__ == "__main__":
    root = os.path.dirname(__file__)
    run_backoff_dataset(
        module_path=os.path.join(root, "esc_router_base.py"),
        output_json="/data/zkx/zkx/iknow-audio/36.margin_backoff/esc50/results_margin_backoff.json",
        launch_title="Starting ESC50 confidence-gated margin backoff over OriginalAgg and Ours",
    )
