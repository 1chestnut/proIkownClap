# -*- coding: utf-8 -*-
import os

from router_common import run_router_dataset


if __name__ == "__main__":
    root = os.path.dirname(__file__)
    run_router_dataset(
        module_path=os.path.join(root, "esc_router_base.py"),
        output_json="/data/zkx/zkx/iknow-audio/29.router/esc50/results_router.json",
        launch_title="Starting ESC50 hard router over OriginalAgg and Ours",
    )
