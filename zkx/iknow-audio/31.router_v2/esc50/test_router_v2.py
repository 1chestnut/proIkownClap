# -*- coding: utf-8 -*-
from router_common import run_router_dataset


if __name__ == "__main__":
    run_router_dataset(
        module_path="/data/zkx/zkx/iknow-audio/29.router/esc50/esc_router_base.py",
        output_json="/data/zkx/zkx/iknow-audio/31.router_v2/esc50/results_router_v2.json",
        launch_title="Starting ESC50 RouterV2 over OriginalAgg and Ours",
    )
