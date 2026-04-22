# -*- coding: utf-8 -*-
from router_common import run_router_dataset


if __name__ == "__main__":
    run_router_dataset(
        module_path="/data/zkx/zkx/iknow-audio/26.sigmoid_moe/tut2017/test_sigmoid_moe.py",
        output_json="/data/zkx/zkx/iknow-audio/29.router/tut2017/results_router.json",
        launch_title="Starting TUT2017 hard router over OriginalAgg and Ours",
    )
