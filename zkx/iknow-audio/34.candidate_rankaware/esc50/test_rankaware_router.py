# -*- coding: utf-8 -*-
from router_common import run_rankaware_router_dataset


if __name__ == "__main__":
    run_rankaware_router_dataset(
        module_path="/data/zkx/zkx/iknow-audio/29.router/esc50/esc_router_base.py",
        output_json="/data/zkx/zkx/iknow-audio/34.candidate_rankaware/esc50/results_rankaware.json",
        launch_title="Starting ESC50 Candidate RankAware Router over OriginalAgg and Ours",
    )
