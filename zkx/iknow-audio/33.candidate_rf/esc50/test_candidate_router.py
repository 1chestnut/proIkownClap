# -*- coding: utf-8 -*-
from router_common import run_candidate_router_dataset


if __name__ == "__main__":
    run_candidate_router_dataset(
        module_path="/data/zkx/zkx/iknow-audio/29.router/esc50/esc_router_base.py",
        output_json="/data/zkx/zkx/iknow-audio/33.candidate_rf/esc50/results_candidate_router_rf.json",
        launch_title="Starting ESC50 CandidateRouter (RandomForest) over OriginalAgg and Ours",
        model_type="rf",
    )
