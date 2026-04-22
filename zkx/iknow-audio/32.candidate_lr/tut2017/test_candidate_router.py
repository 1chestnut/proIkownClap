# -*- coding: utf-8 -*-
from router_common import run_candidate_router_dataset


if __name__ == "__main__":
    run_candidate_router_dataset(
        module_path="/data/zkx/zkx/iknow-audio/26.sigmoid_moe/tut2017/test_sigmoid_moe.py",
        output_json="/data/zkx/zkx/iknow-audio/32.candidate_lr/tut2017/results_candidate_router_lr.json",
        launch_title="Starting TUT2017 CandidateRouter (LogisticRegression) over OriginalAgg and Ours",
        model_type="lr",
    )
