# -*- coding: utf-8 -*-
from oracle35_common import run_oracle_dataset


if __name__ == "__main__":
    run_oracle_dataset(
        module_path="/data/zkx/zkx/iknow-audio/26.sigmoid_moe/tut2017/test_sigmoid_moe.py",
        output_json="/data/zkx/zkx/iknow-audio/35.oracle/tut2017/results_oracle.json",
        launch_title="Starting TUT2017 Oracle analysis over OriginalAgg and Ours",
    )
