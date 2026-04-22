# -*- coding: utf-8 -*-
import os

from oracle35_common import run_oracle_dataset


if __name__ == "__main__":
    root = os.path.dirname(__file__)
    run_oracle_dataset(
        module_path=os.path.join(root, "esc_router_base.py"),
        output_json="/data/zkx/zkx/iknow-audio/35.oracle/esc50/results_oracle.json",
        launch_title="Starting ESC50 Oracle analysis over OriginalAgg and Ours",
    )
