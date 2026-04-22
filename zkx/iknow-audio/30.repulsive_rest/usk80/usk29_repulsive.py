# -*- coding: utf-8 -*-
import os

from repulsive_common import run_repulsive_dataset


if __name__ == "__main__":
    root = os.path.dirname(__file__)
    run_repulsive_dataset(
        module_path=os.path.join(root, "usk_rep_base.py"),
        output_json="/data/zkx/zkx/iknow-audio/30.repulsive_rest/usk80/results_repulsive.json",
        launch_title="Starting US8K repulsive fusion check",
    )
