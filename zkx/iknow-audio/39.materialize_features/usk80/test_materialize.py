# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from materialize39_common import run_materialize


if __name__ == "__main__":
    root = "/data/zkx/zkx/iknow-audio/39.materialize_features/usk80"
    run_materialize(
        module_path="/data/zkx/zkx/iknow-audio/39.materialize_features/bases/usk_base.py",
        output_dir=root,
        launch_title="Starting USK80 feature materialization",
        dataset_name="usk80",
    )
