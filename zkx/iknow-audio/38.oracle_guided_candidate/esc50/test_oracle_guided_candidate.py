# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from oracle38_candidate_common import run_oracle_guided_candidate


if __name__ == "__main__":
    root = "/data/zkx/zkx/iknow-audio/38.oracle_guided_candidate"
    output_dir = os.path.join(root, "esc50")
    run_oracle_guided_candidate(
        root_dir=root,
        target_name="esc50",
        output_dir=output_dir,
        launch_title="Starting ESC50 Oracle-guided candidate router (leave-one-dataset-out)",
    )
