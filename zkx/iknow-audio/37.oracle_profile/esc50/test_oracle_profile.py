# -*- coding: utf-8 -*-
import os

from profile37_common import run_oracle_profile


if __name__ == "__main__":
    root = "/data/zkx/zkx/iknow-audio/37.oracle_profile/esc50"
    run_oracle_profile(
        module_path=os.path.join(root, "esc_router_base.py"),
        output_dir=root,
        launch_title="Starting ESC50 Oracle profile diagnostics",
    )
