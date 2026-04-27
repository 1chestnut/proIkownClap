#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/zkx/zkx/iknow-audio/56.gated_superpot_ablation17"
RUNNER="${ROOT}/structure56_gated_runner.py"
WORKER="${ROOT}/structure56_worker.sh"

bash "${WORKER}" \
  "${RUNNER}" \
  "/data/zkx/zkx/iknow-audio/17_ablation/tut2017/test_ablation.py" \
  "0" \
  "${ROOT}/tut2017/results_gated.json" \
  "${ROOT}/gpu0_tut.log"

bash "${WORKER}" \
  "${RUNNER}" \
  "/data/zkx/zkx/iknow-audio/17_ablation/esc50/test_ablation.py" \
  "0" \
  "${ROOT}/esc50/results_gated.json" \
  "${ROOT}/gpu0_esc.log"
