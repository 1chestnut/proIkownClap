#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/zkx/zkx/iknow-audio/56.gated_superpot_ablation17"
RUNNER="${ROOT}/structure56_gated_runner.py"
WORKER="${ROOT}/structure56_worker.sh"
WAIT_FILE="/data/zkx/zkx/iknow-audio/55.superpot_clean_ablation17/dcase/results_superpot.json"

while [ ! -f "${WAIT_FILE}" ]; do
  sleep 120
done

bash "${WORKER}" \
  "${RUNNER}" \
  "/data/zkx/zkx/iknow-audio/17_ablation/dcase/test_ablation.py" \
  "1" \
  "${ROOT}/dcase/results_gated.json" \
  "${ROOT}/gpu1_dcase.log"
