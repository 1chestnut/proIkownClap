#!/usr/bin/env bash
set -e
source /home/star/anaconda3/etc/profile.d/conda.sh
conda activate zkx
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:$LD_LIBRARY_PATH

WORKER="/data/zkx/zkx/iknow-audio/51.kappa_sweep_ablation17/kappa51_worker.sh"

slot1() {
  "$WORKER" "/data/zkx/zkx/iknow-audio/17.消融/tut2017/test_ablation.py" "/data/zkx/zkx/iknow-audio/51.kappa_sweep_ablation17/tut2017" 5 1 "TUT17"
  "$WORKER" "/data/zkx/zkx/iknow-audio/17.消融/tut2017/test_ablation.py" "/data/zkx/zkx/iknow-audio/51.kappa_sweep_ablation17/tut2017" 50 1 "TUT17"
}

slot2() {
  "$WORKER" "/data/zkx/zkx/iknow-audio/17.消融/esc50/test_ablation.py" "/data/zkx/zkx/iknow-audio/51.kappa_sweep_ablation17/esc50" 20 1 "ESC17"
}

slot1 &
slot2 &
wait
