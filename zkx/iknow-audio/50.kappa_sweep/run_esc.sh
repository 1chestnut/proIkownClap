#!/usr/bin/env bash
set -e
source /home/star/anaconda3/etc/profile.d/conda.sh
conda activate zkx
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:$LD_LIBRARY_PATH
for k in 5 10 20 50 100; do
  echo [RUN] ESC kappa=$k | tee -a /data/zkx/zkx/iknow-audio/50.kappa_sweep/esc50/sweep.log
  python /data/zkx/zkx/iknow-audio/50.kappa_sweep_runner.py --module-path /data/zkx/zkx/iknow-audio/16.综合/esc50/test.py --kappa $k --output-json /data/zkx/zkx/iknow-audio/50.kappa_sweep/esc50/results_k${k}.json 2>&1 | tee /data/zkx/zkx/iknow-audio/50.kappa_sweep/esc50/k${k}.log
done
