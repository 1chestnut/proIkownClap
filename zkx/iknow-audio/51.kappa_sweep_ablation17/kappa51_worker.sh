#!/usr/bin/env bash
set -e
source /home/star/anaconda3/etc/profile.d/conda.sh
conda activate zkx
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:$LD_LIBRARY_PATH

module="$1"
outdir="$2"
k="$3"
gpu="$4"
ds="$5"

echo "[RUN] ${ds} kappa=${k} gpu=${gpu}" | tee -a "${outdir}/sweep.log"
python /data/zkx/zkx/iknow-audio/50.kappa_sweep_runner.py \
  --module-path "$module" \
  --kappa "$k" \
  --gpu-id "$gpu" \
  --output-json "${outdir}/results_k${k}.json" \
  2>&1 | tee "${outdir}/k${k}.log"
