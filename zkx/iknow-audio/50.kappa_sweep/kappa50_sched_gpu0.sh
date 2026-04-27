#!/usr/bin/env bash
set -e
source /home/star/anaconda3/etc/profile.d/conda.sh
conda activate zkx
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:$LD_LIBRARY_PATH

run_task() {
  local module="$1" outdir="$2" k="$3" gpu="$4" ds="$5"
  echo "[RUN] ${ds} kappa=${k} gpu=${gpu}" | tee -a "${outdir}/sweep.log"
  python /data/zkx/zkx/iknow-audio/50.kappa_sweep_runner.py \
    --module-path "$module" \
    --kappa "$k" \
    --gpu-id "$gpu" \
    --output-json "${outdir}/results_k${k}.json" \
    2>&1 | tee "${outdir}/k${k}.log"
}
export -f run_task

printf '%s\n' \
'/data/zkx/zkx/iknow-audio/16.综合/esc50/test.py|/data/zkx/zkx/iknow-audio/50.kappa_sweep/esc50|5|0|ESC' \
'/data/zkx/zkx/iknow-audio/16.综合/tut2017/test.py|/data/zkx/zkx/iknow-audio/50.kappa_sweep/tut2017|10|0|TUT' \
'/data/zkx/zkx/iknow-audio/16.综合/esc50/test.py|/data/zkx/zkx/iknow-audio/50.kappa_sweep/esc50|50|0|ESC' \
'/data/zkx/zkx/iknow-audio/16.综合/tut2017/test.py|/data/zkx/zkx/iknow-audio/50.kappa_sweep/tut2017|100|0|TUT' \
| xargs -I{} -P 2 bash -lc 'IFS="|" read -r module outdir k gpu ds <<< "$1"; run_task "$module" "$outdir" "$k" "$gpu" "$ds"' _ {}
