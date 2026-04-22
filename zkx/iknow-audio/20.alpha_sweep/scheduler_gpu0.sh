#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib
echo "START SCHED GPU0 $(date)" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/scheduler_gpu0.log
pids=()
labels=()
launch_job() {
  local job="$1"
  echo "===== START ${job} $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/scheduler_gpu0.log
  (cd "$job" && ./launch.sh) &
  pids+=("$!")
  labels+=("$job")
}
reap_one() {
  while true; do
    for i in "${!pids[@]}"; do
      local pid="${pids[$i]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || true
        echo "===== END ${labels[$i]} $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/scheduler_gpu0.log
        unset "pids[$i]"
        unset "labels[$i]"
        pids=("${pids[@]}")
        labels=("${labels[@]}")
        return 0
      fi
    done
    sleep 5
  done
}
launch_job "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0408_escfix/esc50"
launch_job "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/fsd50"
reap_one
launch_job "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/fsd50"
reap_one
launch_job "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/dcase"
reap_one
launch_job "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/dcase"
while [ ${#pids[@]} -gt 0 ]; do
  reap_one
done
echo "DONE SCHED GPU0 $(date)" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/scheduler_gpu0.log
