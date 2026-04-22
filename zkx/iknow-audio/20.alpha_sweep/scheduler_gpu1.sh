#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib
echo "START SCHED GPU1 $(date)" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/scheduler_gpu1.log
pids=()
labels=()
launch_job() {
  local job="$1"
  echo "===== START ${job} $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/scheduler_gpu1.log
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
        echo "===== END ${labels[$i]} $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/scheduler_gpu1.log
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
launch_job "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/audioset"
launch_job "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/audioset"
reap_one
launch_job "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/tut2017"
reap_one
launch_job "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/tut2017"
while [ ${#pids[@]} -gt 0 ]; do
  reap_one
done
echo "DONE SCHED GPU1 $(date)" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/scheduler_gpu1.log
