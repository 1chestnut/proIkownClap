#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib
LOG="/data/zkx/zkx/iknow-audio/20.alpha_sweep/scheduler_gpu2_simple.log"
echo "START $(date)" | tee -a "$LOG"
jobs=(
  "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/usk80"
  "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/usk80"
  "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/esc50"
  "/data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/esc50"
)
p1=""; p2=""; l1=""; l2=""; idx=0
start_job() {
  local slot="$1"
  local job="${jobs[$idx]}"
  echo "START_JOB $job $(date)" | tee -a "$LOG"
  ( cd "$job" && ./launch.sh ) &
  local pid=$!
  if [ "$slot" = "1" ]; then p1="$pid"; l1="$job"; else p2="$pid"; l2="$job"; fi
  idx=$((idx+1))
}
if [ $idx -lt ${#jobs[@]} ]; then start_job 1; fi
if [ $idx -lt ${#jobs[@]} ]; then start_job 2; fi
while [ -n "$p1" ] || [ -n "$p2" ] || [ $idx -lt ${#jobs[@]} ]; do
  if [ -n "$p1" ] && ! kill -0 "$p1" 2>/dev/null; then
    wait "$p1" || true
    echo "END_JOB $l1 $(date)" | tee -a "$LOG"
    p1=""; l1=""
    if [ $idx -lt ${#jobs[@]} ]; then start_job 1; fi
  fi
  if [ -n "$p2" ] && ! kill -0 "$p2" 2>/dev/null; then
    wait "$p2" || true
    echo "END_JOB $l2 $(date)" | tee -a "$LOG"
    p2=""; l2=""
    if [ $idx -lt ${#jobs[@]} ]; then start_job 2; fi
  fi
  sleep 5
done
echo "DONE $(date)" | tee -a "$LOG"
