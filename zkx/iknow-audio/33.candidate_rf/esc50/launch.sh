#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:${LD_LIBRARY_PATH:-}
cd /data/zkx/zkx/iknow-audio/33.candidate_rf/esc50
rm -f esc50_candidate_rf.log
/home/star/anaconda3/envs/zkx/bin/python -u test_candidate_router.py 2>&1 | tee esc50_candidate_rf.log
