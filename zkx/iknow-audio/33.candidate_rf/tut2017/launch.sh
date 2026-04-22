#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:${LD_LIBRARY_PATH:-}
cd /data/zkx/zkx/iknow-audio/33.candidate_rf/tut2017
rm -f tut2017_candidate_rf.log
/home/star/anaconda3/envs/zkx/bin/python -u test_candidate_router.py 2>&1 | tee tut2017_candidate_rf.log
