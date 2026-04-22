#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:${LD_LIBRARY_PATH:-}
cd /data/zkx/zkx/iknow-audio/31.router_v2/tut2017
rm -f tut2017_router_v2.log
/home/star/anaconda3/envs/zkx/bin/python -u test_router_v2.py 2>&1 | tee tut2017_router_v2.log
