#!/usr/bin/env bash
set -e
cd /data/zkx/zkx/iknow-audio/19.alpha/esc50
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib
/home/star/anaconda3/envs/zkx/bin/python -u test_alpha.py 2>&1 | tee /data/zkx/zkx/iknow-audio/19.alpha/esc50/esc50_alpha.log
