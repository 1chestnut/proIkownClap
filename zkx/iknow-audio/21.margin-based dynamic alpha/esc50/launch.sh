#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
cd '/data/zkx/zkx/iknow-audio/21.margin-based dynamic alpha/esc50'
/home/star/anaconda3/envs/zkx/bin/python -u test_margin_alpha.py 2>&1 | tee esc50_margin_alpha.log
