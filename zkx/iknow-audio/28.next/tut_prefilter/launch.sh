#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
cd /data/zkx/zkx/iknow-audio/28.next/tut_prefilter
/home/star/anaconda3/envs/zkx/bin/python -u test_prefilter.py 2>&1 | tee tut_prefilter.log
