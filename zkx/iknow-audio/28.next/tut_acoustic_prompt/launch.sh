#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:
cd /data/zkx/zkx/iknow-audio/28.next/tut_acoustic_prompt
/home/star/anaconda3/envs/zkx/bin/python -u test_acoustic_prompt.py 2>&1 | tee tut_acoustic_prompt.log
