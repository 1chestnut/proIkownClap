#!/usr/bin/env bash
set -e
cd /data/zkx/zkx/iknow-audio/30.repulsive_rest/usk80
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:4LD_LIBRARY_PATH
/home/star/anaconda3/envs/zkx/bin/python -u usk29_repulsive.py 2>&1 | tee usk80_repulsive.log
