#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib
echo "START GPU1 $(date)" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu1.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/audioset $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu1.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/audioset
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/audioset $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu1.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/tut2017 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu1.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/tut2017
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/tut2017 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu1.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/tut2017 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu1.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/tut2017
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/tut2017 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu1.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/esc50 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu1.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/esc50
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/esc50 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu1.log
echo "DONE GPU1 $(date)" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu1.log
