#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib
echo "START GPU2 $(date)" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu2.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/audioset $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu2.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/audioset
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/audioset $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu2.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/usk80 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu2.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/usk80
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/usk80 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu2.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/usk80 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu2.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/usk80
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/usk80 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu2.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/esc50 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu2.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/esc50
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/esc50 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu2.log
echo "DONE GPU2 $(date)" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu2.log
