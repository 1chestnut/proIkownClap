#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib
echo "START GPU0 $(date)" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0408_escfix/esc50 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0408_escfix/esc50
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0408_escfix/esc50 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/fsd50 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/fsd50
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/fsd50 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/fsd50 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/fsd50
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/fsd50 $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/dcase $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/dcase
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_035075/dcase $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
echo "===== START /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/dcase $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
cd /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/dcase
./launch.sh
echo "===== END /data/zkx/zkx/iknow-audio/20.alpha_sweep/alpha_0307/dcase $(date) =====" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
echo "DONE GPU0 $(date)" | tee -a /data/zkx/zkx/iknow-audio/20.alpha_sweep/batch_gpu0.log
