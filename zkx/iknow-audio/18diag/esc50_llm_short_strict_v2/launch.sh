export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:$LD_LIBRARY_PATH
cd /data/zkx/zkx/iknow-audio/18diag/esc50_llm_short_strict_v2
/home/star/anaconda3/envs/zkx/bin/python -u run_esc50_llm_short_strict_v2.py | tee esc50_llm_short_strict_v2.log
