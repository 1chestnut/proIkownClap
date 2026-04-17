export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib:$LD_LIBRARY_PATH
cd /data/zkx/zkx/iknow-audio/18diag/tut2017_alpha_topk6_v3
/home/star/anaconda3/envs/zkx/bin/python -u run_tut2017_alpha_topk6_v3.py | tee tut2017_alpha_topk6_v3.log
