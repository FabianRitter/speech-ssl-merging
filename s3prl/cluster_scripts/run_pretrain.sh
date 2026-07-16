#!/bin/bash
##### BASELINE EXPERIMENT ######

source /opt/conda/bin/activate s3prl_old_cuda
BASE_DIR_S3PRL="/workspace/s3prl/s3prl"
cd $BASE_DIR_S3PRL
pip install nnAudio

# to do every time I start the image:
#cd /workspace
# git clone https://github.com/huggingface/transformers.git
# cd transformers
# pip install -e .
#cd $BASE_DIR_S3PRL
gpus=1
model=distilhubert_music4all_and_ls960_2layers
logfile=logfiles/pretrain/$model
current_row=53
export MASTER_PORT=$MASTER_PORT
config_file="pretrain/multi_distiller/config_model.yaml" #pretrain/multi_distiller/config_model_single_teacher.yaml
upstream="multi_distiller" #distill_only_ssast-legit-ssast-init-weight-from-hubert-simple-avg-pool-for-teacher-train-libri-960
export CUDA_VISIBLE_DEVICES=0
# altenratively do on singularity:  nohup python run_pretrain.py -u $upstream -g $config_file -n $model --logfile $logfile --current_row $current_row > $logfile 2>&1 &
nohup python run_pretrain.py -u $upstream -g $config_file -n $model --logfile $logfile --current_row $current_row > $logfile 2>&1 &


# altenratively do on singularity:  nohup python run_pretrain.py -u $upstream -g $config_file -n $model --logfile $logfile --current_row $current_row > $logfile 2>&1 &
