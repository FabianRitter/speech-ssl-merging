#!/bin/bash

source /opt/conda/bin/activate s3prl_old_cuda
BASE_DIR_S3PRL="/workspace/s3prl/s3prl"
cd $BASE_DIR_S3PRL
export PYTHONPATH=/workspace/s3prl:$PYTHONPATH

distilled_model_checkpoint=$1
task=$2
stage=$3
current_row=$4
upstream=$5
logfile=$6
logfile_row=$7
checkpoint_method=${8:-"hardcoded"}  # Default to "hardcoded" method
custom_checkpoint=${9:-""}  # Default to "hardcoded" method



use_paper_method=true  # Set to false if you don't want the paper method

if [ $task == "er" ]; then
  task="emotion"
fi

echo "The upstream model is: ${distilled_model_checkpoint}"
echo "CHECKPOINT_DIR is /workspace/s3prl/s3prl/result/pretrain/${distilled_model_checkpoint}"
echo "CUDA_VISIBLE_DEVICES inside the container: $CUDA_VISIBLE_DEVICES"
echo "nvidia-smi "
nvidia-smi

# Install necessary Python packages
pip install networkx pytorch-nlp transformers datasets==2.14.5 scipy==1.5.4 librosa==0.8.0 scikit-learn==0.24.2 matplotlib==3.3.4 modelscope==1.11.0 fvcore

# Configure Git and pull the latest changes if necessary
cd /workspace/s3prl
git config --global --add safe.directory /workspace/s3prl

# Set up model checkpoint
CHECKPOINT_DIR="/workspace/s3prl/s3prl/result/pretrain/${distilled_model_checkpoint}"
#latest_checkpoint=$(ls ${CHECKPOINT_DIR}/states-*.ckpt | sort -V | tail -n 1)
#latest_checkpoint="result/pretrain/$distilled_model_checkpoint/learning_by_addition.ckpt" #  learning_by_addition_2nd_approach.ckpt     learning_by_addition.ckpt
# Function to determine the latest checkpoint
# Function to determine the latest checkpoint
select_latest_checkpoint() {
  local method=$1
  local custom=$2
  case $method in
    "list_based")
      echo "$(ls ${CHECKPOINT_DIR}/states-*.ckpt | sort -V | tail -n 1)"
      ;;
    "hardcoded")
      echo "${CHECKPOINT_DIR}/learning_by_addition.ckpt"  # Default hardcoded checkpoint
      ;;
    "custom")
      if [ -n "$custom" ]; then
        echo "$custom"  # Use the provided custom checkpoint
      else
        echo "Custom checkpoint not provided. Exiting." >&2
        exit 1
      fi
      ;;
    *)
      echo "Unknown checkpoint selection method: $method" >&2
      exit 1
      ;;
  esac
}

latest_checkpoint=$(select_latest_checkpoint $checkpoint_method $custom_checkpoint)
echo "Loading the latest model: $latest_checkpoint"


echo "Loading the latest model: $latest_checkpoint"
exp_setup=${distilled_model_checkpoint}/${task}
json_file="${S3PRL_GSHEET_JSON:-}"  # optional Google-Sheets logging credential; leave empty to disable
if [ -n "$json_file" ]; then GSHEET="$GSHEET"; else GSHEET=""; fi

# Adjust exp_setup and paper argument based on the method
if [ "$use_paper_method" = true ]; then
  paper_arg="-s paper"
  exp_setup="${exp_setup}_paper_method"
else
  paper_arg=""
fi

echo "##### running evaluation #####"
echo "$stage $task with model $distilled_model_checkpoint"
# Evaluation (evaluate finetune result on test dataset)

downstream_path=/workspace/s3prl/s3prl/result/downstream
echo "CUDA_VISIBLE_DEVICES inside the container: $CUDA_VISIBLE_DEVICES"


if [ $task == "ic" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL

    if [ $stage == "train" ]; then
    echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/fluent_commands/config.yaml" --update_results --current_row_downstream $current_row $GSHEET \
      -u $upstream -k $latest_checkpoint $paper_arg -d fluent_commands -p ${downstream_path}/${exp_setup} --verbose --logfile $logfile --logfile_row_downstream $logfile_row
      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
          -d fluent_commands \
          -c "./downstream/fluent_commands/config.yaml"
          
    elif [ $stage == "resuming" ]; then 
      echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then

      echo "eval ic..."
      # Evaluation (evaluate finetune result on test dataset)
      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
          -d fluent_commands \
          -c "./downstream/fluent_commands/config.yaml"
    fi
fi

if [ $task == "ks" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL

    if [ $stage == "train" ]; then
    echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/speech_commands/config.yaml" -u $upstream -k $latest_checkpoint $paper_arg --update_results \
      --current_row_downstream $current_row -d speech_commands -p ${downstream_path}/${exp_setup} \
      $GSHEET --verbose --logfile $logfile --logfile_row_downstream $logfile_row \
      -o "config.downstream_expert.datarc.speech_commands_root=/livingrooms/public/superb/$task/speech_commands_v0.01,,\
          config.downstream_expert.datarc.speech_commands_test_root=/livingrooms/public/superb/$task/speech_commands_test_set_v0.01"
      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET \
          -d speech_commands --update_results --current_row_downstream $current_row \
          -c "./downstream/speech_commands/config.yaml"
          
    elif [ $stage == "resuming" ]; then 
      echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then
      # Evaluation (evaluate finetune result on test dataset)
      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
          -d speech_commands \
          -c "./downstream/speech_commands/config.yaml"
    fi
fi

if [ $task == "vocalset_singer_id" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL

    if [ $stage == "train" ]; then
    echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/vocalset_singer_id/config_singularity.yaml" -u $upstream -k $latest_checkpoint $paper_arg --update_results \
      --current_row_downstream $current_row $GSHEET -d vocalset_singer_id -p ${downstream_path}/${exp_setup} --verbose --logfile $logfile --logfile_row_downstream $logfile_row
      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
          -d vocalset_singer_id \
          -c "./downstream/vocalset_singer_id/config.yaml"
          
    elif [ $stage == "resuming" ]; then 
      echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then

      echo "eval vocal singer id..."
      # Evaluation (evaluate finetune result on test dataset)
      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
          -d vocalset_singer_id \
          -c "./downstream/vocalset_singer_id/config.yaml"
    fi
fi


if [ $task == "vocalset_technique_id" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL

    if [ $stage == "train" ]; then
    echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/vocalset_technique_id/config_singularity.yaml" -u $upstream -k $latest_checkpoint $paper_arg --update_results --current_row_downstream $current_row \
      --logfile $logfile --logfile_row_downstream $logfile_row $GSHEET -d $task -p ${downstream_path}/${exp_setup} --verbose
      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
          -d $task \
          -c "./downstream/$task/config_singularity.yaml"
          
    elif [ $stage == "resuming" ]; then 
      echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then

      # Evaluation (evaluate finetune result on test dataset)
      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
          -d $task \
          -c "./downstream/$task/config_singularity.yaml"
    fi
fi


if [ $task == "asr" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL
 

    if [ $stage == "train" ]; then
      echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/asr/config.yaml" -u $upstream -k $latest_checkpoint $paper_arg -d asr $GSHEET --current_row_downstream $current_row \
      -p ${downstream_path}/${exp_setup} --verbose --update_results --current_row_downstream $current_row  --logfile $logfile --logfile_row_downstream $logfile_row \
        -o "config.runner.gradient_accumulate_steps=1,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32,,\
        config.downstream_expert.datarc.bucket_file=./data/len_for_bucket"

      
      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-clean-best.ckpt \
          -k $latest_checkpoint $paper_arg $GSHEET \
          -u $upstream --update_results --current_row_downstream $current_row \
          -d asr -t "test-clean" \
          -c "./downstream/asr/config.yaml" \
          -o "config.runner.gradient_accumulate_steps=1,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32,,\
            config.downstream_expert.datarc.bucket_file=./data/len_for_bucket"



    elif [ $stage == "resuming" ]; then 
    echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-clean-best.ckpt \
          -k $latest_checkpoint $paper_arg $GSHEET \
          -u $upstream --update_results --current_row_downstream $current_row --current_row $current_row \
          -d asr -t "test-clean" \
          -c "./downstream/asr/config.yaml" \
          -o "config.runner.gradient_accumulate_steps=1,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32,,\
            config.downstream_expert.datarc.bucket_file=./data/len_for_bucket"


    fi

fi

if [ $task == "instrument_nsynth" ] || [ $task == "pitch_nsynth" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL

    if [ $stage == "train" ]; then
    echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/$task/config_singularity.yaml" -u $upstream -k $latest_checkpoint $paper_arg --update_results --current_row_downstream $current_row \
      --logfile $logfile --logfile_row_downstream $logfile_row $GSHEET -d $task -p ${downstream_path}/${exp_setup} --verbose
      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
          -d $task \
          -c "./downstream/$task/config_singularity.yaml"
          
    elif [ $stage == "resuming" ]; then 
      echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then

      # Evaluation (evaluate finetune result on test dataset)
      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
          -d $task \
          -c "./downstream/$task/config_singularity.yaml"
    fi
fi

if [ $task == "aec_esc50" ] || [ $task == "emotion" ]; then


 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL

    if [ $stage == "train" ]; then
    echo "$stage $task"

    for test_fold in fold1 fold2 fold3 fold4 fold5; do
    echo "running fold $test_fold"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/$task/config.yaml" -u $upstream -k $latest_checkpoint $paper_arg --update_results --current_row_downstream $current_row \
      --logfile $logfile --logfile_row_downstream $logfile_row $GSHEET -d $task -p ${downstream_path}/${exp_setup}_${test_fold} --verbose \
      -o "config.downstream_expert.datarc.test_fold=$test_fold"
      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}_${test_fold}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
          -d $task \
          -c "./downstream/$task/config.yaml" -o "config.downstream_expert.datarc.test_fold=$test_fold"
    done
          
    elif [ $stage == "resuming" ]; then 
      echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}_${test_fold}

    elif [ $stage == "evaluating" ]; then
      for test_fold in fold1 fold2 fold3 fold4 fold5; do

        # Evaluation (evaluate finetune result on test dataset)
        python run_downstream.py \
            -m evaluate --verbose \
            -e ${downstream_path}/${exp_setup}_${test_fold}/dev-best.ckpt \
            -k $latest_checkpoint $paper_arg \
            -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
            -d $task \
            -c "./downstream/$task/config.yaml" -o "config.downstream_expert.datarc.test_fold=$test_fold"
      done
    fi
fi



if [ $task == "sid" ]; then

 echo "running $task downstream"
 echo "running $model model"
 cd $BASE_DIR_S3PRL

    if [ $stage == "train" ]; then
    echo "$stage $task"
      # Training (finetune on downstream task) # weighted sum of enc hdden states.
      python run_downstream.py -m $stage -c "./downstream/voxceleb1/config.yaml" -u $upstream -k $latest_checkpoint $paper_arg --update_results \
      --current_row_downstream $current_row -d voxceleb1 -p ${downstream_path}/${exp_setup} \
      $GSHEET --verbose --logfile $logfile --logfile_row_downstream $logfile_row
      echo "experiment finished so we will run the evaluation."
      echo "experiment finished so we will run the evaluation."
      echo "\n \n \n \n \n."

      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET \
          -d voxceleb1 --update_results --current_row_downstream $current_row \
          -c "./downstream/voxceleb1/config.yaml"
          
    elif [ $stage == "resuming" ]; then 
      echo "$stage $task"
      # If training is interrupted, resume training
      python run_downstream.py -m train -e ${downstream_path}/${exp_setup}

    elif [ $stage == "evaluating" ]; then
      # Evaluation (evaluate finetune result on test dataset)
      python run_downstream.py \
          -m evaluate --verbose \
          -e ${downstream_path}/${exp_setup}/dev-best.ckpt \
          -k $latest_checkpoint $paper_arg \
          -u $upstream $GSHEET --update_results --current_row_downstream $current_row \
          -d voxceleb1 \
          -c "./downstream/voxceleb1/config.yaml"
    fi
fi


