#!/bin/bash

# Usage example:
# sbatch run_downstream_sbatch.sh <distilled_model_checkpoint> <task> <stage> <current_row> <logfile_row>
# sbatch run_downstream_sbatch.sh distill_mert_init_mert_music4all_avgpool pitch_nsynth train 37 13

# sbatch run_downstream_sbatch.sh $model asr
#SBATCH --job-name=evaluate
#SBATCH -p 3d
#SBATCH --nodelist=s03
#SBATCH -n 1
#SBATCH --gres=gpu:RTX4000SFFAdaGeneration:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=28G
#SBATCH --time=0-34:44:00

# Set variables from input arguments or use default values
distilled_model_checkpoint=${1:-"default_checkpoint"}
task=${2:-"ic"}                # Default to "ic" if no task is provided
stage=${3:-"train"}       # Default to "evaluating" if no stage is provided
current_row=${4:-90}           # Default to 90 if no current_row is provided
logfile_row=${5:-90}

log_file="logfiles/downstream/${distilled_model_checkpoint}/${task}/paper_method.log"
upstream="multi_distiller_local" #  distiller_local multi_distiller_local

#tasks :   instrument_nsynth    pitch_nsynth      aec_esc50
# Create the log directory if it doesn't exist
mkdir -p "$(dirname "$log_file")"
echo "log at: $log_file"
# Set the output log file for SLURM
#SBATCH --output="$log_file"
export CUDA_VISIBLE_DEVICES=0
# Run the Singularity container and execute the commands
srun -k --output="$log_file" singularity exec --nv --bind /livingrooms/fabian/projects/distillation-sfm/s3prl/s3prl:/workspace/s3prl/s3prl,\
/groups/ycevan/datasets:/workspace/audio_data,\
/groups/public/benchmark/LibriSpeech/:/workspace/LibriSpeech,\
/livingrooms/fabian/music4all:/workspace/music4all,\
/livingrooms/fabian/AudioSet:/workspace/AudioSet,\
/groups/public/benchmark:/workspace/superb_data1,\
/livingrooms/public/LibriSpeech:/livingrooms/public/LibriSpeech,\
/livingrooms/public/superb:/livingrooms/public/superb ../../s3prl_for_sslm_v2.sif \
/workspace/s3prl/s3prl/run_inside_container_downstream.sh "$distilled_model_checkpoint" "$task" "$stage" "$current_row" "$upstream" "$log_file" "$logfile_row"