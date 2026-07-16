#!/bin/bash
#PBS -P 12004380
#PBS -j oe
#PBS -N DOWNSTREAM_2026
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -m ae
##PBS -M your_email@example.com  # uncomment to receive job mail
#PBS -o /data/projects/12004380/fabian/task-arithmetic-speech-audio/s3prl/logfiles/downstream_2026.out
#PBS -l select=1:ncpus=64:ngpus=2:mem=470gb:container_engine=enroot
#PBS -l container_image=~/images/fabianritterg_torch2_nocuda.sqsh
#PBS -l container_name=fabianritterg_torch2_nocuda
#PBS -l enroot_env_file=~/sample_jobs/container_env.conf

# ============================================================================
# Run downstream tasks (ASR, vocalset_singer_id, genre_gtzan) with 2 GPUs
# ============================================================================
#
# GPU layout:
#   GPU 0 -> ASR then vocalset_singer_id (sequential, both heavy/medium)
#   GPU 1 -> genre_gtzan (light, runs alone)
#
# Usage (distilled model, default upstream=distiller_local, feature_selection=paper):
#   qsub -v model="barlow_no_distort_960_crosscorr_5e-3_selfcorr_5e-2",\
#   current_row="179",logfile_row="212" \
#     run_downstream_enroot_2026.sh
#
# Usage (built-in upstream, no checkpoint needed):
#   qsub -v upstream="hubert_base_robust_mgwham_rbp",\
#   feature_selection="hidden_states",current_row="182",logfile_row="215" \
#     run_downstream_enroot_2026.sh
#
# ============================================================================

# ----- Project paths -----
PROJECT_ROOT="/data/projects/12004380/fabian/task-arithmetic-speech-audio/s3prl"
CONTAINER_IMAGE="/data/projects/12004380/fabian/containers/new_s3prl_torch2/fabianritterg_torch2_nocuda.sqsh"
CONTAINER_NAME="s3prl_downstream_2026"

# ----- Data paths (host) -----
LIBRISPEECH_DATA="/data/projects/12004380/datasets/superb/superb/Librispeech/LibriSpeech"
GTZAN_DATA="/data/projects/12004380/datasets/superb/superb/GTZAN"

# ----- Proxy for internet access inside container -----
PROXY="http://10.104.4.124:10104"
NO_PROXY="localhost,127.0.0.1,10.104.0.0/21"

# ----- Parameters from environment (qsub -v) or defaults -----
model=${model:-""}
stage=${stage:-"train"}
upstream=${upstream:-"distiller_local"}
feature_selection=${feature_selection:-"paper"}
current_row=${current_row:-""}
logfile_row=${logfile_row:-""}

# If logfile_row not set, default to current_row
if [ -n "$current_row" ] && [ -z "$logfile_row" ]; then
    logfile_row="$current_row"
fi

# Checkpoint and experiment naming depend on whether we use a distilled (local) upstream
# or a built-in upstream (e.g. hubert_base_robust_mgwham_rbp)
if [[ "$upstream" == *_local* ]]; then
    # Distilled model: requires model name and checkpoint
    if [ -z "$model" ]; then
        echo "Error: 'model' is required for upstream '$upstream'"
        exit 1
    fi
    CKPT="/workspace/s3prl/s3prl/result/pretrain/${model}/dev-dis-best.ckpt"
    CKPT_ARG="-k ${CKPT}"
    EXP_NAME="${model}"
else
    # Built-in upstream (e.g. hubert_base_robust_mgwham_rbp): no checkpoint needed
    CKPT=""
    CKPT_ARG=""
    EXP_NAME="${upstream}"
fi

# JSON file for Google Sheets results (inside container)
JSON_FILE="${S3PRL_GSHEET_JSON:-}"  # optional Google-Sheets logging credential; leave empty to disable

# ----- Build Google Sheets / logging args -----
GSHEET_ARGS=""
if [ -n "$current_row" ]; then
    GSHEET_ARGS="--update_results --current_row_downstream $current_row --current_row $current_row"
fi
if [ -n "$logfile_row" ]; then
    GSHEET_ARGS="$GSHEET_ARGS --logfile_row_downstream $logfile_row"
fi
if [ -n "$JSON_FILE" ]; then
    GSHEET_ARGS="$GSHEET_ARGS --json_file $JSON_FILE"
fi

# ----- Feature selection flag -----
PAPER_ARG="-s ${feature_selection}"

# ----- PBS or interactive mode -----
if [ -n "$PBS_O_WORKDIR" ]; then
    source ~/.bashrc
    cd "$PBS_O_WORKDIR"
    echo "Running via PBS in: $PBS_O_WORKDIR"
else
    cd "$PROJECT_ROOT"
    echo "Running interactively in: $PROJECT_ROOT"
fi

echo "============================================"
echo "Upstream:     $upstream"
echo "Feature sel:  $feature_selection"
echo "Model:        ${model:-'(not set, built-in upstream)'}"
echo "Stage:        $stage"
echo "Checkpoint:   ${CKPT:-'(none, built-in upstream)'}"
echo "Exp name:     $EXP_NAME"
echo "Current row:  ${current_row:-'(not set)'}"
echo "Logfile row:  ${logfile_row:-'(not set)'}"
echo "GSheet args:  $GSHEET_ARGS"
echo "============================================"

# ----- Log directory -----
log_dir="${PROJECT_ROOT}/logfiles/downstream/${EXP_NAME}"
mkdir -p "$log_dir"
echo "Logs: $log_dir"

# ----- Container setup -----
echo "Checking enroot container..."
if ! enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "Creating container '${CONTAINER_NAME}' from ${CONTAINER_IMAGE}..."
    enroot create --name "$CONTAINER_NAME" "$CONTAINER_IMAGE"
else
    echo "Container '${CONTAINER_NAME}' already exists."
fi

# ----- Pip install command (run once per container start) -----
PIP_INSTALL="pip install networkx pytorch-nlp transformers datasets==2.14.5 scipy==1.5.4 librosa==0.8.0 scikit-learn==0.24.2 matplotlib==3.3.4 modelscope==1.11.0 fvcore addict 2>&1 | tail -5"

# ----- Common container mounts -----
COMMON_MOUNTS="\
    --mount /app/apps/cuda/12.2.2:/usr/local/cuda \
    --mount /usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu/nvidia \
    --mount ${PROJECT_ROOT}:/workspace/s3prl/s3prl \
    --mount ${LIBRISPEECH_DATA}:/workspace/audio_data/LibriSpeech \
    --mount ${GTZAN_DATA}:/workspace/audio_data/GTZAN"

# ----- Common env vars -----
COMMON_ENV="\
    --env http_proxy=${PROXY} \
    --env https_proxy=${PROXY} \
    --env HTTP_PROXY=${PROXY} \
    --env HTTPS_PROXY=${PROXY} \
    --env no_proxy=${NO_PROXY} \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility"

# ----- Downstream output path (inside container) -----
DOWNSTREAM_PATH="/workspace/s3prl/s3prl/result/downstream"

# ============================================================================
# GPU 0: ASR and vocalset_singer_id in PARALLEL (sharing GPU 0)
# ============================================================================
asr_exp="${DOWNSTREAM_PATH}/${EXP_NAME}/asr_paper_method"
singer_exp="${DOWNSTREAM_PATH}/${EXP_NAME}/vocalset_singer_id_paper_method"
asr_logfile="logfiles/downstream/${EXP_NAME}/asr"
singer_logfile="logfiles/downstream/${EXP_NAME}/singid"
asr_log="${log_dir}/asr_${stage}.log"
singer_log="${log_dir}/singer_${stage}.log"

# ---- ASR on GPU 0 (background) ----
echo "Starting ASR on GPU 0..."
enroot start \
    ${COMMON_MOUNTS} \
    ${COMMON_ENV} \
    --env CUDA_VISIBLE_DEVICES=0 \
    ${CONTAINER_NAME} \
    bash -c "
source /opt/conda/bin/activate s3prl_old_cuda
cd /workspace/s3prl/s3prl
export PYTHONPATH=/workspace/s3prl:\$PYTHONPATH

echo '=== Installing packages ==='
${PIP_INSTALL}

echo '=== Environment ==='
python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")'
echo ''

echo '========================================'
echo '=== ASR Training on GPU 0 ==='
echo '========================================'
python run_downstream.py -m train \
    -c './downstream/asr/config.yaml' \
    -u ${upstream} ${CKPT_ARG} ${PAPER_ARG} \
    -d asr \
    -p ${asr_exp} \
    --verbose \
    --early_stopping_patience 35 \
    --logfile ${asr_logfile} \
    ${GSHEET_ARGS} \
    -o 'config.runner.gradient_accumulate_steps=1,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32,,config.downstream_expert.datarc.bucket_file=./data/len_for_bucket,,config.downstream_expert.datarc.libri_root=/workspace/audio_data/LibriSpeech'

if [ \$? -eq 0 ]; then
    echo ''
    echo '=== ASR Evaluation ==='
    python run_downstream.py -m evaluate --verbose \
        -e ${asr_exp}/dev-clean-best.ckpt \
        ${CKPT_ARG} ${PAPER_ARG} \
        -u ${upstream} \
        -d asr -t 'test-clean' \
        -c './downstream/asr/config.yaml' \
        ${GSHEET_ARGS} \
        -o 'config.runner.gradient_accumulate_steps=1,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32,,config.downstream_expert.datarc.bucket_file=./data/len_for_bucket,,config.downstream_expert.datarc.libri_root=/workspace/audio_data/LibriSpeech'
fi

echo ''
echo '=== ASR Done ==='
" > "$asr_log" 2>&1 &
ASR_PID=$!
echo "ASR PID: $ASR_PID, log: $asr_log"

# ---- vocalset_singer_id on GPU 0 (background, parallel with ASR) ----
# Wait for ASR's pip install to finish before starting singer_id's container
echo "Waiting 120s for ASR pip install to complete before launching singer_id..."
sleep 120
echo "Starting vocalset_singer_id on GPU 0 (parallel with ASR)..."
enroot start \
    ${COMMON_MOUNTS} \
    ${COMMON_ENV} \
    --env CUDA_VISIBLE_DEVICES=0 \
    ${CONTAINER_NAME} \
    bash -c "
source /opt/conda/bin/activate s3prl_old_cuda
cd /workspace/s3prl/s3prl
export PYTHONPATH=/workspace/s3prl:\$PYTHONPATH

echo '=== Installing packages ==='
${PIP_INSTALL}

echo '=== Environment ==='
python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")'
echo ''

echo '========================================'
echo '=== vocalset_singer_id Training on GPU 0 ==='
echo '========================================'
python run_downstream.py -m train \
    -c './downstream/vocalset_singer_id/config_nscc.yaml' \
    -u ${upstream} ${CKPT_ARG} ${PAPER_ARG} \
    -d vocalset_singer_id \
    -p ${singer_exp} \
    --verbose \
    --logfile ${singer_logfile} \
    ${GSHEET_ARGS}

if [ \$? -eq 0 ]; then
    echo ''
    echo '=== vocalset_singer_id Evaluation ==='
    python run_downstream.py -m evaluate --verbose \
        -e ${singer_exp}/dev-best.ckpt \
        ${CKPT_ARG} ${PAPER_ARG} \
        -u ${upstream} \
        -d vocalset_singer_id \
        -c './downstream/vocalset_singer_id/config_nscc.yaml' \
        ${GSHEET_ARGS}
fi

echo ''
echo '=== vocalset_singer_id Done ==='
" > "$singer_log" 2>&1 &
SINGER_PID=$!
echo "Singer PID: $SINGER_PID, log: $singer_log"

# ============================================================================
# GPU 1: genre_gtzan (background)
# ============================================================================
echo "Starting genre_gtzan on GPU 1..."
gpu1_log="${log_dir}/gpu1_genre_${stage}.log"
genre_exp="${DOWNSTREAM_PATH}/${EXP_NAME}/genre_gtzan_paper_method"
genre_logfile="logfiles/downstream/${EXP_NAME}/genreid"

enroot start \
    ${COMMON_MOUNTS} \
    ${COMMON_ENV} \
    --env CUDA_VISIBLE_DEVICES=1 \
    ${CONTAINER_NAME} \
    bash -c "
source /opt/conda/bin/activate s3prl_old_cuda
cd /workspace/s3prl/s3prl
export PYTHONPATH=/workspace/s3prl:\$PYTHONPATH

echo '=== Installing packages ==='
${PIP_INSTALL}

echo '=== Environment ==='
python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")'
echo ''

# ---- genre_gtzan ----
echo '========================================'
echo '=== genre_gtzan Training on GPU 1 ==='
echo '========================================'
python run_downstream.py -m train \
    -c './downstream/genre_gtzan/config_nscc.yaml' \
    -u ${upstream} ${CKPT_ARG} ${PAPER_ARG} \
    -d genre_gtzan \
    -p ${genre_exp} \
    --verbose \
    --logfile ${genre_logfile} \
    ${GSHEET_ARGS}

if [ \$? -eq 0 ]; then
    echo ''
    echo '=== genre_gtzan Evaluation ==='
    python run_downstream.py -m evaluate --verbose \
        -e ${genre_exp}/valid-best.ckpt \
        ${CKPT_ARG} ${PAPER_ARG} \
        -u ${upstream} \
        -d genre_gtzan \
        -c './downstream/genre_gtzan/config_nscc.yaml' \
        ${GSHEET_ARGS}
fi

echo ''
echo '=== GPU 1 All Done ==='
" > "$gpu1_log" 2>&1 &
GPU1_PID=$!
echo "GPU1 PID: $GPU1_PID, log: $gpu1_log"

# ============================================================================
# Wait for all 3 tasks to finish
# ============================================================================
echo ""
echo "Waiting for all tasks to complete..."
echo "  GPU 0 - ASR:          PID $ASR_PID     log: $asr_log"
echo "  GPU 0 - singer_id:    PID $SINGER_PID  log: $singer_log"
echo "  GPU 1 - genre_gtzan:  PID $GPU1_PID    log: $gpu1_log"
echo ""

wait $ASR_PID
asr_exit=$?
echo "ASR finished with exit code: $asr_exit"

wait $SINGER_PID
singer_exit=$?
echo "vocalset_singer_id finished with exit code: $singer_exit"

wait $GPU1_PID
gpu1_exit=$?
echo "genre_gtzan finished with exit code: $gpu1_exit"

echo ""
echo "============================================"
echo "All tasks completed for: $EXP_NAME"
echo "  ASR log:        $asr_log"
echo "  Singer log:     $singer_log"
echo "  Genre log:      $gpu1_log"
echo "============================================"
exit 0
