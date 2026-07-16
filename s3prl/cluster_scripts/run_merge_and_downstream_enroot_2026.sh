#!/bin/bash
#PBS -P 12004380
#PBS -j oe
#PBS -N MERGE_DS
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -m ae
##PBS -M your_email@example.com  # uncomment to receive job mail
#PBS -o /data/projects/12004380/fabian/task-arithmetic-speech-audio/s3prl/logfiles/merge_and_downstream_2026.out
#PBS -l select=1:ncpus=64:ngpus=1:mem=470gb:container_engine=enroot
#PBS -l container_image=~/images/fabianritterg_torch2_nocuda.sqsh
#PBS -l container_name=fabianritterg_torch2_nocuda
#PBS -l enroot_env_file=~/sample_jobs/container_env.conf

# ============================================================================
# Step 1: Merge two models at 4 weight combos (0.9/0.1, 0.8/0.2, 0.5/0.5, 0.1/0.9)
# Step 2: Run downstream tasks (singid, instcls, genreid) for each combo
#
# Models:
#   MUSIC:  distill_only_mert-...-960/states-220000.ckpt  (multi_distiller)
#   SPEECH: distilhubert-ls960-own/states-220000.ckpt      (distiller)
#
# Usage:
#   qsub run_merge_and_downstream_enroot_2026.sh
#   qsub -v current_row="70",logfile_row="310" run_merge_and_downstream_enroot_2026.sh
# ============================================================================

PROJECT_ROOT="/data/projects/12004380/fabian/task-arithmetic-speech-audio/s3prl"
CONTAINER_NAME="s3prl_downstream_2026"
CONTAINER_IMAGE="/data/projects/12004380/fabian/containers/new_s3prl_torch2/fabianritterg_torch2_nocuda.sqsh"

GTZAN_DATA="/data/projects/12004380/datasets/superb/superb/GTZAN"
NSYNTH_DATA="/data/projects/12004380/datasets/superb/superb/NSynth"

PROXY="http://10.104.4.124:10104"
NO_PROXY="localhost,127.0.0.1,10.104.0.0/21"

# ----- Parameters -----
current_row=${current_row:-"70"}
logfile_row=${logfile_row:-"310"}
upstream="multi_distiller_local"
feature_selection="paper"

MUSIC_MODEL="distill_only_mert-init-weight-from-hubert_base-models-simple-avg-pool-for-teacher-train-libri-960"
SPEECH_MODEL="distilhubert-ls960-own"
MUSIC_CKPT="states-220000.ckpt"
SPEECH_CKPT="states-220000.ckpt"

# Weight combinations: speech_lambda,music_lambda
WEIGHT_COMBOS="0.9,0.1 0.8,0.2 0.5,0.5 0.1,0.9"
MERGED_CKPT="learning_by_addition.ckpt"

JSON_FILE="${S3PRL_GSHEET_JSON:-}"  # optional Google-Sheets logging credential; leave empty to disable

# ----- GSheet args -----
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

PAPER_ARG="-s ${feature_selection}"

# ----- PBS or interactive -----
if [ -n "$PBS_O_WORKDIR" ]; then
    source ~/.bashrc
    cd "$PBS_O_WORKDIR"
    echo "Running via PBS in: $PBS_O_WORKDIR"
else
    cd "$PROJECT_ROOT"
    echo "Running interactively in: $PROJECT_ROOT"
fi

echo "============================================"
echo "Merge + Downstream Script"
echo "Music model:  $MUSIC_MODEL ($MUSIC_CKPT)"
echo "Speech model: $SPEECH_MODEL ($SPEECH_CKPT)"
echo "Weight combos: $WEIGHT_COMBOS"
echo "Current row:  $current_row"
echo "Logfile row:  $logfile_row"
echo "============================================"

log_dir="${PROJECT_ROOT}/logfiles/downstream/merge_and_downstream_all_combos"
mkdir -p "$log_dir"

# ----- Container setup -----
echo "Checking enroot container..."
if ! enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "Creating container '${CONTAINER_NAME}' from ${CONTAINER_IMAGE}..."
    enroot create --name "$CONTAINER_NAME" "$CONTAINER_IMAGE"
else
    echo "Container '${CONTAINER_NAME}' already exists."
fi

PIP_INSTALL="pip install networkx pytorch-nlp transformers datasets==2.14.5 scipy==1.5.4 librosa==0.8.0 scikit-learn==0.24.2 matplotlib==3.3.4 modelscope==1.11.0 fvcore addict 2>&1 | tail -5"

COMMON_MOUNTS="\
    --mount /app/apps/cuda/12.2.2:/usr/local/cuda \
    --mount /usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu/nvidia \
    --mount ${PROJECT_ROOT}:/workspace/s3prl/s3prl \
    --mount ${GTZAN_DATA}:/workspace/audio_data/GTZAN \
    --mount ${NSYNTH_DATA}:/workspace/audio_data/NSynth"

COMMON_ENV="\
    --env http_proxy=${PROXY} \
    --env https_proxy=${PROXY} \
    --env HTTP_PROXY=${PROXY} \
    --env HTTPS_PROXY=${PROXY} \
    --env no_proxy=${NO_PROXY} \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility"

DOWNSTREAM_PATH="/workspace/s3prl/s3prl/result/downstream"
PRETRAIN_PATH="/workspace/s3prl/s3prl/result/pretrain"

gpu_log="${log_dir}/merge_and_downstream.log"

echo "Starting merge + downstream pipeline..."
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

# ================================================================
# STEP 1: Merge models at 4 weight combinations
# ================================================================
echo '========================================'
echo '=== STEP 1: Merging models ==='
echo '========================================'

MUSIC_CKPT_PATH='result/pretrain/${MUSIC_MODEL}/${MUSIC_CKPT}'
SPEECH_CKPT_PATH='result/pretrain/${SPEECH_MODEL}/${SPEECH_CKPT}'

for COMBO in 0.9,0.1 0.8,0.2 0.5,0.5 0.1,0.9; do
    L1=\${COMBO%%,*}
    L2=\${COMBO##*,}
    SAVE_DIR=\"result/pretrain/task_vector_dhubert_ls_960_weight_\${L1}_and_mert_ls_960_weight_\${L2}_both_init_hubert_both_same_seed\"
    SAVE_FILE=\"\${SAVE_DIR}/learning_by_addition.ckpt\"

    echo \"=== Merging with lambda1=\${L1} (speech) lambda2=\${L2} (music) ===\"
    python merge_two_models.py \
            --music_ckpt \"\${MUSIC_CKPT_PATH}\" \
            --speech_ckpt \"\${SPEECH_CKPT_PATH}\" \
            --save_path \"\${SAVE_FILE}\" \
            --lambda1 \${L1} \
            --lambda2 \${L2}
    echo \"=== Merge \${L1}/\${L2} done ===\"
    echo ''
done

# ================================================================
# STEP 2: Downstream tasks for ALL weight combinations
# ================================================================
for COMBO in 0.9,0.1 0.8,0.2 0.5,0.5 0.1,0.9; do
    L1=\${COMBO%%,*}
    L2=\${COMBO##*,}
    CUR_MODEL=\"task_vector_dhubert_ls_960_weight_\${L1}_and_mert_ls_960_weight_\${L2}_both_init_hubert_both_same_seed\"
    CUR_CKPT_ARG=\"-k ${PRETRAIN_PATH}/\${CUR_MODEL}/${MERGED_CKPT}\"

    echo '========================================'
    echo \"=== STEP 2: Downstream for \${L1}/\${L2} ===\"
    echo '========================================'

    mkdir -p logfiles/downstream/\${CUR_MODEL}

    # --- vocalset_singer_id ---
    echo ''
    echo \"=== vocalset_singer_id Training (\${L1}/\${L2}) ===\"
    singer_exp=\"${DOWNSTREAM_PATH}/\${CUR_MODEL}/vocalset_singer_id_paper_method\"
    python run_downstream.py -m train \
        -c './downstream/vocalset_singer_id/config_nscc.yaml' \
        -u ${upstream} \${CUR_CKPT_ARG} ${PAPER_ARG} \
        -d vocalset_singer_id \
        -p \${singer_exp} \
        --verbose \
        --logfile logfiles/downstream/\${CUR_MODEL}/singid \
        ${GSHEET_ARGS}

    if [ \$? -eq 0 ]; then
        echo \"=== vocalset_singer_id Evaluation (\${L1}/\${L2}) ===\"
        python run_downstream.py -m evaluate --verbose \
            -e \${singer_exp}/dev-best.ckpt \
            \${CUR_CKPT_ARG} ${PAPER_ARG} \
            -u ${upstream} \
            -d vocalset_singer_id \
            -c './downstream/vocalset_singer_id/config_nscc.yaml' \
            ${GSHEET_ARGS}
    fi

    # --- instrument_nsynth ---
    echo ''
    echo \"=== instrument_nsynth Training (\${L1}/\${L2}) ===\"
    inst_exp=\"${DOWNSTREAM_PATH}/\${CUR_MODEL}/instrument_nsynth_paper_method\"
    python run_downstream.py -m train \
        -c './downstream/instrument_nsynth/config_nscc.yaml' \
        -u ${upstream} \${CUR_CKPT_ARG} ${PAPER_ARG} \
        -d instrument_nsynth \
        -p \${inst_exp} \
        --verbose \
        --logfile logfiles/downstream/\${CUR_MODEL}/instcls \
        ${GSHEET_ARGS}

    if [ \$? -eq 0 ]; then
        echo \"=== instrument_nsynth Evaluation (\${L1}/\${L2}) ===\"
        python run_downstream.py -m evaluate --verbose \
            -e \${inst_exp}/dev-best.ckpt \
            \${CUR_CKPT_ARG} ${PAPER_ARG} \
            -u ${upstream} \
            -d instrument_nsynth \
            -c './downstream/instrument_nsynth/config_nscc.yaml' \
            ${GSHEET_ARGS}
    fi

    # --- genre_gtzan ---
    echo ''
    echo \"=== genre_gtzan Training (\${L1}/\${L2}) ===\"
    genre_exp=\"${DOWNSTREAM_PATH}/\${CUR_MODEL}/genre_gtzan_paper_method\"
    python run_downstream.py -m train \
        -c './downstream/genre_gtzan/config_nscc.yaml' \
        -u ${upstream} \${CUR_CKPT_ARG} ${PAPER_ARG} \
        -d genre_gtzan \
        -p \${genre_exp} \
        --verbose \
        --logfile logfiles/downstream/\${CUR_MODEL}/genreid \
        ${GSHEET_ARGS}

    if [ \$? -eq 0 ]; then
        echo \"=== genre_gtzan Evaluation (\${L1}/\${L2}) ===\"
        python run_downstream.py -m evaluate --verbose \
            -e \${genre_exp}/valid-best.ckpt \
            \${CUR_CKPT_ARG} ${PAPER_ARG} \
            -u ${upstream} \
            -d genre_gtzan \
            -c './downstream/genre_gtzan/config_nscc.yaml' \
            ${GSHEET_ARGS}
    fi

    echo ''
    echo \"=== All tasks done for \${L1}/\${L2} ===\"
    echo ''
done

echo ''
echo '=== All weight combinations Done ==='
" 2>&1 | tee "$gpu_log"

final_exit=$?
echo ""
echo "============================================"
echo "Merge + Downstream completed with exit code: $final_exit"
echo "Log: $gpu_log"
echo "============================================"
exit $final_exit
