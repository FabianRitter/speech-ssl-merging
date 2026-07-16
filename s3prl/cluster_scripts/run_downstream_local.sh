#!/bin/bash
# =============================================================================
# run_downstream_local.sh
# Simplified local downstream evaluation script (no SLURM/PBS/Singularity).
# Dispatches multiple downstream tasks in parallel across specified GPUs.
#
# Usage:
#   bash run_downstream_local.sh \
#     --checkpoint /path/to/states-60000.ckpt \
#     --gpus 3,4 \
#     --tasks ic,ks \
#     --upstream multi_distiller_local \
#     --stage train \
#     [--current_row 90] [--logfile_row 90]
#
# Arguments:
#   --checkpoint   Full path to upstream model checkpoint (required)
#   --gpus         Comma-separated GPU IDs, e.g. 3,4 (required)
#   --tasks        Comma-separated task aliases, e.g. ic,ks,asr (required)
#   --upstream     Upstream model name, e.g. multi_distiller_local (required)
#   --stage        One of: train, evaluating (default: train)
#   --current_row  Google Sheet row for writing evaluation metrics/accuracy
#   --logfile_row  Google Sheet row for writing log file paths
#                  (defaults to --current_row if not set)
#
# Task aliases (11 tasks covering SUPERB, MARBLE, and audio benchmarks):
#   --- SUPERB (Speech) ---
#   asr             -> asr (with batch/bucket overrides)      [WER% down]
#   ks              -> speech_commands                         [Acc% up]
#   ic              -> fluent_commands                         [Acc% up]
#   sid             -> voxceleb1                               [Acc% up]
#   er              -> emotion (5-fold cross-validation)       [Acc% up]
#   --- MARBLE (Music) ---
#   singid          -> vocalset_singer_id                      [Acc% up]
#   vocid           -> vocalset_technique_id                   [Acc% up]
#   instcls         -> instrument_nsynth                       [Acc% up]
#   pitchid         -> pitch_nsynth                            [Acc% up]
#   genreid         -> genre_gtzan (GTZAN genre classification) [Acc% up]
#   --- Audio ---
#   aec_esc50       -> aec_esc50 (5-fold cross-validation)     [Acc% up]
#   <other>         -> uses alias as downstream name with downstream/{alias}/config.yaml
# =============================================================================

set -euo pipefail

# ----------------------------- Parse arguments --------------------------------

CHECKPOINT=""
GPUS=""
TASKS=""
UPSTREAM=""
STAGE="train"
CURRENT_ROW=""
LOGFILE_ROW=""
FEATURE_SELECTION="paper"
JSON_FILE="${S3PRL_GSHEET_JSON:-}"  # optional Google-Sheets logging credential; leave empty to disable
SHEET_NAME=""
WORKSHEET_NAME=""
WORKSHEET_LOGFILE_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --gpus)       GPUS="$2";       shift 2 ;;
    --tasks)      TASKS="$2";      shift 2 ;;
    --upstream)   UPSTREAM="$2";   shift 2 ;;
    --stage)      STAGE="$2";      shift 2 ;;
    --current_row) CURRENT_ROW="$2"; shift 2 ;;
    --logfile_row) LOGFILE_ROW="$2"; shift 2 ;;
    --feature_selection) FEATURE_SELECTION="$2"; shift 2 ;;
    --sheet_name) SHEET_NAME="$2"; shift 2 ;;
    --worksheet_name) WORKSHEET_NAME="$2"; shift 2 ;;
    --worksheet_logfile_name) WORKSHEET_LOGFILE_NAME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Validate required arguments
if [[ -z "$GPUS" ]];       then echo "Error: --gpus is required";       exit 1; fi
if [[ -z "$TASKS" ]];      then echo "Error: --tasks is required";      exit 1; fi
if [[ -z "$UPSTREAM" ]];   then echo "Error: --upstream is required";   exit 1; fi
# Checkpoint is required for local/distilled upstreams
if [[ "$UPSTREAM" == *_local* && -z "$CHECKPOINT" ]]; then
  echo "Error: --checkpoint is required for upstream '$UPSTREAM'"; exit 1
fi

# If logfile_row not set, default to current_row
if [[ -n "$CURRENT_ROW" && -z "$LOGFILE_ROW" ]]; then
  LOGFILE_ROW="$CURRENT_ROW"
fi

# ----------------------------- Environment ------------------------------------

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

set +u
source /export/home2/fabian/miniconda3/bin/activate s3prl_multidistiller
set -u
cd /export/home2/fabian/projects/multi_distiller/s3prl/s3prl

# Make only the requested GPUs visible to all child processes
export CUDA_VISIBLE_DEVICES="$GPUS"

DOWNSTREAM_PATH="${BASE_DIR}/result/downstream-thesis-pending-exps"
# Derive experiment name from checkpoint path or upstream name
if [[ -n "$CHECKPOINT" ]]; then
  CHECKPOINT_NAME="$(basename "$(dirname "$CHECKPOINT")")"
else
  CHECKPOINT_NAME="$UPSTREAM"
fi

# Feature selection: "paper" for distilled models, "hidden_states" for standard upstreams
PAPER_ARG="-s $FEATURE_SELECTION"

# Build checkpoint arg (empty if no checkpoint provided, for built-in upstreams)
CKPT_ARG=""
if [[ -n "$CHECKPOINT" ]]; then
  CKPT_ARG="-k $CHECKPOINT"
fi

# Split tasks and GPUs into arrays
IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"

# ----------------------------- Google Sheets args -----------------------------

build_gsheet_args() {
  local args=""
  if [[ -n "$CURRENT_ROW" ]]; then
    args="$args --update_results --current_row_downstream $CURRENT_ROW --current_row $CURRENT_ROW"
  fi
  if [[ -n "$LOGFILE_ROW" ]]; then
    args="$args --logfile_row_downstream $LOGFILE_ROW"
  fi
  if [[ -n "$JSON_FILE" ]]; then
    args="$args --json_file $JSON_FILE"
  fi
  if [[ -n "$SHEET_NAME" ]]; then
    args="$args --sheet_name $SHEET_NAME"
  fi
  if [[ -n "$WORKSHEET_NAME" ]]; then
    args="$args --worksheet_name $WORKSHEET_NAME"
  fi
  if [[ -n "$WORKSHEET_LOGFILE_NAME" ]]; then
    args="$args --worksheet_logfile_name $WORKSHEET_LOGFILE_NAME"
  fi
  echo "$args"
}

GSHEET_ARGS=$(build_gsheet_args)

# ----------------------------- GPU hogger helpers -----------------------------

HOGGER_SCRIPT="${BASE_DIR}/matrix_multiply_efficient.py"

start_hogger() {
  local gpu_list="$1"   # e.g. "0,1,2"
  local batch_num="$2"

  echo "[Hogger] Starting hogger for batch $batch_num on GPUs $gpu_list"
  python "$HOGGER_SCRIPT" \
    --gpu-ids "$gpu_list" \
    --safety-margin-gb 28 \
    --poll-interval 10 \
    --resize-threshold-mb 1000 \
    > "${LOGDIR}/hg_batch_${batch_num}.log" 2>&1 &
  HOGGER_PID=$!
  echo "[Hogger] PID=$HOGGER_PID"
  # Give the hogger time to do initial allocation
  sleep 10
}

stop_hogger() {
  local batch_num="$1"

  echo "[Hogger] Stopping hogger for batch $batch_num (PID=$HOGGER_PID)"

  # Send SIGTERM for graceful cleanup (releases GPU memory)
  if kill -0 "$HOGGER_PID" 2>/dev/null; then
    kill "$HOGGER_PID" 2>/dev/null || true
    # Wait a few seconds for graceful shutdown
    sleep 3
  fi

  # Force-kill if still alive
  if kill -0 "$HOGGER_PID" 2>/dev/null; then
    echo "[Hogger] Force-killing hogger PID=$HOGGER_PID"
    kill -9 "$HOGGER_PID" 2>/dev/null || true
    sleep 1
  fi

  echo "[Hogger] Batch $batch_num hogger stopped"
}

# ----------------------------- Task runner ------------------------------------

run_task() {
  local task_alias="$1"
  local gpu_id="$2"

  export CUDA_VISIBLE_DEVICES="$gpu_id"

  # Map task alias to downstream name, config, and overrides
  local downstream_name=""
  local config_path=""
  local override_args=""
  local eval_ckpt="dev-best.ckpt"
  local eval_split=""
  local train_extra_args=""
  local is_crossval=false

  case "$task_alias" in
    ic)
      downstream_name="fluent_commands"
      config_path="./downstream/fluent_commands/config.yaml"
      override_args='-o "config.downstream_expert.datarc.file_path=/dataset/fabian/superb/fluent_speech_commands_dataset/fluent_speech_commands_dataset"'
      ;;
    ks)
      downstream_name="speech_commands"
      config_path="./downstream/speech_commands/config.yaml"
      override_args='-o "config.downstream_expert.datarc.speech_commands_root=/dataset/fabian/superb/ks/speech_commands_v0.01,,config.downstream_expert.datarc.speech_commands_test_root=/dataset/fabian/superb/ks/speech_commands_test_set_v0.01"'
      ;;
    asr)
      downstream_name="asr"
      config_path="./downstream/asr/config.yaml"
      override_args='-o "config.runner.gradient_accumulate_steps=1,,config.downstream_expert.datarc.train_batch_size=32,,config.downstream_expert.datarc.eval_batch_size=32,,config.downstream_expert.datarc.libri_root=/dataset/fabian/LibriSpeech/,,config.downstream_expert.datarc.bucket_file=./data/len_for_bucket"'
      eval_ckpt="dev-clean-best.ckpt"
      eval_split='-t "test-clean"'
      train_extra_args='--early_stopping_patience 35'
      ;;
    sid)
      downstream_name="voxceleb1"
      config_path="./downstream/voxceleb1/config.yaml"
      ;;
    er)
      downstream_name="emotion"
      config_path="./downstream/emotion/config.yaml"
      is_crossval=true
      ;;
    aec_esc50)
      downstream_name="aec_esc50"
      config_path="./downstream/aec_esc50/config.yaml"
      is_crossval=true
      ;;
    singid)
      downstream_name="vocalset_singer_id"
      config_path="./downstream/vocalset_singer_id/config.yaml"
      ;;
    vocid)
      downstream_name="vocalset_technique_id"
      config_path="./downstream/vocalset_technique_id/config.yaml"
      ;;
    instcls)
      downstream_name="instrument_nsynth"
      config_path="./downstream/instrument_nsynth/config.yaml"
      ;;
    pitchid)
      downstream_name="pitch_nsynth"
      config_path="./downstream/pitch_nsynth/config.yaml"
      ;;
    genreid)
      downstream_name="genre_gtzan"
      config_path="./downstream/genre_gtzan/config.yaml"
      eval_ckpt="valid-best.ckpt"
      ;;
    *)
      # Generic: use alias as-is
      downstream_name="$task_alias"
      config_path="./downstream/${task_alias}/config.yaml"
      ;;
  esac

  local exp_setup="${CHECKPOINT_NAME}/${downstream_name}_paper_method"
  local logfile="logfiles/downstream/${CHECKPOINT_NAME}/${task_alias}"
  local logfile_args=""
  if [[ -n "$LOGFILE_ROW" ]]; then
    logfile_args="--logfile $logfile"
  fi

  echo "[GPU $gpu_id] Starting task '$task_alias' (downstream=$downstream_name)"

  if [[ "$is_crossval" == true ]]; then
    # 5-fold cross-validation (emotion, aec_esc50)
    for test_fold in fold1 fold2 fold3 fold4 fold5; do
      local fold_exp="${exp_setup}_${test_fold}"
      local fold_override="-o \"config.downstream_expert.datarc.test_fold=${test_fold}\""

      if [[ "$STAGE" == "train" ]]; then
        echo "[GPU $gpu_id] Training $downstream_name $test_fold ..."
        eval python run_downstream.py -m train \
          -c "$config_path" \
          -u "$UPSTREAM" $CKPT_ARG $PAPER_ARG \
          -d "$downstream_name" \
          -p "${DOWNSTREAM_PATH}/${fold_exp}" \
          --verbose $GSHEET_ARGS $logfile_args \
          "$fold_override"

        echo "[GPU $gpu_id] Evaluating $downstream_name $test_fold ..."
        eval python run_downstream.py -m evaluate --verbose \
          -e "${DOWNSTREAM_PATH}/${fold_exp}/dev-best.ckpt" \
          $CKPT_ARG $PAPER_ARG \
          -u "$UPSTREAM" $GSHEET_ARGS \
          -d "$downstream_name" \
          -c "$config_path" \
          "$fold_override"

      elif [[ "$STAGE" == "evaluating" ]]; then
        echo "[GPU $gpu_id] Evaluating $downstream_name $test_fold ..."
        eval python run_downstream.py -m evaluate --verbose \
          -e "${DOWNSTREAM_PATH}/${fold_exp}/dev-best.ckpt" \
          $CKPT_ARG $PAPER_ARG \
          -u "$UPSTREAM" $GSHEET_ARGS \
          -d "$downstream_name" \
          -c "$config_path" \
          "$fold_override"
      fi
    done
  else
    # Standard single-run task
    if [[ "$STAGE" == "train" ]]; then
      echo "[GPU $gpu_id] Training $downstream_name ..."
      eval python run_downstream.py -m train \
        -c "$config_path" \
        -u "$UPSTREAM" $CKPT_ARG $PAPER_ARG \
        -d "$downstream_name" \
        -p "${DOWNSTREAM_PATH}/${exp_setup}" \
        --verbose $GSHEET_ARGS $logfile_args \
        $override_args $eval_split $train_extra_args

      echo "[GPU $gpu_id] Training done. Running evaluation ..."
      eval python run_downstream.py -m evaluate --verbose \
        -e "${DOWNSTREAM_PATH}/${exp_setup}/${eval_ckpt}" \
        $CKPT_ARG $PAPER_ARG \
        -u "$UPSTREAM" $GSHEET_ARGS \
        -d "$downstream_name" \
        -c "$config_path" \
        $eval_split $override_args

    elif [[ "$STAGE" == "evaluating" ]]; then
      echo "[GPU $gpu_id] Evaluating $downstream_name ..."
      eval python run_downstream.py -m evaluate --verbose \
        -e "${DOWNSTREAM_PATH}/${exp_setup}/${eval_ckpt}" \
        $CKPT_ARG $PAPER_ARG \
        -u "$UPSTREAM" $GSHEET_ARGS \
        -d "$downstream_name" \
        -c "$config_path" \
        $eval_split $override_args
    fi
  fi

  echo "[GPU $gpu_id] Task '$task_alias' completed."
}

# ----------------------------- Queue-based dispatch ---------------------------
# Instead of batching tasks and waiting for the entire batch to finish, each GPU
# runs a worker loop that grabs the next available task from a shared queue as
# soon as it finishes the current one.  This eliminates idle GPU time when tasks
# within the same "batch" have very different durations (e.g. KS ~3h vs ASR ~16h).

NUM_GPUS=${#GPU_ARRAY[@]}
NUM_TASKS=${#TASK_ARRAY[@]}

echo "============================================================"
echo "Downstream local runner (queue-based)"
echo "  Checkpoint : $CHECKPOINT"
echo "  Upstream   : $UPSTREAM"
echo "  Tasks      : ${TASK_ARRAY[*]}"
echo "  GPUs       : ${GPU_ARRAY[*]}"
echo "  Stage      : $STAGE"
echo "  Workers    : $NUM_GPUS GPU(s) processing $NUM_TASKS task(s)"
echo "============================================================"

LOGDIR="logfiles/downstream/${CHECKPOINT_NAME}"
mkdir -p "$LOGDIR"

# --- Shared task queue (atomic via flock) ------------------------------------
TASK_INDEX_FILE=$(mktemp)
echo "0" > "$TASK_INDEX_FILE"
QUEUE_LOCK="${LOGDIR}/.queue_lock"

FAIL_COUNT_FILE=$(mktemp)
echo "0" > "$FAIL_COUNT_FILE"
FAIL_LOCK="${LOGDIR}/.fail_lock"

# Atomically get-and-increment the next task index
get_next_task_index() {
  (
    flock 200
    local idx
    idx=$(cat "$TASK_INDEX_FILE")
    echo $((idx + 1)) > "$TASK_INDEX_FILE"
    echo "$idx"
  ) 200>"$QUEUE_LOCK"
}

# Atomically increment the failure counter
record_failure() {
  (
    flock 201
    local f
    f=$(cat "$FAIL_COUNT_FILE")
    echo $((f + 1)) > "$FAIL_COUNT_FILE"
  ) 201>"$FAIL_LOCK"
}

# Worker loop: keep pulling tasks until the queue is empty
gpu_worker() {
  local gpu_id="$1"
  while true; do
    local idx
    idx=$(get_next_task_index)
    if [[ $idx -ge $NUM_TASKS ]]; then
      echo "[Worker GPU $gpu_id] No more tasks, exiting."
      break
    fi
    local task_alias="${TASK_ARRAY[$idx]}"
    local task_logfile="${LOGDIR}/${task_alias}"
    echo "[Worker GPU $gpu_id] Starting task '$task_alias' ($((idx+1))/$NUM_TASKS), logging to $task_logfile"
    if run_task "$task_alias" "$gpu_id" > "$task_logfile" 2>&1; then
      echo "[Worker GPU $gpu_id] Task '$task_alias' completed successfully"
    else
      echo "[Worker GPU $gpu_id] Task '$task_alias' FAILED" >&2
      record_failure
    fi
  done
}

# --- Start GPU hogger for all GPUs (once, for the full run) ------------------
ALL_GPU_LIST=$(IFS=,; echo "${GPU_ARRAY[*]}")
HOGGER_PID=""
start_hogger "$ALL_GPU_LIST" "all"

# --- Launch one worker per GPU -----------------------------------------------
WORKER_PIDS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
  gpu_worker "$gpu_id" &
  WORKER_PIDS+=($!)
done

# --- Wait for all workers to finish ------------------------------------------
for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid"
done

# --- Stop hogger -------------------------------------------------------------
stop_hogger "all"

# --- Cleanup & report --------------------------------------------------------
TOTAL_FAILED=$(cat "$FAIL_COUNT_FILE")
rm -f "$TASK_INDEX_FILE" "$FAIL_COUNT_FILE"

echo ""
if [[ $TOTAL_FAILED -gt 0 ]]; then
  echo "WARNING: $TOTAL_FAILED task(s) failed overall."
  exit 1
fi

echo "All $NUM_TASKS tasks completed successfully."
