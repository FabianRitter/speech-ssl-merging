#!/bin/bash
#SBATCH --job-name=ds_3LLKD_G1
#SBATCH --ntasks=1 --cpus-per-task=24
#SBATCH --partition=RTXA6Kq --nodelist=node09
#SBATCH --time=02-12:00:00
#SBATCH --output=/export/home2/fabian/projects/multi_distiller/s3prl/s3prl/logfiles/downstream/ds_3L_LKD_G1_slurm_%j.out

cd /export/home2/fabian/projects/multi_distiller/s3prl/s3prl
set +u
source /export/home2/fabian/miniconda3/bin/activate s3prl_multidistiller
set -euo pipefail

declare -A ROWS=( ["0.9_0.1"]=150 ["0.8_0.2"]=151 ["0.5_0.5"]=152 ["0.2_0.8"]=153 ["0.1_0.9"]=154 )

for interp in "0.9_0.1" "0.8_0.2" "0.5_0.5" "0.2_0.8" "0.1_0.9"; do
    ROW=${ROWS[$interp]}
    LOGFILE_ROW=$((ROW + 200))
    CKPT="result/pretrain/perm_3L_LKD_G1_${interp}/learning_by_addition.ckpt"

    echo "[$(date)] Starting downstream for 3L_LKD G1 ${interp} (row=${ROW})"

    bash run_downstream_local.sh \
        --checkpoint "$CKPT" \
        --gpus 0,1,2,3 \
        --tasks ic,instcls,genreid,aec_esc50 \
        --upstream multi_distiller_local \
        --stage train \
        --current_row "$ROW" \
        --logfile_row "$LOGFILE_ROW" \
        --worksheet_name "Correlation-Merging-Results" \
        --worksheet_logfile_name "Correlation-Merging-Results-Logfiles"

    echo "[$(date)] Completed ${interp}"
done

echo "[$(date)] All 3L_LKD G1 downstream evaluations complete"
