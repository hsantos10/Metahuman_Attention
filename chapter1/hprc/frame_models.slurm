#!/bin/bash
# Target-frame 2x2 factorial (Sec. IV-B) repeated for the CNN-LSTM and the CNN, each at its tuned per-dataset
# configuration, on the v2 pipeline (GroundLink kN fix). Closes the Limitations sentence "the frame study uses
# one fixed ICF configuration". One task per (model, dataset, arm) = 16 tasks; CSVs land in
# outputs/v2/ablation_<model>/<tag>_<arm>.csv (one file per task, so tasks cannot overwrite each other).
#   sbatch --array=0-15%12 chapter1/hprc/frame_models.slurm
#   smoke: sbatch --array=0,8 --time=00:20:00 --export=ALL,ABL_SMOKE=1,ABL_EPOCHS=1,FM_OUT=outputs/smoke_frame chapter1/hprc/frame_models.slurm
# Index i -> model i/8 (0 cnn-lstm, 1 cnn), dataset (i%8)/4 (0 GL, 1 PH), arm i%4 (baseline corot perfold both).
#SBATCH --job-name=ch1-frame_models
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
source hprc/activate.sh
export GROUNDLINK_GRF_UNITS=kilonewton ENABLE_BIOMECH_PREPROCESSING=1 FORCE_TARGET_MODE=nbw A0_SEED=42 WT_MARKER_SET=n/a
i=$SLURM_ARRAY_TASK_ID
MODELS=(cnn-lstm cnn); DS=(GROUNDLINK PATIENT_HANDLING); ARMS=(baseline corot perfold both)
export ABL_MODEL="${MODELS[$((i / 8))]}" ABL_DATASETS="${DS[$(((i % 8) / 4))]}" ABL_ARMS="${ARMS[$((i % 4))]}"
export ABL_EPOCHS="${ABL_EPOCHS:-120}" ABL_OUT="${FM_OUT:-outputs/v2}/ablation_${ABL_MODEL}"
export A0_MODELS="frame:${ABL_MODEL}:${ABL_ARMS}" JOB_NOTE="Ch1 frame factorial ${ABL_MODEL} ${ABL_DATASETS} ${ABL_ARMS}"; log_label
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_$i model=$ABL_MODEL dataset=$ABL_DATASETS arm=$ABL_ARMS out=$ABL_OUT"
python -u chapter1/experiments/ablation_runner.py
