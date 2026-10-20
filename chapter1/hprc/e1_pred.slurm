#!/bin/bash
# Chapter 1 headline LOSO (e1_multiseed_corrected.py) rerun with per-frame predictions saved, for the
# biomechanical-unit tables of the JBHI paper. One array task per (dataset, config, seed) = 24 tasks,
# each running every fold of its dataset (6 GL, 10 PH) sequentially.
#   sbatch --array=0-23%12 chapter1/hprc/e1_pred.slurm
#   sbatch --array=0-23%12 --gres=gpu:a40:1 chapter1/hprc/e1_pred.slurm     # when the A100 queue is long
# Index i -> dataset i / 12 (0 = GROUNDLINK, 1 = PATIENT_HANDLING), config (i % 12) / 3, seed 42 + i % 3.
# Outputs: outputs/e1_pred/{gl,ph}_<config>_s<seed>_multiseed.csv and outputs/e1_pred/pred/*.npz
# (pull with copying them back from the cluster, then chapter1/experiments/biomech_units.py).
# SU: FASTER reserves tasks x limit x 136 up front = 24 x 1.5 h x 136 = 4.9k SU; billed by elapsed time
# (~3 min per fold on an A100 -> 20-30 min per task).
#SBATCH --job-name=ch1-e1_pred
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
source hprc/activate.sh
DATASETS=(GROUNDLINK PATIENT_HANDLING)
CONFIGS=(ICF-hpo ICF-grid128 CNN-LSTM-hpo CNN-hpo)
i=$SLURM_ARRAY_TASK_ID
export SELECTED_DATASET="${DATASETS[$((i / 12))]}"
export E1_CONFIGS="${CONFIGS[$(((i % 12) / 3))]}"
export E1_SEEDS="$((42 + i % 3))"
export E1_EPOCHS="${E1_EPOCHS:-120}"
export E1_OUT="${E1_OUT:-outputs/e1_pred}"
export E1_SAVE_PRED="$E1_OUT/pred"
export E1_TAG="_${E1_CONFIGS}_s${E1_SEEDS}"
export GROUNDLINK_GRF_UNITS="${GROUNDLINK_GRF_UNITS:-kilonewton}"   # released GL forces are kN (fix of 2026-10-08); beats a stale .env
export A0_MODELS="$E1_CONFIGS" A0_SEED="$E1_SEEDS"            # for log_label only
export JOB_NOTE="${JOB_NOTE:-Ch1 e1 LOSO with predictions, $SELECTED_DATASET $E1_CONFIGS seed $E1_SEEDS}"
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_$i dataset=$SELECTED_DATASET config=$E1_CONFIGS seed=$E1_SEEDS out=$E1_OUT"
log_label
python -u chapter1/experiments/e1_multiseed_corrected.py
