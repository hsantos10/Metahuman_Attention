#!/bin/bash
# Chapter 1 trial-level split (trial_split_corrected.py) rerun with per-frame predictions saved, three
# seeds (the published row was seed 42 only), for the leakage table in biomechanical units.
# One array task per (dataset, config, seed) = 24 tasks, one training run each.
#   sbatch --array=0-23%12 chapter1/hprc/ts_pred.slurm
# Index i -> dataset i / 12 (0 = GROUNDLINK, 1 = PATIENT_HANDLING), config (i % 12) / 3, seed 42 + i % 3.
# Outputs: outputs/ts_pred/{gl,ph}_<config>_s<seed>_trial_split.csv and outputs/ts_pred/pred/*.npz
# SU reservation: 24 x 0.5 h x 136 = 1.6k SU; billed by elapsed time (one training run, ~5-15 min).
#SBATCH --job-name=ch1-ts_pred
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
source hprc/activate.sh
DATASETS=(GROUNDLINK PATIENT_HANDLING)
CONFIGS=(ICF-hpo ICF-grid128 CNN-LSTM-hpo CNN-hpo)
i=$SLURM_ARRAY_TASK_ID
export SELECTED_DATASET="${DATASETS[$((i / 12))]}"
export TS_CONFIGS="${CONFIGS[$(((i % 12) / 3))]}"
export TS_SEED="$((42 + i % 3))"
export TS_EPOCHS="${TS_EPOCHS:-120}"
export TS_OUT="${TS_OUT:-outputs/ts_pred}"
export TS_SAVE_PRED="$TS_OUT/pred"
export TS_TAG="_${TS_CONFIGS}_s${TS_SEED}"
export GROUNDLINK_GRF_UNITS="${GROUNDLINK_GRF_UNITS:-kilonewton}"   # released GL forces are kN (fix of 2026-10-08); beats a stale .env
export A0_MODELS="$TS_CONFIGS" A0_SEED="$TS_SEED"            # for log_label only
export JOB_NOTE="${JOB_NOTE:-Ch1 trial split with predictions, $SELECTED_DATASET $TS_CONFIGS seed $TS_SEED}"
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_$i dataset=$SELECTED_DATASET config=$TS_CONFIGS seed=$TS_SEED out=$TS_OUT"
log_label
python -u chapter1/experiments/trial_split_corrected.py
