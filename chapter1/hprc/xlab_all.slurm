#!/bin/bash
# Cross-lab transfer on EVERY subject of the other laboratory (the protocol the paper's Methods and Table V
# describe), using the fold models already trained by gl_family.slurm tasks 13-15 and 17-19. No training:
# XLAB_EVAL_ONLY loads runs/v2/<run>/loso_models/fold_<subject>.pt; each fold's data split and normalizers are
# rebuilt deterministically, and the in-domain held-out score is re-reported as a check that the rebuild matches.
#   sbatch --account=YOUR_ACCOUNT --array=0-5 chapter1/hprc/xlab_all.slurm
# Index: 0-2 GL-trained ICF / CNN-LSTM / CNN -> all PH subjects; 3-5 PH-trained -> all GL subjects.
#SBATCH --job-name=ch1-xlab_all
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
source hprc/activate.sh
export GROUNDLINK_GRF_UNITS=kilonewton ROTATE_FORCE_VECTORS=1 INPUT_NORM_MODE=per_fold ENABLE_BIOMECH_PREPROCESSING=1 FORCE_TARGET_MODE=nbw
export USE_LOSO=1 OUTPUT_ROOT=runs/v2 CROSS_LAB_SUBJECT=all A0_SEED=42 WT_MARKER_SET=n/a
i=$SLURM_ARRAY_TASK_ID; k=$(( i % 3 ))
MODELS=(ImprovedConvFormer CNN_LSTM CNN); TAGS=(icf cnn-lstm cnn)
GLP=(configs/best_params_icf.json configs/best_params_cnn-lstm_gl.json configs/best_params_cnn_gl.json)
PHP=(configs/best_params_icf.json configs/best_params_cnn-lstm_ph.json configs/best_params_cnn_ph.json)
export MODEL_NAME="${MODELS[$k]}"
if [ "$i" -le 2 ]; then
  SRC="m6_${TAGS[$k]}_gl"; export SELECTED_DATASET=GROUNDLINK LOSO_BEST_PARAMS="${GLP[$k]}"; CFG=configs/loso_crosslab.yaml
else
  SRC="m6_phxlab_${TAGS[$k]}"; export SELECTED_DATASET=PATIENT_HANDLING LOSO_BEST_PARAMS="${PHP[$k]}"; CFG=configs/ph_loso_crosslab.yaml
fi
export XLAB_EVAL_ONLY="runs/v2/$SRC/loso_models" RUN_NAME_OVERRIDE="${SRC}_xlaball"
[ -d "$XLAB_EVAL_ONLY" ] || { echo "missing $XLAB_EVAL_ONLY"; exit 1; }
export A0_MODELS="$RUN_NAME_OVERRIDE" JOB_NOTE="Ch1 cross-lab on all subjects, eval only, from $SRC"; log_label
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_$i src=$SRC cfg=$CFG"
python -u chapter1/pipeline/main.py --config $CFG
