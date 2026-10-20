#!/bin/bash
# GroundLink retraining after the force-unit fix of 2026-10-08 (released .npy forces are kN; every earlier GL run
# trained on targets with a hidden per-subject gain). One array task per experiment cell; the headline LOSO and the
# trial split use their own arrays (see README). All outputs go under runs/$V2 and outputs/$V2 so nothing published is
# overwritten. Requires GROUNDLINK_GRF_UNITS=kilonewton (now the code default and the .env value).
#   sbatch --array=0-22%12 chapter1/hprc/gl_family.slurm
#   smoke: sbatch --array=9,13 --time=00:25:00 --export=ALL,V2=smoke,SMOKE_EPOCHS=1 chapter1/hprc/gl_family.slurm
# Index -> cell:
#    0-8  ablation_prep GL, one arm each, seeds 42,43,44            (Supplementary Table S10)
#    9-12 ablation_runner GL, 2x2 frame factorial, one arm each    (Sec. IV-B, Fig. 3, Supp. S3)
#   13-15 m6 GL LOSO, paper HP, ICF / CNN-LSTM / CNN + GL->PH eval (per-channel table, Table V GL->PH, matched comparison)
#   16    m6 GL LOSO with s001 reinstated, ICF                       (exclusion sensitivity)
#   17-19 m6 PH LOSO with PH->GL eval, ICF / CNN-LSTM / CNN         (Table V PH->GL; PH training unchanged, GL targets changed)
#   20-22 pooled dual-holdout, ICF / CNN-LSTM / CNN                 (Table IV, matched comparison)
# Time limits are per cell; FASTER reserves tasks x limit x 136 SU up front (about 7.1k SU for this array).
#SBATCH --job-name=ch1-gl_v2
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
V2="${V2:-v2}"
export GROUNDLINK_GRF_UNITS=kilonewton ROTATE_FORCE_VECTORS=1 INPUT_NORM_MODE=per_fold ENABLE_BIOMECH_PREPROCESSING=1 FORCE_TARGET_MODE=nbw
mkdir -p runs/$V2 outputs/$V2
i=$SLURM_ARRAY_TASK_ID
ABL9=(full no_biomech no_derived no_rel no_flip no_peak no_pelvis no_heading no_height)
ARM4=(baseline corot perfold both)
MODELS=(ImprovedConvFormer CNN_LSTM CNN); TAGS=(icf cnn-lstm cnn)
GLP=(configs/best_params_icf.json configs/best_params_cnn-lstm_gl.json configs/best_params_cnn_gl.json)
PHP=(configs/best_params_icf.json configs/best_params_cnn-lstm_ph.json configs/best_params_cnn_ph.json)
export A0_SEED=42 WT_MARKER_SET=n/a
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_$i"
if   [ "$i" -le 8 ]; then
  export SELECTED_DATASET=GROUNDLINK ABL_ARMS="${ABL9[$i]}" ABL_SEEDS=42,43,44 ABL_EPOCHS=${SMOKE_EPOCHS:-120} ABL_OUT=outputs/$V2/ablation_prep/${ABL9[$i]}
  export A0_MODELS="ablation_prep:${ABL_ARMS}" JOB_NOTE="Ch1 GL v2 ablation_prep ${ABL_ARMS}"; log_label
  python -u chapter1/experiments/ablation_prep.py
elif [ "$i" -le 12 ]; then
  export ABL_DATASETS=GROUNDLINK ABL_ARMS="${ARM4[$((i - 9))]}" ABL_EPOCHS=${SMOKE_EPOCHS:-120} ABL_OUT=outputs/$V2/ablation
  export A0_MODELS="ablation2x2:${ABL_ARMS}" JOB_NOTE="Ch1 GL v2 frame factorial ${ABL_ARMS}"; log_label
  python -u chapter1/experiments/ablation_runner.py
elif [ "$i" -le 16 ]; then
  k=$(( i - 13 )); [ "$i" -eq 16 ] && k=0
  export SELECTED_DATASET=GROUNDLINK USE_LOSO=1 MODEL_NAME="${MODELS[$k]}" LOSO_BEST_PARAMS="${GLP[$k]}" OUTPUT_ROOT=runs/$V2 CROSS_LAB_SUBJECT=ph_1   # .env on the cluster may set another subject; pin it
  if [ "$i" -eq 16 ]; then export RECOVER_EXCLUDED=1 RUN_NAME_OVERRIDE=m6_recover_gl; else export RUN_NAME_OVERRIDE="m6_${TAGS[$k]}_gl"; fi
  export A0_MODELS="$RUN_NAME_OVERRIDE" JOB_NOTE="Ch1 GL v2 $RUN_NAME_OVERRIDE (GL LOSO + GL->PH eval)"; log_label
  CFG=configs/loso_crosslab.yaml; [ -n "${SMOKE_EPOCHS:-}" ] && { CFG=configs/_smoke_loso.yaml; sed "s/final_epochs: .*/final_epochs: $SMOKE_EPOCHS/" configs/loso_crosslab.yaml > $CFG; }
  python -u chapter1/pipeline/main.py --config $CFG
elif [ "$i" -le 19 ]; then
  k=$(( i - 17 ))
  export SELECTED_DATASET=PATIENT_HANDLING USE_LOSO=1 MODEL_NAME="${MODELS[$k]}" LOSO_BEST_PARAMS="${PHP[$k]}" OUTPUT_ROOT=runs/$V2 CROSS_LAB_SUBJECT=gl_2 RUN_NAME_OVERRIDE="m6_phxlab_${TAGS[$k]}"
  export A0_MODELS="$RUN_NAME_OVERRIDE" JOB_NOTE="Ch1 GL v2 $RUN_NAME_OVERRIDE (PH LOSO + PH->GL eval)"; log_label
  CFG=configs/ph_loso_crosslab.yaml; [ -n "${SMOKE_EPOCHS:-}" ] && { CFG=configs/_smoke_ph.yaml; sed "s/final_epochs: .*/final_epochs: $SMOKE_EPOCHS/" configs/ph_loso_crosslab.yaml > $CFG; }
  python -u chapter1/pipeline/main.py --config $CFG
else
  k=$(( i - 20 ))
  export MODEL_NAME="${MODELS[$k]}" RUN_NAME_OVERRIDE="v2_m6_combined_${TAGS[$k]}"
  export A0_MODELS="$RUN_NAME_OVERRIDE" JOB_NOTE="Ch1 GL v2 pooled dual-holdout ${TAGS[$k]}"; log_label
  python -u chapter1/pipeline/combined_hpo_cv.py --config configs/combined_dual_holdout.yaml --best-params "${GLP[$k]}"
fi
