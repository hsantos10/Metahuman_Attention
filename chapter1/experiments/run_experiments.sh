#!/usr/bin/env bash
# Ledger of the Chapter 1 revision experiments as they were run (Sep 21-23, 2026, RTX 4090, svd env),
# and the way to rerun one:   bash chapter1/experiments/run_experiments.sh <job>
#
#   e1c                 e1_multiseed_corrected.py  GL then PH  -> outputs/e1_multiseed_corrected/   (Sep 21; log runs/e1c_multiseed.log)
#   trial_split         trial_split_corrected.py   GL then PH  -> outputs/trial_split_corrected/    (Sep 21; log runs/trial_split.log; ran alongside e1c)
#   ablation_seed42     ablation_prep.py, seed 42, the seven original arms, GL then PH -> outputs/ablation_prep/
#                       (Sep 21-22; log runs/ablation_prep.log; the original driver waited for "E1c DONE" in runs/e1c_multiseed.log first)
#   ablation_v2         ablation_prep.py: block A = no_pelvis/no_heading/no_height x seeds 42,43,44;
#                       block B = seeds 43,44 on the six seed-42 arms; GL then PH per block -> outputs/ablation_prep/
#                       (Sep 22-23; log runs/ablation_v2.log)
#   ablation_v2_resume  block B on PH only: the resume after the process was killed 2026-09-23 ~03:50 mid-fold
#                       (no traceback; external kill). Same command as the last line of ablation_v2.
#
# Every command and environment value below is byte-for-byte what the five original driver scripts ran
# (chapter1/launchers/*.sh before 2026-10-02; `git show pre-cleanup-2026-10:chapter1/launchers/<name>.sh`).
# All scripts RESUME from their CSV (keyed by config/arm, subject, seed): for a from-scratch rerun move
# outputs/<name>/ aside first. "corrected" = ROTATE_FORCE_VECTORS=1 + INPUT_NORM_MODE=per_fold, set inside each script.
# The ablation_seed42 arm list is pinned here because ablation_prep.py's default list has since grown
# (no_pelvis/no_heading/no_height added 2026-09-22); no_metadata is listed as run but produced zero folds.
set -uo pipefail
cd "$(dirname "$0")/../.."                 # repo root: scripts use relative configs/ and outputs/ paths
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY="${PY:-python}"
ABL_ORIGINAL_ARMS=full,no_biomech,no_derived,no_rel,no_flip,no_metadata,no_peak

case "${1:-}" in
  e1c)
    export E1_SEEDS=42,43,44 E1_EPOCHS=120
    echo "===== E1c GL start: $(date)"
    SELECTED_DATASET=GROUNDLINK $PY -u chapter1/experiments/e1_multiseed_corrected.py
    echo "GL exit=$?  $(date)"
    echo "===== E1c PH start: $(date)"
    SELECTED_DATASET=PATIENT_HANDLING $PY -u chapter1/experiments/e1_multiseed_corrected.py
    echo "PH exit=$?  $(date)"
    echo "===== E1c DONE: $(date)" ;;
  trial_split)
    export TS_EPOCHS=120 TS_SEED=42
    echo "===== trial-split GL start: $(date)"
    SELECTED_DATASET=GROUNDLINK $PY -u chapter1/experiments/trial_split_corrected.py
    echo "GL exit=$?  $(date)"
    echo "===== trial-split PH start: $(date)"
    SELECTED_DATASET=PATIENT_HANDLING $PY -u chapter1/experiments/trial_split_corrected.py
    echo "PH exit=$?  $(date)"
    echo "===== trial-split DONE: $(date)" ;;
  ablation_seed42)
    export ABL_EPOCHS=120 ABL_SEED=42 ABL_ARMS=$ABL_ORIGINAL_ARMS
    echo "===== ablation GL start: $(date)"
    SELECTED_DATASET=GROUNDLINK $PY -u chapter1/experiments/ablation_prep.py
    echo "GL exit=$?  $(date)"
    echo "===== ablation PH start: $(date)"
    SELECTED_DATASET=PATIENT_HANDLING $PY -u chapter1/experiments/ablation_prep.py
    echo "PH exit=$?  $(date)"
    echo "===== ablation_prep DONE: $(date)" ;;
  ablation_v2)
    export ABL_EPOCHS=120
    echo "===== block A (biomech split, seeds 42,43,44) GL start: $(date)"
    SELECTED_DATASET=GROUNDLINK ABL_ARMS=no_pelvis,no_heading,no_height ABL_SEEDS=42,43,44 $PY -u chapter1/experiments/ablation_prep.py
    echo "A-GL exit=$?  $(date)"
    echo "===== block A PH start: $(date)"
    SELECTED_DATASET=PATIENT_HANDLING ABL_ARMS=no_pelvis,no_heading,no_height ABL_SEEDS=42,43,44 $PY -u chapter1/experiments/ablation_prep.py
    echo "A-PH exit=$?  $(date)"
    echo "===== block B (extra seeds on the six arms) GL start: $(date)"
    SELECTED_DATASET=GROUNDLINK ABL_ARMS=full,no_biomech,no_derived,no_rel,no_flip,no_peak ABL_SEEDS=43,44 $PY -u chapter1/experiments/ablation_prep.py
    echo "B-GL exit=$?  $(date)"
    echo "===== block B PH start: $(date)"
    SELECTED_DATASET=PATIENT_HANDLING ABL_ARMS=full,no_biomech,no_derived,no_rel,no_flip,no_peak ABL_SEEDS=43,44 $PY -u chapter1/experiments/ablation_prep.py
    echo "B-PH exit=$?  $(date)"
    echo "===== ablation_v2 DONE: $(date)" ;;
  ablation_v2_resume)
    export ABL_EPOCHS=120
    echo "===== block B PH (resume) start: $(date)"
    SELECTED_DATASET=PATIENT_HANDLING ABL_ARMS=full,no_biomech,no_derived,no_rel,no_flip,no_peak ABL_SEEDS=43,44 $PY -u chapter1/experiments/ablation_prep.py
    echo "B-PH exit=$?  $(date)"
    echo "===== ablation_v2 DONE: $(date)" ;;
  *)
    sed -n '2,15p' "$0"; exit 1 ;;
esac
