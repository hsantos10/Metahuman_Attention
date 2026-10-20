# Source in every job / interactive shell:  source hprc/activate.sh
_h="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_h/hprc.env"
module purge
module load WebProxy GCCcore/13.3.0 Python/3.12.3
source "$REMOTE_VENV/bin/activate"
export TAMU_DATA_ROOT="$REMOTE_DATA"
export GROUNDLINK_ROOT="$REMOTE_GROUNDLINK"
export WANDB_MODE="${WANDB_MODE:-offline}"   # `wandb sync wandb/offline-run-*` later, or `wandb login` once and set online
export PYTHONUNBUFFERED=1
cd "$REMOTE_CODE"
# Call right before the python line of a job: appends one row to runs/job_labels.csv (job id -> cell label).
log_label() {
  local id="${SLURM_JOB_ID:-local}"
  [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  mkdir -p runs
  printf '%s,%s,%s,%s,%s,%s\n' "$id" "${SELECTED_DATASET:-}" "${A0_MODELS:-all}" "${A0_SEED:-42}" \
    "${WT_MARKER_SET:-common20}" "${JOB_NOTE:-}" >> runs/job_labels.csv
}
