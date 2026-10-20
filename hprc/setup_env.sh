#!/usr/bin/env bash
# One-time Python env on FASTER. Run on a login node:  bash hprc/setup_env.sh
# PyTorch pip wheels bundle CUDA, so only a Python module is needed (the node driver does the rest).
set -euo pipefail
source "$(dirname "$0")/hprc.env"
module purge
module load WebProxy                      # outbound internet for pip / wandb
module load GCCcore/13.3.0 Python/3.12.3  # check `module spider Python` if this pair changes
python -m venv --clear "$REMOTE_VENV"
source "$REMOTE_VENV/bin/activate"
pip install --upgrade pip wheel
# Versions pinned to the workstation `svd` env (hprc/requirements-lock.txt) so results match Chapter 1.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r "$REMOTE_CODE/hprc/requirements-lock.txt"
python -c "import torch, polars, optuna; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
