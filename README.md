# Subject-Held-Out Evaluation of Deep Ground Reaction Force Estimators Across Two Laboratories

Code, configurations and run outputs for the paper

> Hernan David Santos, Yinong Chen, Yu Ding, Xudong Zhang, Jian Tao.
> *Subject-Held-Out Evaluation of Deep Ground Reaction Force Estimators Across Two Laboratories: Leakage and
> Cross-Laboratory Transfer.* Submitted to IEEE Journal of Biomedical and Health Informatics, 2026.

Three sequence models map 3-D marker trajectories to per-frame ground reaction forces and centres of pressure for two
force plates (10 output channels: `ground_force_{1,2}_{ml,vert,ap,px,pz}`). The models are a residual 1-D CNN, a
CNN-LSTM, and a CNN-Transformer hybrid (`ImprovedConvFormer`, ICF in the code). They are evaluated on two datasets from
different laboratories, GroundLink (locomotion) and Patient Handling (occupational tasks), under four protocols:
trial-level random split, leave-one-subject-out (LOSO), pooled two-laboratory training with matched held-out subjects,
and cross-laboratory transfer. The paper also reports a target-frame factorial for all three models, a preprocessing
ablation, a ridge baseline and a Newton's-law frame check.

The tag `jbhi-2026` is the code state that produced the paper's numbers.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Shared library: dataset adapters and file readers, preprocessing (pelvis rooting, heading alignment, height scaling, force co-rotation), features, models (`src/model.py`), loss, training loop, normalisation |
| `chapter1/pipeline/` | Entry points: `main.py` (standard or LOSO training, `USE_LOSO=1`, cross-lab evaluation) and `combined_hpo_cv.py` (pooled GL+PH dual-holdout CV) |
| `chapter1/experiments/` | One script per paper experiment (multi-seed LOSO, trial split, frame factorial, preprocessing ablation, ridge baseline, HPO, unit rescoring, staging for the paper build) |
| `chapter1/hprc/` | Slurm array scripts used for the final runs (TAMU HPRC FASTER and Grace; A100) |
| `hprc/` | Cluster environment kit sourced by the Slurm scripts (`activate.sh`, `setup_env.sh`, pinned `requirements-lock.txt`, `hprc.env.example`) |
| `configs/` | Dataset YAMLs and the tuned hyperparameters (`best_params_*.json`) |
| `data/groundlink_metadata.csv` | GroundLink subject mass and height (from the public dataset) |
| `results/paper_v2_stage/` | Every run output the paper's tables and figures are built from (CSV/JSON) |
| `paper_build/` | Scripts that turn `results/` into the paper's numbers, tables and figures |
| `tests/` | Unit tests (`pytest`) |

## Environment

The paper's runs used Python 3.12.3 and PyTorch 2.5.1 (CUDA 12.4 wheel), with the versions pinned in
`hprc/requirements-lock.txt`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r hprc/requirements-lock.txt          # pinned versions used for the paper
# or, unpinned: pip install -r requirements.txt     (or: pip install -e .)
python -m pytest -q                                  # 28 unit tests, no data needed
```

Experiments are configured with environment variables, not command-line flags. Settings resolve in this order, later
ones winning: code defaults, then the YAML, then `.env`, then the shell. Copy `.env.example` to `.env` before running;
it holds the values used for the paper. Run every script from the repository root. The variables that matter for the
paper:

| Variable | Paper value | Effect |
|---|---|---|
| `GROUNDLINK_GRF_UNITS` | `kilonewton` | The released GroundLink force arrays are in kN. Any other value reproduces a units bug fixed before the final runs. |
| `GLOBAL_MAX_LEN_OVERRIDE` | `2500` | Maximum trial length in frames (longer trials are truncated). |
| `FORCE_TARGET_MODE` | `nbw` | Forces as multiples of body weight. |
| `ROTATE_FORCE_VECTORS` | `1` | Co-rotates the horizontal force targets with the heading-aligned inputs. `0` leaves them in the plate frame (the frame factorial's other arm). |
| `INPUT_NORM_MODE` | `per_fold` | Input z-score fitted on the training fold only. |
| `ENABLE_BIOMECH_PREPROCESSING` | `1` | Pelvis rooting, heading alignment and height scaling. |
| `USE_BILATERAL_FLIP` | `1` | Left-right mirror augmentation. |
| `USE_LOSO`, `CROSS_LAB_SUBJECT` | see `chapter1/README.md` | LOSO and cross-laboratory evaluation in `chapter1/pipeline/main.py`. |
| `GROUNDLINK_ROOT`, `TAMU_DATA_ROOT` | your paths | Dataset locations. Read at import time, so set them in the shell, not in `.env`. |

The experiment scripts in `chapter1/experiments/` and the Slurm files set `ROTATE_FORCE_VECTORS`, `INPUT_NORM_MODE`,
`ENABLE_BIOMECH_PREPROCESSING` and `FORCE_TARGET_MODE` themselves.

## Data

Neither dataset is redistributed here. Point the code at local copies (in the shell):

```bash
export GROUNDLINK_ROOT=/path/to/GroundLink          # contains c3d/c3d/<subject>/*.c3d and force/force/<s>_force/<s>/*.npy
export TAMU_DATA_ROOT="/path/to/TAMU Data"         # contains 01.PatientHandlingTasks/03.OpenSim/subjNN/... and Experiment Detail.xlsx
```

**GroundLink** is public. Cite: X. Han, B. Senderling, S. To, D. Kumar, E. Whiting, J. Saito. "GroundLink: A Dataset
Unifying Human Body Movement and Ground Reaction Dynamics." *SIGGRAPH Asia 2023 Conference Papers*, ACM, 2023.
doi:[10.1145/3610548.3618249](https://doi.org/10.1145/3610548.3618249). Obtain the C3D marker files and force `.npy`
files from the dataset's authors. Subject `s001` is excluded (force anomalies; `configs/groundlink.yaml`); 279 trials
from 6 subjects remain.

**Patient Handling** was collected at Texas A&M University under IRB protocol STUDY2024-0979 (approved October 1, 2024).
It is **not** included in this repository and may not be redistributed. For the data, see
Y. Chen, W. Yin, L. Zheng, R. Mehta, X. Zhang. "Biodynamic Modeling and Analysis of Human-Exoskeleton Interactions in
Simulated Patient Handling Tasks." *Human Factors* 67(7):641-655, 2025.
doi:[10.1177/00187208241311271](https://doi.org/10.1177/00187208241311271).
The code expects the OpenSim export layout: `subjNN/MarkerData/<trial>.trc` with
`subjNN/ForceData/ForceFile(50Hz)/<trial>_Resampled.mot`, and the participant sheet `Experiment Detail.xlsx` (mass,
height). `results/` contains Patient Handling results only as metrics under pseudonymous subject IDs (`ph_1` ...
`ph_12`); no captures, predictions or anthropometrics.

## Reproducing the paper

### Rebuilding the tables and figures (no data, no GPU)

`results/paper_v2_stage/` is the staged run-output folder used for the submission:

```bash
cd paper_build
python build_results.py      # -> RESULTS_CORRECTED.md, results.json
python make_numbers.py       # -> numbers.tex (every number quoted in the text)
python make_figures.py       # -> figures/fig_{inflation,frame,xlab,models,subjects}.pdf
python make_supplement.py    # -> supp_tables.tex
```

These reproduce the submission's `results.json`, `numbers.tex` and `supp_tables.tex` byte for byte. To build from your
own reruns, set `JBHI_CODE_ROOT=<repo>/outputs/paper_v2_stage`. `make_figures.py` also reads `newton_check.csv`, the
per-trial Newton's-law frame check, shipped precomputed.

### Retraining

The final numbers come from one run family, `v2`, trained on TAMU HPRC clusters with the Slurm arrays in
`chapter1/hprc/`. Each array index maps to one (dataset, model, seed or arm) cell, documented at the top of each file.
Set up the cluster environment once:

```bash
cp hprc/hprc.env.example hprc/hprc.env      # set HPRC_ACCOUNT and the scratch paths
bash hprc/setup_env.sh                      # on a login node: venv + torch 2.5.1 (cu124) + pinned requirements
mkdir -p runs/slurm                         # Slurm logs land here; submit from the repo root
```

| Paper element | Command (from the repo root on the cluster) | Output |
|---|---|---|
| Headline LOSO, GroundLink (4 configs x seeds 42-44) | `sbatch --array=0-11%12 --export=ALL,E1_OUT=outputs/v2/e1 chapter1/hprc/e1_pred.slurm` | `outputs/v2/e1/` |
| Headline LOSO, Patient Handling | `sbatch --array=12-23%12 chapter1/hprc/e1_pred.slurm` | `outputs/e1_pred/` |
| Trial-level split, GroundLink | `sbatch --array=0-11%12 --export=ALL,TS_OUT=outputs/v2/ts,TS_SPLIT_SEED=42 chapter1/hprc/ts_pred.slurm` | `outputs/v2/ts/` |
| Trial-level split, Patient Handling | `sbatch --array=12-23%12 chapter1/hprc/ts_pred.slurm` | `outputs/ts_pred/` |
| Preprocessing ablation (GL), ICF frame factorial (GL), per-model LOSO with cross-lab evaluation, `s001` sensitivity, pooled dual-holdout | `sbatch --array=0-22%12 chapter1/hprc/gl_family.slurm` | `outputs/v2/ablation_prep/`, `outputs/v2/ablation/`, `runs/v2/m6_*`, `outputs/v2_m6_combined_*` |
| Frame factorial for CNN-LSTM and CNN (GL and PH) | `sbatch --array=0-15%12 chapter1/hprc/frame_models.slurm` | `outputs/v2/ablation_{cnn-lstm,cnn}/` |
| Cross-lab transfer to every subject of the other lab (eval only; needs `gl_family` tasks 13-15 and 17-19) | `sbatch --array=0-5 chapter1/hprc/xlab_all.slurm` | `runs/v2/*_xlaball/` |
| Ridge baseline (CPU) | `SELECTED_DATASET=GROUNDLINK INPUT_NORM_MODE=per_fold E3_OUT=outputs/v2 python -u chapter1/experiments/e3_ridge_baseline.py --config configs/groundlink.yaml` (and `PATIENT_HANDLING` with `configs/patient_handling.yaml`) | `outputs/v2/e3_ridge_loso_{gl,ph}.csv` |

Without Slurm, run the same Python line with the same variables. For example, one cell of `e1_pred.slurm`:

```bash
SELECTED_DATASET=GROUNDLINK E1_CONFIGS=ICF-hpo E1_SEEDS=42 E1_EPOCHS=120 E1_OUT=outputs/v2/e1 \
E1_SAVE_PRED=outputs/v2/e1/pred GROUNDLINK_GRF_UNITS=kilonewton \
python -u chapter1/experiments/e1_multiseed_corrected.py
```

Several Patient Handling inputs predate the `v2` family; Patient Handling targets were not affected by the GroundLink
units fix, so they were reused:
- PH LOSO per-model runs (`runs/m6_*_ph`): `chapter1/pipeline/main.py` with `USE_LOSO=1`, `MODEL_NAME`,
  `LOSO_BEST_PARAMS=configs/best_params_*.json`, `INPUT_NORM_MODE=per_fold` and `RUN_NAME_OVERRIDE=m6_*`. These runs
  have no recorded launcher; the command is reconstructed from their logs.
- PH frame factorial for ICF (`outputs/ablation/ph_*.csv`): `chapter1/experiments/ablation_runner.py`.
- PH preprocessing ablation (`outputs/ablation_prep/ph_ablation_prep.csv`):
  `bash chapter1/experiments/run_experiments.sh ablation_seed42`, then `ablation_v2`.
- Standard trial-split runs of the original pipeline (`outputs/{gl,ph}_standard_*`):
  `USE_LOSO=0 python chapter1/pipeline/main.py --config configs/{groundlink,patient_handling}.yaml`.

After the arrays finish, copy the cluster's `runs/` and `outputs/` folders to `runs/hprc/` in the local clone, rescore
in biomechanical units and stage:

```bash
python chapter1/experiments/rescore_masked.py --pred runs/hprc/outputs/v2/e1/pred runs/hprc/outputs/e1_pred/pred \
       runs/hprc/outputs/v2/ts/pred runs/hprc/outputs/ts_pred/pred --out outputs/rescore_masked
python chapter1/experiments/biomech_units.py --pred runs/hprc/outputs/v2/e1/pred runs/hprc/outputs/e1_pred/pred \
       runs/hprc/outputs/v2/ts/pred runs/hprc/outputs/ts_pred/pred \
       --r2 "runs/hprc/outputs/v2/e1/*_multiseed.csv" "runs/hprc/outputs/e1_pred/*_multiseed.csv" \
            "runs/hprc/outputs/v2/ts/*_trial_split.csv" "runs/hprc/outputs/ts_pred/*_trial_split.csv" \
       --out outputs/biomech_units_v2
python chapter1/experiments/stage_v2_for_paper.py      # -> outputs/paper_v2_stage/
```

The `biomech_units.py` invocation is reconstructed from the output folder names; the exact command was not recorded.

### What `results/` leaves out

Model checkpoints, per-frame prediction files (they contain Patient Handling targets and subject anthropometrics), the
cached test tensors of the standard runs, training logs, and the subject mass and height columns of
`biomech_units_v2/trials.csv`. None of the build scripts read them.

## Known limitations

- `e2_matched_{gl,ph}.csv` (matched comparison) were assembled from the per-subject LOSO CSVs by hand; no script in
  this repository writes them.
- GPU training is not bit-for-bit deterministic. Reruns should agree with `results/` within seed variation.
- `src/` is shared with ongoing work by the same authors; dataset adapters and options that the paper does not use
  (for example a wind-turbine adapter and causal-model flags) are present but inactive with the shipped configs.

## Funding

This work was supported in part by the Ocean Energy Safety Institute Consortium (OESIC) through a grant from the U.S.
Department of the Interior, Bureau of Safety and Environmental Enforcement (BSEE), and the U.S. Department of Energy
(DOE), under Agreement Number E21AC00000. The views and conclusions contained in this repository are those of the
authors and should not be interpreted as representing the opinions or policies of the U.S. Government.

## License

MIT (see `LICENSE`). The datasets have their own terms; see Data.
