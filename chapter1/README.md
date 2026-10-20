# chapter1/

Code for the paper's experiments: GroundLink (locomotion, C3D) and Patient Handling (occupational tasks, TRC/MOT),
20 shared markers, 10 per-plate targets `ground_force_{1,2}_{ml,vert,ap,px,pz}`. Models: ImprovedConvFormer (ICF),
CNN-LSTM, CNN. Setup, data and the full reproduction recipe are in the top-level `README.md`.

Run everything from the repository root: scripts use relative `configs/` and `outputs/` paths, and they set their
pipeline flags (`ROTATE_FORCE_VECTORS`, `INPUT_NORM_MODE`, ...) **before** importing `src`, because those flags are read
when trials load. The "corrected" pipeline is `ROTATE_FORCE_VECTORS=1` (horizontal forces co-rotated into the heading
frame) plus `INPUT_NORM_MODE=per_fold` (input z-score fitted on the training fold).

## Layout

| Folder | What |
|---|---|
| `pipeline/` | `main.py` (env-driven: `USE_LOSO=1` runs `loso_pipeline.py`, otherwise `standard_pipeline.py`; `hpo.py` is the Optuna search) and `combined_hpo_cv.py` (GL + PH dual-holdout CV) |
| `analysis/` | Used by the pipelines: `post_analysis.py` (per-feature R², attention maps), `visualization.py`, `trial_ops.py` |
| `experiments/` | One script per paper experiment; the script name is the `outputs/` folder name. `run_experiments.sh` records the local commands as run |
| `hprc/` | Slurm arrays for the final (`v2`) runs; the index-to-cell mapping is at the top of each file |

## Which script produced which result

| Script | Output folder | In the paper |
|---|---|---|
| `experiments/e1_multiseed_corrected.py` (`hprc/e1_pred.slurm`) | `e1_multiseed_corrected` | subject-held-out R², mean ± SD over seeds |
| `experiments/trial_split_corrected.py` (`hprc/ts_pred.slurm`) | `trial_split_corrected` | trial-level split (leakage) |
| `experiments/ablation_runner.py` (`hprc/gl_family.slurm`, `hprc/frame_models.slurm`) | `ablation`, `ablation_cnn-lstm`, `ablation_cnn` | target-frame factorial |
| `experiments/ablation_prep.py` | `ablation_prep` | supplement: preprocessing ablation |
| `experiments/e3_ridge_baseline.py` | `e3_ridge_loso_*.csv` | ridge baseline |
| `pipeline/main.py` with `USE_LOSO=1` (`hprc/gl_family.slurm`, `hprc/xlab_all.slurm`) | `runs/m6_*` | per-model LOSO and cross-laboratory transfer |
| `pipeline/combined_hpo_cv.py` | `m6_combined_*` | pooled two-laboratory training |
| `experiments/rescore_masked.py`, `experiments/biomech_units.py` | `rescore_masked`, `biomech_units_v2` | metrics in %BW and cm |
| `experiments/hpo_confirm.py`, `smart_hpo.py`, `grid_arch.py`, `grid_confirm.py` | `hpo_confirm`, `smart_hpo`, `grid_*` | hyperparameter search behind the tuned configs |
| `experiments/stage_v2_for_paper.py` | `paper_v2_stage` | collects all of the above for `paper_build/` |

The five `configs/best_params_*.json` are the Optuna winners the per-model LOSO runs loaded. The experiment scripts
resume from their CSV (rows keyed by config/arm, subject, seed): for a from-scratch rerun, move `outputs/<name>/` aside.

## Loading checkpoints

LOSO fold files `runs/<run>/loso_models/fold_<subject>.pt` are plain `state_dict`s: build the model with
`src.pipeline.model_factory.build_model` and the JSON hyperparameters, then `load_state_dict`. Standard runs save a
dict with `model_state_dict` and `hyperparameters` (`pipeline/standard_pipeline.py`). Checkpoints are not distributed.
