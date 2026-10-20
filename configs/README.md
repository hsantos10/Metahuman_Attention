# configs/

Dataset YAMLs (passed with `--config <file>`) and the tuned hyperparameters the runs loaded.

| File | Used by |
|---|---|
| `groundlink.yaml`, `patient_handling.yaml` | `chapter1/pipeline/main.py` standard runs; every `chapter1/experiments/*.py` loads one of them |
| `loso_crosslab.yaml`, `ph_loso_crosslab.yaml` | LOSO with cross-laboratory evaluation (`USE_LOSO=1`) |
| `combined_dual_holdout.yaml` | `chapter1/pipeline/combined_hpo_cv.py` (pooled two-laboratory training) |
| `best_params_{icf, cnn_gl, cnn_ph, cnn-lstm_gl, cnn-lstm_ph}.json` | Optuna winners; read by `combined_hpo_cv.py --best_params` and by `main.py` through `LOSO_BEST_PARAMS` |

**How YAML is applied** (`src/pipeline/config_loader.py`): each known key is copied into an environment variable only
if that variable is still unset, so the order is defaults, then YAML, then `.env`, then the shell (later wins).
Several YAML keys are documentation that the code does not read: `dataset.root`, `dataset.metadata`, `markers.set`,
`processing.marker_units`, `processing.nan_threshold`, `targets.columns`. Data paths come from `GROUNDLINK_ROOT` and
`TAMU_DATA_ROOT`, and the target layout from `TARGET_LAYOUT`.
