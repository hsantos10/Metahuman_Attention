# -*- coding: utf-8 -*-
"""Smart HPO: Optuna TPE + Hyperband pruning + warm-start, per model, on the
corrected ("both") pipeline. Objective = maximize R2 on a 1-fold held-out proxy
(the LOSO generalization target); logs ML/AP per trial. Widened ICF space with
layers in {2,3,4} so L2 and L4 are both explored.

  SELECTED_DATASET=GROUNDLINK MODEL_NAME=ImprovedConvFormer python -u experiments/smart_hpo.py
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader, TensorDataset

DATASET = os.getenv("SELECTED_DATASET", "GROUNDLINK")
MODEL = os.getenv("MODEL_NAME", "ImprovedConvFormer")
N_TRIALS = int(os.getenv("HPO_TRIALS", "60"))
MAX_EPOCHS = int(os.getenv("HPO_MAX_EPOCHS", "60"))
CONFIG = {"GROUNDLINK": ("configs/groundlink.yaml", "gl", "gl_5"),
          "PATIENT_HANDLING": ("configs/patient_handling.yaml", "ph", "ph_12")}[DATASET]
config_path, tag, VAL_SUBJECT = CONFIG

os.environ["SELECTED_DATASET"] = DATASET
os.environ["ROTATE_FORCE_VECTORS"] = "1"          # corrected pipeline
os.environ["INPUT_NORM_MODE"] = "per_fold"
os.environ["ENABLE_BIOMECH_PREPROCESSING"] = "1"
sys.argv = ["x", "--config", config_path]

from src.pipeline.config_loader import load_config, setup_dataset_adapter
from src.pipeline.prepare_data import _prepare_data
from src.data.data_pipeline import prepare_dataset_for_modeling
from src.utils.target_normalization import fit_target_normalizer, apply_target_normalizer
from src.utils.input_normalization import fit_input_normalizer, apply_input_normalizer
from src.utils.main_script_helpers import ensure_metadata_split_tuple
from src.train import train_and_evaluate, validate_model, analyze_results
from src.loss import NormalizedPeakWeightedLoss
from src.pipeline.model_factory import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = load_config(); adapter = setup_dataset_adapter(cfg)
tcols = cfg.target_columns
data = _prepare_data(cfg, adapter); pa = data["prepare_args"]

# Build the 1-fold proxy ONCE: train = all but VAL_SUBJECT, val = VAL_SUBJECT.
ftr, fva, fte, MLEN, ISZ, OSZ, _ = prepare_dataset_for_modeling(**{**pa, "held_out_subject": VAL_SUBJECT})
tn = fit_target_normalizer(ftr[2], ftr[3], tcols)
ftr = apply_target_normalizer(ftr, True, tn); fte = apply_target_normalizer(fte, True, tn)
inorm = fit_input_normalizer(ftr[0], ftr[3])
ftr = (apply_input_normalizer(ftr[0], ftr[3], inorm),) + tuple(ftr[1:])
fte = (apply_input_normalizer(fte[0], fte[3], inorm),) + tuple(fte[1:])
ftr = ensure_metadata_split_tuple(ftr, 2); fte = ensure_metadata_split_tuple(fte, 2)
TR = TensorDataset(*(torch.stack(t) for t in ftr))
VL = TensorDataset(*(torch.stack(t) for t in fte))
print(f"[{DATASET}/{MODEL}] proxy: train={len(ftr[0])} trials, val={VAL_SUBJECT} ({len(fte[0])} trials), "
      f"input={ISZ}, trials={N_TRIALS}, max_epochs={MAX_EPOCHS}", flush=True)


def suggest(trial):
    if MODEL == "CNN":
        p = dict(num_filters=trial.suggest_categorical("num_filters", [128, 256, 512]),
                 kernel_size=trial.suggest_categorical("kernel_size", [3, 7, 15]),
                 dropout=trial.suggest_float("dropout", 0.05, 0.40))
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    elif MODEL == "CNN_LSTM":
        p = dict(hidden_size=trial.suggest_categorical("hidden_size", [128, 256, 512]),
                 num_layers=trial.suggest_categorical("num_layers", [1, 2]),
                 num_filters=trial.suggest_categorical("num_filters", [128, 256]),
                 kernel_size=trial.suggest_categorical("kernel_size", [3, 7]),
                 dropout=trial.suggest_float("dropout", 0.05, 0.40))
        lr = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-3, 0.05, log=True)
    else:  # ImprovedConvFormer — widened, layers {2,3,4}
        p = dict(d_model=trial.suggest_categorical("d_model", [64, 96, 128, 160, 192, 256]),
                 num_encoder_layers=trial.suggest_categorical("num_encoder_layers", [2, 3, 4]),
                 dim_feedforward=trial.suggest_categorical("dim_feedforward", [256, 512, 1024]),
                 num_cnn_filters=trial.suggest_categorical("num_cnn_filters", [128, 256]),
                 dropout=trial.suggest_float("dropout", 0.05, 0.30), nhead=8)
        lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 2e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32])
    p.update(learning_rate=lr, weight_decay=wd, batch_size=bs)
    return p


def objective(trial):
    p = suggest(trial)
    bs = int(p["batch_size"])
    model = build_model(MODEL, p, ISZ, OSZ, MLEN, device)
    crit = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=[1.0] * OSZ)
    opt = optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    def cb(ep, val_r2):
        trial.report(val_r2, ep)
        if trial.should_prune():
            raise optuna.TrialPruned()

    train_and_evaluate(model, DataLoader(TR, batch_size=bs, shuffle=True),
                       DataLoader(VL, batch_size=16), crit, opt, sch,
                       num_epochs=MAX_EPOCHS, device=device, early_stopping_patience=12,
                       target_feature_names=tcols, on_epoch_end=cb)
    preds, acts, msk = validate_model(model, DataLoader(VL, batch_size=16), device)
    res = analyze_results(preds, acts, tcols, msk)
    per = {r["Feature"]: float(r["R2"]) for _, r in res.iterrows()}
    r2 = float(res["R2"].mean())
    trial.set_user_attr("ml", float(np.nanmean([v for k, v in per.items() if k.endswith("_ml")])))
    trial.set_user_attr("ap", float(np.nanmean([v for k, v in per.items() if k.endswith("_ap")])))
    trial.set_user_attr("params_M", round(sum(pp.numel() for pp in model.parameters()) / 1e6, 2))
    return r2


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8),
    pruner=optuna.pruners.HyperbandPruner(min_resource=8, max_resource=MAX_EPOCHS, reduction_factor=3),
)
# warm-start with known-good configs
WARM = {
    "ImprovedConvFormer": [
        dict(d_model=128, num_encoder_layers=2, dim_feedforward=512, num_cnn_filters=256, dropout=0.137, learning_rate=0.00046, weight_decay=0.000736, batch_size=16),
        dict(d_model=128, num_encoder_layers=4, dim_feedforward=512, num_cnn_filters=256, dropout=0.137, learning_rate=0.00046, weight_decay=0.000736, batch_size=16),
        dict(d_model=64,  num_encoder_layers=2, dim_feedforward=512, num_cnn_filters=256, dropout=0.137, learning_rate=0.00046, weight_decay=0.000736, batch_size=16),
    ],
}.get(MODEL, [])
for w in WARM:
    study.enqueue_trial(w)

t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
print(f"\n==================== SMART HPO DONE [{DATASET}/{MODEL}] {time.time()-t0:.0f}s ====================", flush=True)
print(f"best R2 = {study.best_value:.4f}", flush=True)
print(f"best params = {json.dumps(study.best_params)}", flush=True)
ba = study.best_trial.user_attrs
print(f"best trial ML={ba.get('ml'):.3f} AP={ba.get('ap'):.3f} params={ba.get('params_M')}M", flush=True)

OUT = Path("outputs/smart_hpo"); OUT.mkdir(parents=True, exist_ok=True)
df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
df.to_csv(OUT / f"{tag}_{MODEL}.csv", index=False)
n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
print(f"{len(study.trials)} trials ({n_pruned} pruned early) -> {OUT}/{tag}_{MODEL}.csv", flush=True)
# best config at each ICF depth (user asked for L2 vs L4)
if MODEL == "ImprovedConvFormer":
    comp = df[df.state == "COMPLETE"]
    for L in (2, 4):
        sub = comp[comp["params_num_encoder_layers"] == L] if "params_num_encoder_layers" in comp else comp.iloc[0:0]
        if len(sub):
            b = sub.loc[sub["value"].idxmax()]
            print(f"  best L{L}: R2={b['value']:.4f} d_model={int(b['params_d_model'])} "
                  f"ff={int(b['params_dim_feedforward'])} dropout={b['params_dropout']:.3f}", flush=True)
