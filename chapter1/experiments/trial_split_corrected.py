# -*- coding: utf-8 -*-
"""Trial-level random split (80/10/10 by trial, subjects mixed across splits) on the CORRECTED
pipeline with the same per-model configurations as the LOSO headline (hpo_confirm / e1c), so that
the leakage inflation is measured on one pipeline with one configuration per model.
Writes outputs/trial_split_corrected/{gl,ph}_trial_split.csv (one row per model, per-channel R2).

  SELECTED_DATASET=GROUNDLINK       python -u chapter1/experiments/trial_split_corrected.py
  SELECTED_DATASET=PATIENT_HANDLING python -u chapter1/experiments/trial_split_corrected.py
Env: TS_SEED (42), TS_SPLIT_SEED (defaults to TS_SEED; set it to fix the split across training seeds), TS_EPOCHS (120), TS_CONFIGS (comma list),
     TS_OUT (output folder, default outputs/trial_split_corrected), TS_TAG (CSV name suffix for array tasks),
     TS_SAVE_PRED (folder: per-model .npz with per-frame predictions, targets, trial names, subjects,
                   [mass, height] metadata and the fold's target mean/std; see biomech_units.py).
The resume key is (config, seed), so one CSV can hold several seeds.
"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DATASET = os.getenv("SELECTED_DATASET", "GROUNDLINK")
config_path, tag = {"GROUNDLINK": ("configs/groundlink.yaml", "gl"),
                    "PATIENT_HANDLING": ("configs/patient_handling.yaml", "ph")}[DATASET]
SEED = int(os.getenv("TS_SEED", "42")); EPOCHS = int(os.getenv("TS_EPOCHS", "120"))
ONLY = [c for c in os.getenv("TS_CONFIGS", "").split(",") if c.strip()]
os.environ.update(SELECTED_DATASET=DATASET, ROTATE_FORCE_VECTORS="1", INPUT_NORM_MODE="per_fold", ENABLE_BIOMECH_PREPROCESSING="1")
sys.argv = ["x", "--config", config_path]

ICF_GRID = dict(d_model=128, num_encoder_layers=2, dim_feedforward=512, num_cnn_filters=256, dropout=0.137, learning_rate=0.00046, weight_decay=0.000736, batch_size=16, nhead=8)
CONFIGS = {
    "GROUNDLINK": [
        ("ICF-hpo", "ImprovedConvFormer", dict(d_model=96, num_encoder_layers=2, dim_feedforward=1024, num_cnn_filters=128, dropout=0.2872, learning_rate=0.002669, weight_decay=0.007247, batch_size=16, nhead=8)),
        ("ICF-grid128", "ImprovedConvFormer", ICF_GRID),
        ("CNN-LSTM-hpo", "CNN_LSTM", dict(hidden_size=512, num_layers=2, num_filters=256, kernel_size=7, dropout=0.1057, learning_rate=0.0006745, weight_decay=0.010448, batch_size=16)),
        ("CNN-hpo", "CNN", dict(num_filters=512, kernel_size=7, dropout=0.3316, learning_rate=0.0005248, weight_decay=0.000483, batch_size=16)),
    ],
    "PATIENT_HANDLING": [
        ("ICF-hpo", "ImprovedConvFormer", dict(d_model=96, num_encoder_layers=2, dim_feedforward=512, num_cnn_filters=256, dropout=0.08657, learning_rate=0.0007289, weight_decay=0.0011419, batch_size=16, nhead=8)),
        ("ICF-grid128", "ImprovedConvFormer", ICF_GRID),
        ("CNN-LSTM-hpo", "CNN_LSTM", dict(hidden_size=256, num_layers=1, num_filters=256, kernel_size=7, dropout=0.3213, learning_rate=0.003784, weight_decay=0.033135, batch_size=32)),
        ("CNN-hpo", "CNN", dict(num_filters=256, kernel_size=15, dropout=0.3557, learning_rate=0.0008636, weight_decay=0.003897, batch_size=16)),
    ],
}[DATASET]
if ONLY: CONFIGS = [c for c in CONFIGS if c[0] in ONLY]

from src.pipeline.config_loader import load_config, setup_dataset_adapter
from src.pipeline.prepare_data import _prepare_data
from src.data.data_pipeline import prepare_dataset_for_modeling
from src.utils.target_normalization import fit_target_normalizer, apply_target_normalizer
from src.utils.input_normalization import fit_input_normalizer, apply_input_normalizer
from src.utils.main_script_helpers import ensure_metadata_split_tuple
from src.utils.utils import extract_subject_id_from_filepath as esid
from src.train import train_and_evaluate, validate_model, analyze_results
from src.loss import NormalizedPeakWeightedLoss
from src.pipeline.model_factory import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = load_config(); adapter = setup_dataset_adapter(cfg); tcols = cfg.target_columns
data = _prepare_data(cfg, adapter); pa = dict(data["prepare_args"])
# TS_SPLIT_SEED fixes the trial draw while TS_SEED varies the training seed, so seeds measure training noise and
# not test-set variance (the published seed-42 row drew the split with SEED; default keeps that behaviour)
pa["held_out_subject"] = None; pa["test_size"] = 0.1; pa["val_size"] = 0.1; pa["random_state"] = int(os.getenv("TS_SPLIT_SEED", str(SEED)))
import src.data.data_pipeline as dp
OUT = Path(os.getenv("TS_OUT", "outputs/trial_split_corrected")); OUT.mkdir(parents=True, exist_ok=True)
SAVE_PRED = os.getenv("TS_SAVE_PRED", "").strip()
out_csv = OUT / f"{tag}{os.getenv('TS_TAG', '')}_trial_split.csv"
rows, done = [], set()
if out_csv.exists():
    prev = pd.read_csv(out_csv); rows = prev.to_dict("records"); done = set(zip(prev["config"], prev["seed"]))

torch.manual_seed(SEED); np.random.seed(SEED)
ftr, fva, fte, mlen, isz, osz, _ = prepare_dataset_for_modeling(**pa)
test_paths = list(dp.LAST_KEPT_PATHS.get("Test", []))
print(f"[{DATASET}] trial split: train {len(ftr[0])} val {len(fva[0]) if fva[0] else 0} test {len(fte[0])} trials (flip doubles train)", flush=True)
tn = fit_target_normalizer(ftr[2], ftr[3], tcols)
ftr = apply_target_normalizer(ftr, True, tn); fva = apply_target_normalizer(fva, True, tn) if fva[0] else fva; fte = apply_target_normalizer(fte, True, tn)
inorm = fit_input_normalizer(ftr[0], ftr[3])
ftr = (apply_input_normalizer(ftr[0], ftr[3], inorm),) + tuple(ftr[1:])
if fva[0]: fva = (apply_input_normalizer(fva[0], fva[3], inorm),) + tuple(fva[1:])
fte = (apply_input_normalizer(fte[0], fte[3], inorm),) + tuple(fte[1:])
ftr = ensure_metadata_split_tuple(ftr, 2); fva = ensure_metadata_split_tuple(fva, 2); fte = ensure_metadata_split_tuple(fte, 2)
tr = TensorDataset(*(torch.stack(t) for t in ftr)); va = TensorDataset(*(torch.stack(t) for t in fva)) if fva[0] else None; te = TensorDataset(*(torch.stack(t) for t in fte))

for name, mtype, hp in CONFIGS:
    if (name, SEED) in done: continue
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = build_model(mtype, hp, isz, osz, mlen, device)
    crit = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=[1.0] * osz)
    opt = optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    train_and_evaluate(model, DataLoader(tr, batch_size=int(hp.get("batch_size", 16)), shuffle=True), DataLoader(va, batch_size=16) if va else None,
                       crit, opt, sch, num_epochs=EPOCHS, device=device, early_stopping_patience=15, target_feature_names=tcols)
    preds, acts, msk = validate_model(model, DataLoader(te, batch_size=16), device)
    if SAVE_PRED:
        sp = Path(SAVE_PRED); sp.mkdir(parents=True, exist_ok=True)
        tp = test_paths if len(test_paths) == len(preds) else []
        if not tp: print(f"[{name}] {len(test_paths)} kept test paths != {len(preds)} predictions; trial names not saved", flush=True)
        np.savez_compressed(sp / f"{tag}_trialsplit_{name}_s{SEED}.npz",
                            preds=preds, acts=acts, mask=msk, meta=np.stack([m.numpy() for m in fte[1]]),
                            trials=np.array([Path(p).stem for p in tp]), subjects=np.array([esid(p) for p in tp]),
                            mean=np.asarray(tn["mean"]), std=np.asarray(tn["std"]), columns=np.array(tcols),
                            force_mode=str(cfg.force_target_mode), dataset=tag, protocol="trial_split",
                            config=name, model=mtype, seed=SEED, held_out="")
    res = analyze_results(preds, acts, tcols, msk)
    per = {f"r2_{r['Feature']}": float(r["R2"]) for _, r in res.iterrows()}
    r2 = float(res["R2"].mean())
    rows.append(dict(config=name, model=mtype, dataset=tag, seed=SEED, protocol="trial_split_80_10_10", n_test_trials=len(fte[0]), r2=r2,
                     vert=np.nanmean([v for k, v in per.items() if k.endswith("_vert")]), ml=np.nanmean([v for k, v in per.items() if k.endswith("_ml")]),
                     ap=np.nanmean([v for k, v in per.items() if k.endswith("_ap")]), **per))
    print(f"[{name}] trial-split test R2={r2:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
print(pd.DataFrame(rows)[["config", "r2", "vert", "ml", "ap"]].round(3).to_string(index=False), flush=True)
