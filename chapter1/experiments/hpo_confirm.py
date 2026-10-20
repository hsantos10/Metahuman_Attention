# -*- coding: utf-8 -*-
"""Full-LOSO confirmation of the smart-HPO winners (+ grid's robust ICF 128/L2),
on the corrected pipeline, for a fair re-tuned model ranking.

  SELECTED_DATASET=GROUNDLINK python -u experiments/hpo_confirm.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DATASET = os.getenv("SELECTED_DATASET", "GROUNDLINK")
CFG = {"GROUNDLINK": ("configs/groundlink.yaml", "gl"),
       "PATIENT_HANDLING": ("configs/patient_handling.yaml", "ph")}[DATASET]
config_path, tag = CFG
EPOCHS = int(os.getenv("CONFIRM_EPOCHS", "120"))

os.environ["SELECTED_DATASET"] = DATASET
os.environ["ROTATE_FORCE_VECTORS"] = "1"
os.environ["INPUT_NORM_MODE"] = "per_fold"
os.environ["ENABLE_BIOMECH_PREPROCESSING"] = "1"
sys.argv = ["x", "--config", config_path]

# (name, model_type, params) — smart-HPO winners + grid ICF 128/L2 as robustness check
ICF_GRID = dict(d_model=128, num_encoder_layers=2, dim_feedforward=512, num_cnn_filters=256,
                dropout=0.137, learning_rate=0.00046, weight_decay=0.000736, batch_size=16, nhead=8)
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
cfg = load_config(); adapter = setup_dataset_adapter(cfg)
tcols = cfg.target_columns
data = _prepare_data(cfg, adapter); pa = data["prepare_args"]
subjects = list(dict.fromkeys(esid(p) for p in pa["trc_file_paths"]))
OUT = Path("outputs/hpo_confirm"); OUT.mkdir(parents=True, exist_ok=True)
print(f"[{DATASET}] confirm {len(CONFIGS)} configs x {len(subjects)} folds, {EPOCHS} epochs", flush=True)

rows = []
for name, mtype, hp in CONFIGS:
    bs = int(hp.get("batch_size", 16))
    for held in subjects:
        torch.manual_seed(42); np.random.seed(42)
        ftr, fva, fte, mlen, isz, osz, _ = prepare_dataset_for_modeling(**{**pa, "held_out_subject": held})
        if not ftr[0] or not fte[0]:
            continue
        tn = fit_target_normalizer(ftr[2], ftr[3], tcols)
        ftr = apply_target_normalizer(ftr, True, tn)
        fva = apply_target_normalizer(fva, True, tn) if fva[0] else fva
        fte = apply_target_normalizer(fte, True, tn)
        inorm = fit_input_normalizer(ftr[0], ftr[3])
        ftr = (apply_input_normalizer(ftr[0], ftr[3], inorm),) + tuple(ftr[1:])
        if fva[0]:
            fva = (apply_input_normalizer(fva[0], fva[3], inorm),) + tuple(fva[1:])
        fte = (apply_input_normalizer(fte[0], fte[3], inorm),) + tuple(fte[1:])
        ftr = ensure_metadata_split_tuple(ftr, 2); fva = ensure_metadata_split_tuple(fva, 2); fte = ensure_metadata_split_tuple(fte, 2)
        tr = TensorDataset(*(torch.stack(t) for t in ftr))
        va = TensorDataset(*(torch.stack(t) for t in fva)) if fva[0] else None
        te = TensorDataset(*(torch.stack(t) for t in fte))
        model = build_model(mtype, hp, isz, osz, mlen, device)
        crit = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=[1.0] * osz)
        opt = optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
        train_and_evaluate(model, DataLoader(tr, batch_size=bs, shuffle=True),
                           DataLoader(va, batch_size=16) if va else None, crit, opt, sch,
                           num_epochs=EPOCHS, device=device, early_stopping_patience=15, target_feature_names=tcols)
        preds, acts, msk = validate_model(model, DataLoader(te, batch_size=16), device)
        res = analyze_results(preds, acts, tcols, msk)
        per = {r["Feature"]: float(r["R2"]) for _, r in res.iterrows()}
        r2 = float(res["R2"].mean())
        ml = float(np.nanmean([v for k, v in per.items() if k.endswith("_ml")]))
        ap = float(np.nanmean([v for k, v in per.items() if k.endswith("_ap")]))
        rows.append(dict(config=name, subject=held, r2=r2, ml=ml, ap=ap))
        print(f"[{name}] {held}: R2={r2:.3f} ML={ml:.3f} AP={ap:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"{tag}_confirm.csv", index=False)

df = pd.DataFrame(rows)
if len(df):
    g = df.groupby("config").agg(R2=("r2", "mean"), R2_std=("r2", "std"), ML=("ml", "mean"), AP=("ap", "mean")).reset_index()
    print(f"\n==================== HPO CONFIRM [{DATASET}] full LOSO ====================", flush=True)
    print(g.sort_values("R2", ascending=False).to_string(index=False), flush=True)
