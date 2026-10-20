# -*- coding: utf-8 -*-
"""Coarse ICF architecture grid on the corrected ("both") pipeline.

Grid over d_model x num_encoder_layers (other HP fixed at the tuned values),
evaluated on a fast 3-fold GroundLink proxy spanning fold difficulty, at reduced
epochs, to find the best depth/width region now that the AP-frame bug is fixed.
Confirm the winner with full LOSO afterward. Tracks per-channel ML/AP too.

  python -u experiments/grid_arch.py
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.environ["SELECTED_DATASET"] = "GROUNDLINK"
os.environ["ROTATE_FORCE_VECTORS"] = "1"          # corrected pipeline
os.environ["INPUT_NORM_MODE"] = "per_fold"
os.environ["ENABLE_BIOMECH_PREPROCESSING"] = "1"

BASE = json.load(open("configs/best_params_icf.json"))   # nhead8, ff512, cnn256, dropout, lr, wd, bs16
D_MODELS = [64, 128, 256]
LAYERS = [2, 3, 4]
PROXY_FOLDS = ["gl_2", "gl_5", "gl_7"]                    # mid / hard / easy
EPOCHS = int(os.getenv("GRID_EPOCHS", "80"))
PATIENCE = int(os.getenv("GRID_PATIENCE", "12"))
SEED = 42
OUT = Path("outputs/grid_arch"); OUT.mkdir(parents=True, exist_ok=True)

sys.argv = ["x", "--config", "configs/groundlink.yaml"]
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
print(f"Grid on {device}: d_model={D_MODELS} x layers={LAYERS}, folds={PROXY_FOLDS}, epochs={EPOCHS}", flush=True)

rows = []
for dm in D_MODELS:
    for nl in LAYERS:
        hp = dict(BASE); hp["d_model"] = dm; hp["num_encoder_layers"] = nl
        bs = int(hp.get("batch_size", 16))
        for held in PROXY_FOLDS:
            t0 = time.time()
            torch.manual_seed(SEED); np.random.seed(SEED)
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
            model = build_model("ImprovedConvFormer", hp, isz, osz, mlen, device)
            npar = sum(p.numel() for p in model.parameters())
            crit = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=[1.0] * osz)
            opt = optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
            sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
            train_and_evaluate(model, DataLoader(tr, batch_size=bs, shuffle=True),
                               DataLoader(va, batch_size=16) if va else None, crit, opt, sch,
                               num_epochs=EPOCHS, device=device, early_stopping_patience=PATIENCE,
                               target_feature_names=tcols)
            preds, acts, msk = validate_model(model, DataLoader(te, batch_size=16), device)
            res = analyze_results(preds, acts, tcols, msk)
            per = {r["Feature"]: float(r["R2"]) for _, r in res.iterrows()}
            r2 = float(res["R2"].mean())
            ml = float(np.nanmean([v for k, v in per.items() if k.endswith("_ml")]))
            ap = float(np.nanmean([v for k, v in per.items() if k.endswith("_ap")]))
            rows.append(dict(d_model=dm, layers=nl, fold=held, r2=r2, ml=ml, ap=ap, params_M=round(npar / 1e6, 2)))
            print(f"[dm{dm} L{nl} {npar/1e6:.1f}M] {held}: R2={r2:.3f} ML={ml:.3f} AP={ap:.3f} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "grid_arch.csv", index=False)

df = pd.DataFrame(rows)
if len(df):
    g = df.groupby(["d_model", "layers"]).agg(r2=("r2", "mean"), ml=("ml", "mean"),
                                              ap=("ap", "mean"), params_M=("params_M", "first")).reset_index()
    print("\n==================== GRID SUMMARY (mean over proxy folds) ====================", flush=True)
    print(g.sort_values("r2", ascending=False).to_string(index=False), flush=True)
