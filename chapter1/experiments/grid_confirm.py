# -*- coding: utf-8 -*-
"""Full 6-fold LOSO confirmation of the top architecture-grid configs.

Runs the promising (d_model, num_encoder_layers) configs from grid_arch on the
full GroundLink LOSO at 120 epochs, corrected ("both") pipeline, and reports
per-subject + per-channel R2 (mean/ML/AP), vs the current tuned config.

  python -u experiments/grid_confirm.py
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
os.environ["ROTATE_FORCE_VECTORS"] = "1"
os.environ["INPUT_NORM_MODE"] = "per_fold"
os.environ["ENABLE_BIOMECH_PREPROCESSING"] = "1"

BASE = json.load(open("configs/best_params_icf.json"))
# (d_model, num_encoder_layers) — grid winners + current baseline for reference
CONFIGS = [(64, 2), (128, 2), (128, 4), (256, 3)]
EPOCHS = int(os.getenv("CONFIRM_EPOCHS", "120"))
PATIENCE = int(os.getenv("CONFIRM_PATIENCE", "15"))
SEED = 42
OUT = Path("outputs/grid_confirm"); OUT.mkdir(parents=True, exist_ok=True)

sys.argv = ["x", "--config", "configs/groundlink.yaml"]
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
print(f"Confirm on {device}: configs={CONFIGS}, folds={subjects}, epochs={EPOCHS}", flush=True)

rows = []
for dm, nl in CONFIGS:
    hp = dict(BASE); hp["d_model"] = dm; hp["num_encoder_layers"] = nl
    bs = int(hp.get("batch_size", 16))
    for held in subjects:
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
        crit = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=[1.0] * osz)
        opt = optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
        train_and_evaluate(model, DataLoader(tr, batch_size=bs, shuffle=True),
                           DataLoader(va, batch_size=16) if va else None, crit, opt, sch,
                           num_epochs=EPOCHS, device=device, early_stopping_patience=PATIENCE, target_feature_names=tcols)
        preds, acts, msk = validate_model(model, DataLoader(te, batch_size=16), device)
        res = analyze_results(preds, acts, tcols, msk)
        per = {r["Feature"]: float(r["R2"]) for _, r in res.iterrows()}
        r2 = float(res["R2"].mean())
        ml = float(np.nanmean([v for k, v in per.items() if k.endswith("_ml")]))
        ap = float(np.nanmean([v for k, v in per.items() if k.endswith("_ap")]))
        rows.append(dict(d_model=dm, layers=nl, subject=held, r2=r2, ml=ml, ap=ap))
        print(f"[dm{dm} L{nl}] {held}: R2={r2:.3f} ML={ml:.3f} AP={ap:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "grid_confirm.csv", index=False)

df = pd.DataFrame(rows)
if len(df):
    g = df.groupby(["d_model", "layers"]).agg(r2=("r2", "mean"), r2_std=("r2", "std"),
                                              ml=("ml", "mean"), ap=("ap", "mean")).reset_index()
    print("\n==================== CONFIRM SUMMARY (full 6-fold LOSO) ====================", flush=True)
    print(g.sort_values("r2", ascending=False).to_string(index=False), flush=True)
