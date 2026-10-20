# -*- coding: utf-8 -*-
"""
Scoped ablation: does force co-rotation (#1) and/or per-fold input normalization
(#2) improve ICF LOSO? 2x2 factorial, ICF only, GL + PH, fixed hyperparameters
(paper best config), same seed. Baseline arm reproduces the current pipeline.

Arms (env flags read inside src/data/data_pipeline.py at trial-load time):
  baseline : INPUT_NORM_MODE=per_trial  ROTATE_FORCE_VECTORS=0
  corot    : INPUT_NORM_MODE=per_trial  ROTATE_FORCE_VECTORS=1
  perfold  : INPUT_NORM_MODE=per_fold   ROTATE_FORCE_VECTORS=0
  both     : INPUT_NORM_MODE=per_fold   ROTATE_FORCE_VECTORS=1

Launch (single background job runs GL then PH, all 4 arms each):
  python -u scripts/ablation_runner.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

# Make the repo root importable when run as `python chapter1/experiments/ablation_runner.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

HP = dict(d_model=64, nhead=8, num_encoder_layers=2, dim_feedforward=512,
          dropout=0.137, num_cnn_filters=256, learning_rate=4.6e-4, weight_decay=7.36e-4)
BATCH = 16
EPOCHS = int(os.getenv("ABL_EPOCHS", "120"))
PATIENCE = int(os.getenv("ABL_PATIENCE", "15"))
SEED = 42
SMOKE = os.getenv("ABL_SMOKE", "0") == "1"   # 1 arm, 1 subject, GL only

ARMS = [
    ("baseline", "per_trial", "0"),
    ("corot",    "per_trial", "1"),
    ("perfold",  "per_fold",  "0"),
    ("both",     "per_fold",  "1"),
]
DATASETS = [("GROUNDLINK", "configs/groundlink.yaml", "gl"),
            ("PATIENT_HANDLING", "configs/patient_handling.yaml", "ph")]
# ABL_DATASETS / ABL_ARMS (comma lists) restrict a run, e.g. one cluster task per (dataset, arm);
# ABL_OUT points a new run family at its own folder (the CSVs are the resume state).
_want_ds = [d for d in os.getenv("ABL_DATASETS", "").split(",") if d.strip()]
_want_arms = [a for a in os.getenv("ABL_ARMS", "").split(",") if a.strip()]
if _want_ds:
    DATASETS = [d for d in DATASETS if d[0] in _want_ds]
if _want_arms:
    ARMS = [a for a in ARMS if a[0] in _want_arms]

OUT = Path(os.getenv("ABL_OUT", "outputs/ablation")); OUT.mkdir(parents=True, exist_ok=True)

# ABL_MODEL picks the network (default icf = the fixed ICF configuration of the paper's frame study).
# cnn-lstm / cnn use each dataset's tuned configuration, copied from e1_multiseed_corrected.py CONFIGS,
# so the factorial for those models matches their Table II rows.
ABL_MODEL = os.getenv("ABL_MODEL", "icf").lower()
MODEL_HP = {
    "icf": ("ImprovedConvFormer", {"gl": HP, "ph": HP}),
    "cnn-lstm": ("CNN_LSTM", {
        "gl": dict(hidden_size=512, num_layers=2, num_filters=256, kernel_size=7, dropout=0.1057, learning_rate=0.0006745, weight_decay=0.010448, batch_size=16),
        "ph": dict(hidden_size=256, num_layers=1, num_filters=256, kernel_size=7, dropout=0.3213, learning_rate=0.003784, weight_decay=0.033135, batch_size=32)}),
    "cnn": ("CNN", {
        "gl": dict(num_filters=512, kernel_size=7, dropout=0.3316, learning_rate=0.0005248, weight_decay=0.000483, batch_size=16),
        "ph": dict(num_filters=256, kernel_size=15, dropout=0.3557, learning_rate=0.0008636, weight_decay=0.003897, batch_size=16)}),
}
if ABL_MODEL not in MODEL_HP:
    raise SystemExit(f"ABL_MODEL={ABL_MODEL!r}: expected one of {sorted(MODEL_HP)}")


def fit_input_norm(feats, masks):
    chunks = []
    for f, m in zip(feats, masks):
        T = int(m.sum().item())
        if T > 0:
            chunks.append(f[:T].numpy())
    X = np.concatenate(chunks, 0)
    mean = X.mean(0).astype(np.float32)
    std = X.std(0).astype(np.float32); std[std < 1e-6] = 1.0
    return mean, std


def apply_input_norm(feats, masks, mean, std):
    mt, st = torch.tensor(mean), torch.tensor(std)
    out = []
    for f, m in zip(feats, masks):
        T = int(m.sum().item()); g = f.clone()
        if T > 0:
            g[:T] = (g[:T] - mt) / st
        out.append(g)
    return out


def run_dataset(dataset, config_path, tag, device):
    os.environ["SELECTED_DATASET"] = dataset
    sys.argv = ["ablation", "--config", config_path]
    from src.pipeline.config_loader import load_config, setup_dataset_adapter
    from src.pipeline.prepare_data import _prepare_data
    from src.data.data_pipeline import prepare_dataset_for_modeling
    from src.utils.target_normalization import fit_target_normalizer, apply_target_normalizer
    from src.utils.utils import extract_subject_id_from_filepath as esid
    from src.utils.main_script_helpers import ensure_metadata_split_tuple
    from src.train import train_and_evaluate, validate_model, analyze_results
    from src.loss import NormalizedPeakWeightedLoss
    from src.pipeline.model_factory import build_model

    cfg = load_config(); adapter = setup_dataset_adapter(cfg)
    tcols = cfg.target_columns
    data = _prepare_data(cfg, adapter)
    pa = data["prepare_args"]
    subjects = list(dict.fromkeys(esid(p) for p in pa["trc_file_paths"]))
    print(f"\n[{tag}] {len(subjects)} LOSO subjects: {subjects}", flush=True)

    arms = ARMS[:1] if SMOKE else ARMS
    if SMOKE:
        subjects = subjects[:1]
    for arm, in_mode, rotate in arms:
        os.environ["INPUT_NORM_MODE"] = in_mode
        os.environ["ROTATE_FORCE_VECTORS"] = rotate
        rows = []
        for held in subjects:
            t0 = time.time()
            torch.manual_seed(SEED); np.random.seed(SEED)
            try:
                ftr, fva, fte, mlen, isz, osz, _ = prepare_dataset_for_modeling(
                    **{**pa, "held_out_subject": held})
            except Exception as e:
                print(f"[{tag}/{arm}] {held}: prep failed {e}", flush=True); continue
            if not ftr[0] or not fte[0]:
                continue
            # per-fold target normalization (always)
            tn = fit_target_normalizer(ftr[2], ftr[3], tcols)
            if tn:
                ftr = apply_target_normalizer(ftr, True, tn)
                fva = apply_target_normalizer(fva, True, tn) if fva[0] else fva
                fte = apply_target_normalizer(fte, True, tn)
            # per-fold input normalization (only for per_fold arms)
            if in_mode == "per_fold":
                mean, std = fit_input_norm(ftr[0], ftr[3])
                ftr = (apply_input_norm(ftr[0], ftr[3], mean, std),) + tuple(ftr[1:])
                if fva[0]:
                    fva = (apply_input_norm(fva[0], fva[3], mean, std),) + tuple(fva[1:])
                fte = (apply_input_norm(fte[0], fte[3], mean, std),) + tuple(fte[1:])

            ftr = ensure_metadata_split_tuple(ftr, metadata_dim=2)
            fva = ensure_metadata_split_tuple(fva, metadata_dim=2)
            fte = ensure_metadata_split_tuple(fte, metadata_dim=2)
            tr_ds = TensorDataset(*(torch.stack(t) for t in ftr))
            va_ds = TensorDataset(*(torch.stack(t) for t in fva)) if fva[0] else None
            te_ds = TensorDataset(*(torch.stack(t) for t in fte))

            mname, hps = MODEL_HP[ABL_MODEL]; hp = hps[tag]; bs = int(hp.get("batch_size", BATCH))
            model = build_model(mname, hp, isz, osz, mlen, device)
            crit = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=[1.0]*osz)
            opt = optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
            sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
            tl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
            vl = DataLoader(va_ds, batch_size=bs, shuffle=False) if va_ds else None
            train_and_evaluate(model, tl, vl, crit, opt, sched, num_epochs=EPOCHS,
                               device=device, early_stopping_patience=PATIENCE,
                               target_feature_names=tcols)
            preds, acts, masks = validate_model(model, DataLoader(te_ds, batch_size=BATCH), device)
            res = analyze_results(preds, acts, tcols, masks)
            r2 = float(res["R2"].mean()); mae = float(res["MAE"].mean())
            row = {"subject": held, "r2": r2, "mae": mae}
            for _, rr in res.iterrows():
                row[f"r2_{rr['Feature']}"] = float(rr["R2"])
            rows.append(row)
            print(f"[{tag}/{arm}] {held}: R2={r2:.4f} MAE={mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / f"{tag}_{arm}.csv", index=False)  # incremental save
        if rows:
            m = np.mean([r["r2"] for r in rows])
            print(f"===> [{tag}/{arm}] mean R2 = {m:.4f} over {len(rows)} folds", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ablation on {device}. Model={ABL_MODEL} Arms={[a[0] for a in ARMS]} smoke={SMOKE}", flush=True)
    datasets = DATASETS[:1] if SMOKE else DATASETS
    for dataset, cfgp, tag in datasets:
        run_dataset(dataset, cfgp, tag, device)
    # summary
    print("\n\n==================== ABLATION SUMMARY ====================", flush=True)
    for _, _, tag in DATASETS:
        for arm, _, _ in ARMS:
            f = OUT / f"{tag}_{arm}.csv"
            if f.exists():
                d = pd.read_csv(f)
                print(f"  {tag:3s} {arm:9s}: mean R2 = {d['r2'].mean():.4f}  (n={len(d)})", flush=True)


if __name__ == "__main__":
    main()
