# -*- coding: utf-8 -*-
"""
E3: Non-deep (ridge regression) LOSO baseline  [Reviewer ujzf request].

Fits a per-channel linear model (RidgeCV, alpha chosen by generalized CV on the
training frames only) on the SAME per-fold, within-fold-normalized features and
targets that the deep models consume, evaluated with the SAME LOSO folds and the
SAME per-channel R^2/MAE metric (src.train.analyze_results).

The ridge model is frame-wise / memoryless: it sees no temporal context. It
therefore isolates how much of the deep models' accuracy comes from temporal
modeling vs. from preprocessing + feature engineering.

Usage:
  SELECTED_DATASET=GROUNDLINK       python -u chapter1/experiments/e3_ridge_baseline.py --config configs/groundlink.yaml
  SELECTED_DATASET=PATIENT_HANDLING python -u chapter1/experiments/e3_ridge_baseline.py --config configs/patient_handling.yaml
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the repo root importable when run as `python chapter1/experiments/e3_ridge_baseline.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV

from src.pipeline.config_loader import load_config, setup_dataset_adapter
from src.pipeline.prepare_data import _prepare_data
from src.data.data_pipeline import prepare_dataset_for_modeling
from src.train import analyze_results
from src.utils.target_normalization import apply_target_normalizer, fit_target_normalizer
from src.utils.input_normalization import fit_input_normalizer, apply_input_normalizer
from src.utils.utils import extract_subject_id_from_filepath as extract_sid

ALPHAS = np.logspace(-2, 4, 13)


def main() -> None:
    cfg = load_config()
    adapter = setup_dataset_adapter(cfg)
    print(f"\n[E3 ridge] dataset={cfg.selected_dataset}")

    data = _prepare_data(cfg, adapter)
    prepare_args = data["prepare_args"]
    trc_files = data["trc_files"]
    use_metadata = data["use_metadata"]
    target_columns = cfg.target_columns
    tag = "ph" if cfg.selected_dataset == "PATIENT_HANDLING" else "gl"

    tgt_idx = 2 if use_metadata else 1
    msk_idx = 3 if use_metadata else 2
    feat_idx = 0

    subjects = list(dict.fromkeys(extract_sid(p) for p in trc_files))
    print(f"[E3 ridge] {len(subjects)} LOSO subjects: {subjects}")

    rows = []
    for held in subjects:
        try:
            ftr, fva, fte, fml, fis, fos, _ = prepare_dataset_for_modeling(
                **{**prepare_args, "held_out_subject": held}
            )
        except Exception as exc:
            print(f"  fold {held}: prep failed — {exc}")
            continue
        if not ftr[0] or not fte[0]:
            print(f"  fold {held}: empty train/test, skip")
            continue

        # Fit target normalizer on fold-train only, apply to train + test (as in loso_pipeline)
        norm = fit_target_normalizer(ftr[tgt_idx], ftr[msk_idx], target_columns)
        if norm is not None:
            ftr = apply_target_normalizer(ftr, use_metadata, norm)
            fte = apply_target_normalizer(fte, use_metadata, norm)

        # Per-fold input z-score, as e1_multiseed_corrected.py and loso_pipeline.py apply it: with
        # INPUT_NORM_MODE=per_fold the data pipeline leaves inputs unstandardized for the script to do.
        if os.getenv("INPUT_NORM_MODE", "per_trial") == "per_fold":
            inorm = fit_input_normalizer(ftr[feat_idx], ftr[msk_idx])
            ftr = tuple(apply_input_normalizer(ftr[feat_idx], ftr[msk_idx], inorm) if i == feat_idx else x for i, x in enumerate(ftr))
            fte = tuple(apply_input_normalizer(fte[feat_idx], fte[msk_idx], inorm) if i == feat_idx else x for i, x in enumerate(fte))

        Xtr = torch.stack(ftr[feat_idx]).numpy()
        Ytr = torch.stack(ftr[tgt_idx]).numpy()
        Mtr = torch.stack(ftr[msk_idx]).numpy()
        Xte = torch.stack(fte[feat_idx]).numpy()
        Yte = torch.stack(fte[tgt_idx]).numpy()
        Mte = torch.stack(fte[msk_idx]).numpy()

        # Fit on valid (non-padded) training frames only
        vtr = Mtr > 0
        Xtr_f = Xtr[vtr].astype(np.float64)
        Ytr_f = Ytr[vtr].astype(np.float64)

        model = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
        model.fit(Xtr_f, Ytr_f)

        N, T, D = Xte.shape
        pred = model.predict(Xte.reshape(N * T, D).astype(np.float64)).reshape(N, T, -1)

        res = analyze_results(pred, Yte, target_columns, Mte)
        r2 = float(res["R2"].mean())
        mae = float(res["MAE"].mean())
        row = {"subject": held, f"{tag}_r2": r2, f"{tag}_mae": mae,
               "alpha": float(np.mean(np.atleast_1d(model.alpha_)))}
        for _, rr in res.iterrows():
            row[f"{tag}_r2_{rr['Feature']}"] = float(rr["R2"])
        rows.append(row)
        print(f"  fold {held}: R2={r2:.4f}  MAE={mae:.4f}  (n_train_frames={len(Xtr_f)})")

    if not rows:
        print("[E3 ridge] no folds completed."); return

    df = pd.DataFrame(rows)
    out = Path(os.getenv("E3_OUT", "outputs")) / f"e3_ridge_loso_{tag}.csv"   # E3_OUT: new folder for a new run family
    out.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print("\n" + "=" * 60)
    print(f"E3 RIDGE LOSO [{tag.upper()}]  n={len(df)} folds")
    print(f"  mean R2  = {df[f'{tag}_r2'].mean():.4f} ± {df[f'{tag}_r2'].std():.4f}")
    print(f"  mean MAE = {df[f'{tag}_mae'].mean():.4f} ± {df[f'{tag}_mae'].std():.4f}")
    print(f"  saved -> {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
