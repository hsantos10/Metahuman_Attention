# -*- coding: utf-8 -*-
"""E1c: multi-seed headline LOSO on the CORRECTED pipeline (force co-rotation +
per-fold input normalisation), using the per-model configs selected by smart HPO
on the corrected pipeline (experiments/hpo_confirm.py). Purpose: seed-level
uncertainty for the TBME headline table and the model ranking.

Pipeline flags: ROTATE_FORCE_VECTORS=1, INPUT_NORM_MODE=per_fold, biomech ON,
per-fold target norm ON. Identical training loop to hpo_confirm.py; only the
seed varies. Writes per-fold, per-channel R2 incrementally.

  SELECTED_DATASET=GROUNDLINK       python -u chapter1/experiments/e1_multiseed_corrected.py
  SELECTED_DATASET=PATIENT_HANDLING python -u chapter1/experiments/e1_multiseed_corrected.py
Env knobs: E1_SEEDS (default "42,43,44"), E1_EPOCHS (default 120),
           E1_CONFIGS (comma list; default all four),
           E1_OUT (output folder, default outputs/e1_multiseed_corrected),
           E1_TAG (suffix on the CSV name so array tasks never share a file),
           E1_SAVE_PRED (folder: also write per-fold .npz with per-frame predictions and targets in
                         normalised units plus the fold's target mean/std, trial names, [mass, height]
                         metadata and the force unit; read by chapter1/experiments/biomech_units.py).
"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DATASET = os.getenv("SELECTED_DATASET", "GROUNDLINK")
CFG = {"GROUNDLINK": ("configs/groundlink.yaml", "gl"),
       "PATIENT_HANDLING": ("configs/patient_handling.yaml", "ph")}[DATASET]
config_path, tag = CFG
SEEDS = [int(s) for s in os.getenv("E1_SEEDS", "42,43,44").split(",") if s.strip()]
EPOCHS = int(os.getenv("E1_EPOCHS", "120"))
ONLY = [c for c in os.getenv("E1_CONFIGS", "").split(",") if c.strip()]

os.environ["SELECTED_DATASET"] = DATASET
os.environ["ROTATE_FORCE_VECTORS"] = "1"        # corrected pipeline
os.environ["INPUT_NORM_MODE"] = "per_fold"
os.environ["ENABLE_BIOMECH_PREPROCESSING"] = "1"
sys.argv = ["x", "--config", config_path]

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
if ONLY:
    CONFIGS = [c for c in CONFIGS if c[0] in ONLY]

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
import src.data.data_pipeline as dp
OUT = Path(os.getenv("E1_OUT", "outputs/e1_multiseed_corrected")); OUT.mkdir(parents=True, exist_ok=True)
SAVE_PRED = os.getenv("E1_SAVE_PRED", "").strip()
out_csv = OUT / f"{tag}{os.getenv('E1_TAG', '')}_multiseed.csv"
json.dump({n: hp for n, _, hp in CONFIGS}, open(OUT / f"{tag}_configs.json", "w"), indent=1)

# resume: skip (config, seed, subject) already on disk
rows = []
done = set()
if out_csv.exists():
    prev = pd.read_csv(out_csv)
    rows = prev.to_dict("records")
    done = set(zip(prev["config"], prev["seed"], prev["subject"]))
print(f"[{DATASET}] E1c multi-seed CORRECTED | {len(CONFIGS)} configs x {len(subjects)} folds x "
      f"{len(SEEDS)} seeds {SEEDS}, {EPOCHS} epochs | resuming {len(done)} done", flush=True)

for name, mtype, hp in CONFIGS:
    bs = int(hp.get("batch_size", 16))
    for seed in SEEDS:
        for held in subjects:
            if (name, seed, held) in done:
                continue
            torch.manual_seed(seed); np.random.seed(seed)
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
            n_params = sum(p.numel() for p in model.parameters())
            crit = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=[1.0] * osz)
            opt = optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
            sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
            train_and_evaluate(model, DataLoader(tr, batch_size=bs, shuffle=True),
                               DataLoader(va, batch_size=16) if va else None, crit, opt, sch,
                               num_epochs=EPOCHS, device=device, early_stopping_patience=15, target_feature_names=tcols)
            preds, acts, msk = validate_model(model, DataLoader(te, batch_size=16), device)
            if SAVE_PRED:
                sp = Path(SAVE_PRED); sp.mkdir(parents=True, exist_ok=True)
                test_paths = dp.LAST_KEPT_PATHS.get("Test", [])
                if len(test_paths) != len(preds):
                    print(f"[{name} seed{seed}] {held}: {len(test_paths)} kept test paths != {len(preds)} predictions; trial names not saved", flush=True)
                    test_paths = []
                np.savez_compressed(sp / f"{tag}_loso_{name}_s{seed}_{held}.npz",
                                    preds=preds, acts=acts, mask=msk,
                                    meta=np.stack([m.numpy() for m in fte[1]]),            # [mass kg, height m] per trial
                                    trials=np.array([Path(p).stem for p in test_paths]),
                                    subjects=np.array([esid(p) for p in test_paths]),
                                    mean=np.asarray(tn["mean"]), std=np.asarray(tn["std"]), columns=np.array(tcols),
                                    force_mode=str(cfg.force_target_mode), dataset=tag, protocol="loso",
                                    config=name, model=mtype, seed=seed, held_out=held)
            res = analyze_results(preds, acts, tcols, msk)
            per_r2 = {f"r2_{r['Feature']}": float(r["R2"]) for _, r in res.iterrows()}
            per_mae = {f"mae_{r['Feature']}": float(r["MAE"]) for _, r in res.iterrows()} if "MAE" in res.columns else {}
            r2 = float(res["R2"].mean())
            ml = float(np.nanmean([v for k, v in per_r2.items() if k.endswith("_ml")]))
            ap = float(np.nanmean([v for k, v in per_r2.items() if k.endswith("_ap")]))
            vert = float(np.nanmean([v for k, v in per_r2.items() if k.endswith("_vert")]))
            rows.append(dict(config=name, model=mtype, dataset=tag, seed=seed, subject=held, params=n_params,
                             r2=r2, vert=vert, ml=ml, ap=ap, **per_r2, **per_mae))
            print(f"[{name} seed{seed}] {held}: R2={r2:.3f} V={vert:.3f} ML={ml:.3f} AP={ap:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

df = pd.DataFrame(rows)
if len(df):
    per_seed = df.groupby(["config", "seed"])["r2"].mean().reset_index()
    g = per_seed.groupby("config")["r2"].agg(["mean", "std", "count"]).reset_index()
    print(f"\n==================== E1c [{DATASET}] mean fold-R2 across seeds (corrected pipeline) ====================", flush=True)
    print(g.sort_values("mean", ascending=False).to_string(index=False), flush=True)
