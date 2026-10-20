# -*- coding: utf-8 -*-
"""Preprocessing / training-choice ablation for the TBME supplement (reviewer ujzf #6).
ICF at the fixed paper configuration (d64/L2), CORRECTED pipeline (force co-rotation +
per-fold input norm), seed 42, full LOSO on GL and PH. One factor removed per arm:

  full         corrected pipeline, all features, flip on, metadata on, peak weight 2
  no_biomech   pelvis rooting + heading alignment + height scaling off (co-rotation then
               has no heading to use, so targets stay in the plate frame)
  no_derived   velocity / acceleration features off (FEATURES_DERIVED=0)
  no_rel       relative-position features off (FEATURES_REL=0)
  no_flip      bilateral flip augmentation off
  no_metadata  mass/height metadata off (zeros)
  no_peak      peak-weight multiplier 0 (plain MSE)

  SELECTED_DATASET=GROUNDLINK python -u chapter1/experiments/ablation_prep.py
Env: ABL_ARMS (comma list), ABL_EPOCHS (120), ABL_SEED (42). Resumes from the CSV.
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
EPOCHS = int(os.getenv("ABL_EPOCHS", "120"))
SEEDS = [int(s) for s in os.getenv("ABL_SEEDS", os.getenv("ABL_SEED", "42")).split(",") if s.strip()]
# no_pelvis / no_heading / no_height split the biomechanics block (added 2026-09-22); no_metadata is
# kept in the list for completeness but is NOT a valid arm under body-weight target normalisation
# (mass is needed for the targets) -- it produced zero folds on 2026-09-21.
ARMS_ALL = ["full", "no_biomech", "no_pelvis", "no_heading", "no_height", "no_derived", "no_rel", "no_flip", "no_peak"]
ARMS = [a for a in os.getenv("ABL_ARMS", ",".join(ARMS_ALL)).split(",") if a.strip()]
HP = dict(d_model=64, nhead=8, num_encoder_layers=2, dim_feedforward=512, dropout=0.137,
          num_cnn_filters=256, learning_rate=4.6e-4, weight_decay=7.36e-4, batch_size=16)

os.environ["SELECTED_DATASET"] = DATASET
os.environ["ROTATE_FORCE_VECTORS"] = "1"
os.environ["INPUT_NORM_MODE"] = "per_fold"
os.environ["ENABLE_BIOMECH_PREPROCESSING"] = "1"
sys.argv = ["x", "--config", config_path]

from src.utils.utils import extract_subject_id_from_filepath as esid
from src.utils.target_normalization import fit_target_normalizer, apply_target_normalizer
from src.utils.input_normalization import fit_input_normalizer, apply_input_normalizer
from src.utils.main_script_helpers import ensure_metadata_split_tuple
from src.train import train_and_evaluate, validate_model, analyze_results
from src.loss import NormalizedPeakWeightedLoss
from src.pipeline.model_factory import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(os.getenv("ABL_OUT", "outputs/ablation_prep")); OUT.mkdir(parents=True, exist_ok=True)   # ABL_OUT: new folder for a new run family
out_csv = OUT / f"{tag}_ablation_prep.csv"
rows, done = [], set()
if out_csv.exists():
    prev = pd.read_csv(out_csv); rows = prev.to_dict("records"); done = set(zip(prev["arm"], prev["subject"], prev["seed"].astype(int)))

def load_data():
    """(Re)load config + trials with the current env switches (features are computed at load time)."""
    for k in [m for m in list(sys.modules) if m.startswith("src.data") or m.startswith("src.features") or m.startswith("src.pipeline.prepare_data") or m.startswith("src.pipeline.config_loader")]:
        del sys.modules[k]
    from src.pipeline.config_loader import load_config, setup_dataset_adapter
    from src.pipeline.prepare_data import _prepare_data
    from src.data.data_pipeline import prepare_dataset_for_modeling
    cfg = load_config(); adapter = setup_dataset_adapter(cfg)
    data = _prepare_data(cfg, adapter)
    return cfg, dict(data["prepare_args"]), prepare_dataset_for_modeling

print(f"[{DATASET}] ablation_prep | arms {ARMS} | seeds {SEEDS} | {EPOCHS} epochs | resuming {len(done)} done", flush=True)
for arm in ARMS:
    os.environ["FEATURES_DERIVED"] = "0" if arm == "no_derived" else "1"
    os.environ["FEATURES_REL"] = "0" if arm == "no_rel" else "1"
    os.environ["BIOMECH_PELVIS"] = "0" if arm == "no_pelvis" else "1"
    os.environ["BIOMECH_HEADING"] = "0" if arm == "no_heading" else "1"
    os.environ["BIOMECH_HEIGHT"] = "0" if arm == "no_height" else "1"
    cfg, pa, prepare = load_data()
    tcols = list(cfg.target_columns)
    subjects = list(dict.fromkeys(esid(p) for p in pa["trc_file_paths"]))
    if arm == "no_biomech": pa["enable_biomech_preprocessing"] = False
    if arm == "no_flip": pa["use_bilateral_flip"] = False
    if arm == "no_metadata": pa["use_metadata"] = False
    peak = 0.0 if arm == "no_peak" else 2.0
    for SEED in SEEDS:
      for held in subjects:
        if (arm, held, SEED) in done: continue
        torch.manual_seed(SEED); np.random.seed(SEED)
        ftr, fva, fte, mlen, isz, osz, _ = prepare(**{**pa, "held_out_subject": held})
        if not ftr[0] or not fte[0]: continue
        tn = fit_target_normalizer(ftr[2], ftr[3], tcols)
        ftr = apply_target_normalizer(ftr, True, tn); fva = apply_target_normalizer(fva, True, tn) if fva[0] else fva; fte = apply_target_normalizer(fte, True, tn)
        inorm = fit_input_normalizer(ftr[0], ftr[3])
        ftr = (apply_input_normalizer(ftr[0], ftr[3], inorm),) + tuple(ftr[1:])
        if fva[0]: fva = (apply_input_normalizer(fva[0], fva[3], inorm),) + tuple(fva[1:])
        fte = (apply_input_normalizer(fte[0], fte[3], inorm),) + tuple(fte[1:])
        ftr = ensure_metadata_split_tuple(ftr, 2); fva = ensure_metadata_split_tuple(fva, 2); fte = ensure_metadata_split_tuple(fte, 2)
        tr = TensorDataset(*(torch.stack(t) for t in ftr)); va = TensorDataset(*(torch.stack(t) for t in fva)) if fva[0] else None; te = TensorDataset(*(torch.stack(t) for t in fte))
        model = build_model("ImprovedConvFormer", HP, isz, osz, mlen, device)
        crit = NormalizedPeakWeightedLoss(peak_weight_multiplier=peak, feature_weights=[1.0] * osz)
        opt = optim.AdamW(model.parameters(), lr=HP["learning_rate"], weight_decay=HP["weight_decay"])
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
        train_and_evaluate(model, DataLoader(tr, batch_size=HP["batch_size"], shuffle=True), DataLoader(va, batch_size=16) if va else None, crit, opt, sch,
                           num_epochs=EPOCHS, device=device, early_stopping_patience=15, target_feature_names=tcols)
        preds, acts, msk = validate_model(model, DataLoader(te, batch_size=16), device)
        res = analyze_results(preds, acts, tcols, msk)
        per = {f"r2_{r['Feature']}": float(r["R2"]) for _, r in res.iterrows()}
        r2 = float(res["R2"].mean())
        rows.append(dict(arm=arm, dataset=tag, subject=held, seed=SEED, n_features=int(isz), r2=r2,
                         vert=np.nanmean([v for k, v in per.items() if k.endswith("_vert")]), ml=np.nanmean([v for k, v in per.items() if k.endswith("_ml")]),
                         ap=np.nanmean([v for k, v in per.items() if k.endswith("_ap")]), **per))
        print(f"[{arm}] seed{SEED} {held}: R2={r2:.3f} (D={isz})", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

df = pd.DataFrame(rows)
if len(df):
    print(f"\n==== ablation_prep [{DATASET}] mean LOSO R2 by arm (all seeds pooled) ====", flush=True)
    print(df.groupby("arm")[["r2", "vert", "ml", "ap"]].mean().round(3).to_string(), flush=True)
    fm = df.groupby(["arm", "seed"])["r2"].mean().unstack("seed").round(3)
    print(f"\n==== ablation_prep [{DATASET}] fold-mean R2 per seed ====", flush=True)
    print(fm.to_string(), flush=True)
