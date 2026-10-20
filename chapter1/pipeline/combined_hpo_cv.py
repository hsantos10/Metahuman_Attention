# -*- coding: utf-8 -*-
"""
Combined (GL + PH) Dual-Holdout Cross-Validation with Optuna HPO.

Each fold holds out 1 GroundLink subject + 1 Patient Handling subject for
validation, then trains on the remaining combined data.  HPO is run once on
the first fold to find the best hyperparameters, which are re-used for all
subsequent folds.

Usage:
    python chapter1/pipeline/combined_hpo_cv.py --config configs/combined_dual_holdout.yaml
    python chapter1/pipeline/combined_hpo_cv.py  # uses defaults

Key design:
    - Exclude gl_1 (s001), ph_9 (subj09), ph_13 (subj13), ph_14 (subj14)
    - 6 folds: each holds out exactly 1 GL + 1 PH subject
    - PH subjects wrap across folds when fewer GL subjects exist
    - Per-fold target z-score normalization (fit on training set only)
    - Domain-split evaluation: separate GL and PH validation R2
    - shuffle=True in DataLoader for cross-domain batch mixing
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure the project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    ESSENTIAL_MARKER_SHORT_NAMES_COMBINED,
    MARKER_NAME_MAP,
    ProcessingConfig,
    TARGET_COLUMNS,
)
from src.data.data_pipeline import load_and_process_trial_data, prepare_dataset_for_modeling
from src.data.dataset_adapters import GroundLinkAdapter, PatientHandlingAdapter, AdapterResult
from src.loss import NormalizedPeakWeightedLoss
from src.model import ImprovedConvFormer
from src.train import train_and_evaluate, validate_model, analyze_results
from src.pipeline.model_factory import build_model
from src.utils.target_normalization import (
    fit_target_normalizer,
    apply_target_normalizer,
    inverse_target_normalization,
)
from src.utils.input_normalization import fit_input_normalizer, apply_input_normalizer
from src.utils.main_script_helpers import ensure_metadata_split_tuple


def _apply_input_norm_to_trials(trials, inorm):
    """Apply a fitted per-fold input normalizer to a list of (trc, meta, tgt, mask) trials."""
    if inorm is None or not trials:
        return trials
    normed = apply_input_normalizer([t[0] for t in trials], [t[3] for t in trials], inorm)
    return [(trc,) + tuple(t[1:]) for trc, t in zip(normed, trials)]
from src.utils.utils import extract_subject_id_from_filepath
from chapter1.analysis.trial_ops import (
    build_canonical_columns,
    apply_znorm_to_trials,
    trials_to_tensordataset,
)

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed — HPO will be skipped")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ── Constants ───────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "ImprovedConvFormer")

FORCE_SUFFIXES = ("_ml", "_vert", "_ap")
COP_SUFFIXES = ("_px", "_pz")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── YAML config loader ─────────────────────────────────────────────────────

def load_yaml_config(path: str) -> dict:
    if not HAS_YAML:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"Warning: config {path} not found, using defaults")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── Data loading helpers ────────────────────────────────────────────────────
# build_canonical_columns, apply_znorm_to_trials, trials_to_tensordataset
# are imported from chapter1.analysis.trial_ops.

def load_combined_trials(
    gl_adapter: GroundLinkAdapter,
    ph_adapter: PatientHandlingAdapter,
    metadata_df: pd.DataFrame,
    meta_weight_col: str,
    meta_height_col: str,
    meta_subject_col: str,
    canonical_cols: List[str],
    essential_markers: List[str],
    max_length: int,
    force_target_mode: str = "nbw",
    enable_biomech: bool = True,
) -> Tuple[Dict[str, List], int]:
    """Load all valid trials from both datasets, keyed by subject ID.

    Returns:
        trials_by_subject: {subject_id: [(trc, meta, target, mask), ...]}
        actual_max_length: the determined global max sequence length
    """
    gl_pairs = gl_adapter.pair_files()
    ph_pairs = ph_adapter.pair_files()

    all_trc = gl_pairs.trc_files + ph_pairs.trc_files
    all_dat = gl_pairs.data_files + ph_pairs.data_files

    print(f"  GL trials: {len(gl_pairs.trc_files)}, PH trials: {len(ph_pairs.trc_files)}")
    print(f"  Total raw trials: {len(all_trc)}")

    # Determine global_max_len
    from src.data.data_pipeline import pad_or_truncate
    from src.data.file_readers import read_trc_file
    from src.data.groundlink_adapter import read_groundlink_c3d

    if max_length > 0:
        global_max_len = max_length
        print(f"  Using configured max_length: {global_max_len}")
    else:
        global_max_len = 0
        for trc_fp, dat_fp in zip(all_trc, all_dat):
            try:
                if trc_fp.endswith(".c3d"):
                    df = read_groundlink_c3d(trc_fp)
                else:
                    result = read_trc_file(trc_fp, essential_marker_short_names_list=essential_markers)
                    df = result.dataframe
                if df is not None and not df.empty:
                    global_max_len = max(global_max_len, len(df))
            except Exception:
                continue
        if global_max_len == 0:
            global_max_len = 1750
        print(f"  Determined global_max_len: {global_max_len}")

    # Load all trials, keyed by subject
    trials_by_subject: Dict[str, List] = {}
    n_loaded = 0
    n_failed = 0

    for trc_fp, dat_fp in zip(all_trc, all_dat):
        sid = extract_subject_id_from_filepath(trc_fp)

        # Determine if this is a PH trial (needs coordinate corrections)
        is_ph = trc_fp.lower().replace("\\", "/").find("opensim") >= 0 or \
                trc_fp.lower().replace("\\", "/").find("tamu") >= 0 or \
                trc_fp.endswith(".trc")
        trc_suffix = ".trc" if is_ph else ".c3d"

        config = ProcessingConfig(
            trc_filepath=trc_fp,
            data_filepath=dat_fp,
            use_metadata=True,
            metadata_df=metadata_df,
            max_length=global_max_len,
            data_columns_to_use=TARGET_COLUMNS,
            split_name="Combined",
            trc_file_prefix="",
            trc_file_suffix=trc_suffix,
            subject_id_col_in_metadata=meta_subject_col,
            weight_col_in_metadata=meta_weight_col,
            height_col_in_metadata=meta_height_col,
            canonical_marker_columns=canonical_cols,
            essential_marker_short_names_list=essential_markers,
            nan_threshold=0.10,
            apply_coordinate_corrections=is_ph,
            strict_target_columns=False,
            enable_data_diagnostics=False,
            force_target_mode=force_target_mode,
            enable_biomech_preprocessing=enable_biomech,
        )

        trc_t, meta_t, tgt_t, mask_t, problems = load_and_process_trial_data(config)

        if trc_t is not None:
            trials_by_subject.setdefault(sid, []).append((trc_t, meta_t, tgt_t, mask_t))
            n_loaded += 1
        else:
            n_failed += 1
            if problems:
                print(f"    SKIP {Path(trc_fp).name}: {problems[0][:100]}")

    print(f"  Loaded: {n_loaded}, Failed: {n_failed}")
    print(f"  Subjects: {sorted(trials_by_subject.keys())}")

    return trials_by_subject, global_max_len


def augment_bilateral_flip(
    trial_list: List[Tuple],
    canonical_cols: List[str],
) -> List[Tuple]:
    """Add bilateral flip augmentations to a list of trials."""
    from src.data.augmentation import bilateral_flip

    augmented = list(trial_list)  # copy original
    for trc, meta, tgt, mask in trial_list:
        flip_result = bilateral_flip(trc, tgt, mask, canonical_cols, TARGET_COLUMNS)
        if flip_result is not None:
            flip_trc, flip_tgt, flip_mask = flip_result
            augmented.append((flip_trc, meta, flip_tgt, flip_mask))
    return augmented


# ── Fold schedule builder ──────────────────────────────────────────────────

def build_dual_holdout_folds(
    gl_subjects: List[str],
    ph_subjects: List[str],
    n_folds: int = 6,
) -> List[Tuple[str, str]]:
    """Build fold schedule: each fold holds out (gl_subj, ph_subj).

    GL subjects are assigned 1:1 (must have exactly n_folds GL subjects).
    PH subjects wrap cyclically if there are more folds than PH subjects.
    """
    gl_sorted = sorted(gl_subjects)
    ph_sorted = sorted(ph_subjects)

    if len(gl_sorted) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} GL subjects, got {len(gl_sorted)}: {gl_sorted}"
        )

    folds = []
    for i in range(n_folds):
        gl_held = gl_sorted[i]
        ph_held = ph_sorted[i % len(ph_sorted)]
        folds.append((gl_held, ph_held))

    return folds


# ── HPO objective ──────────────────────────────────────────────────────────

def make_combined_hpo_objective(
    train_ds: TensorDataset,
    val_ds: Optional[TensorDataset],
    input_size: int,
    output_size: int,
    max_length: int,
    hpo_epochs: int = 25,
    model_name: str = "ImprovedConvFormer",
):
    """Return an Optuna objective for the combined HPO search."""

    def objective(trial: optuna.Trial) -> float:
        params: dict = {}

        if model_name == "CNN_LSTM":
            lr  = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
            bs  = trial.suggest_categorical("batch_size", [8, 16, 32])
            dr  = trial.suggest_float("dropout", 0.05, 0.40)
            wd  = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)
            params["hidden_size"] = trial.suggest_categorical("hidden_size", [256, 512])
            params["num_layers"]  = trial.suggest_categorical("num_layers", [1, 2])
            params["num_filters"] = trial.suggest_categorical("num_filters", [128, 256])
            params["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 7])
            params["dropout"]     = dr
        else:
            # ImprovedConvFormer — refined search space (fixed architecture winners)
            bs   = 16
            nl   = 2      # num_encoder_layers
            cnn  = 256    # num_cnn_filters
            ffn  = 512    # dim_feedforward
            nhead = 8

            lr  = trial.suggest_float("learning_rate", 2e-4, 8e-4, log=True)
            dr  = trial.suggest_float("dropout", 0.08, 0.15)
            wd  = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
            dm  = trial.suggest_categorical("d_model", [64, 128])

            if dm % nhead != 0:
                raise optuna.exceptions.TrialPruned()

            params.update(dict(d_model=dm, nhead=nhead, num_encoder_layers=nl,
                               dim_feedforward=ffn, num_cnn_filters=cnn, dropout=dr))

        params["learning_rate"] = lr
        params["batch_size"]    = bs
        params["dropout"]       = dr

        model = build_model(model_name, params, input_size, output_size, max_length, DEVICE)

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False) if val_ds else None

        criterion = NormalizedPeakWeightedLoss(
            peak_weight_multiplier=2.0,
            feature_weights=[1.0] * output_size,
        )
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        tl, vl, tm, vm, tr2, vr2 = train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=hpo_epochs,
            device=DEVICE,
            diagnostics=False,
            target_feature_names=TARGET_COLUMNS,
        )

        # Log to W&B if active
        if HAS_WANDB and wandb.run is not None:
            for ep in range(len(vl)):
                log_d = {"epoch": ep + 1, "hpo/val_loss": vl[ep]}
                if tl and ep < len(tl):
                    log_d["hpo/train_loss"] = tl[ep]
                if vm and ep < len(vm):
                    log_d["hpo/val_mae"] = vm[ep]
                if vr2 and ep < len(vr2):
                    log_d["hpo/val_r2"] = vr2[ep]
                wandb.log(log_d)

        return vl[-1] if vl else float("inf")

    return objective


# ── Fold evaluation ────────────────────────────────────────────────────────

def evaluate_domain_split(
    model: torch.nn.Module,
    gl_trials: List[Tuple],
    ph_trials: List[Tuple],
    batch_size: int,
    target_columns: List[str],
) -> Dict:
    """Evaluate model on GL and PH held-out trials separately.

    Returns a dict with gl_r2, ph_r2, per-feature R2, and combined metrics.
    """
    metrics: Dict = {}

    for tag, trials in [("gl", gl_trials), ("ph", ph_trials)]:
        if not trials:
            metrics[f"{tag}_r2"] = float("nan")
            metrics[f"{tag}_mae"] = float("nan")
            continue

        ds = trials_to_tensordataset(trials)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds, actuals, masks = validate_model(model, loader, DEVICE)
        res = analyze_results(preds, actuals, target_columns, masks)

        r2_mean = float(res["R2"].mean())
        mae_mean = float(res["MAE"].mean())
        metrics[f"{tag}_r2"] = r2_mean
        metrics[f"{tag}_mae"] = mae_mean

        for _, row in res.iterrows():
            feat = row["Feature"]
            metrics[f"{tag}_r2_{feat}"] = float(row["R2"])
            metrics[f"{tag}_mae_{feat}"] = float(row["MAE"])

    # Combined R2 (weighted by trial count)
    n_gl = len(gl_trials)
    n_ph = len(ph_trials)
    total = n_gl + n_ph
    if total > 0:
        gl_r2 = metrics.get("gl_r2", 0.0)
        ph_r2 = metrics.get("ph_r2", 0.0)
        if np.isnan(gl_r2): gl_r2 = 0.0
        if np.isnan(ph_r2): ph_r2 = 0.0
        metrics["combined_r2"] = (n_gl * gl_r2 + n_ph * ph_r2) / total
    else:
        metrics["combined_r2"] = float("nan")

    return metrics


# ── Main pipeline ──────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 70)
    print("COMBINED DUAL-HOLDOUT CV WITH HPO")
    print(f"Device: {DEVICE}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── CLI args ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/combined_dual_holdout.yaml")
    parser.add_argument("--best-params", type=str, default=None,
                        help="Path to best_params.json — skips HPO and uses these params directly")
    args, _ = parser.parse_known_args()
    cfg = load_yaml_config(args.config)

    # ── Resolve settings ────────────────────────────────────────────────
    max_length = int((cfg.get("processing", {}) or {}).get("max_length", 1750))
    force_mode = (cfg.get("targets", {}) or {}).get("force_mode", "nbw")
    enable_biomech = bool((cfg.get("processing", {}) or {}).get("biomech_preprocessing", True))
    enable_flip = bool((cfg.get("processing", {}) or {}).get("bilateral_flip", True))
    hpo_trials = int((cfg.get("hpo", {}) or {}).get("num_trials", 30))
    hpo_epochs = int((cfg.get("hpo", {}) or {}).get("epochs_per_trial", 25))
    fold_epochs = int(os.getenv("COMBINED_FOLD_EPOCHS", "") or (cfg.get("training", {}) or {}).get("fold_epochs", 200))
    es_patience = int((cfg.get("early_stopping", {}) or {}).get("patience", 35))
    es_min_delta = float((cfg.get("early_stopping", {}) or {}).get("min_delta", 0.0))
    n_folds = int((cfg.get("dual_holdout", {}) or {}).get("n_folds", 6))

    gl_excl = list((cfg.get("dataset", {}) or {}).get("gl_excluded_subjects", ["s001"]))
    ph_excl = list((cfg.get("dataset", {}) or {}).get("ph_excluded_subjects", ["subj09", "subj13", "subj14"]))

    run_name = os.getenv(
        "RUN_NAME_OVERRIDE",
        f"combined_dual_holdout_{datetime.now().strftime('%d-%m-%Y')}",
    )

    print(f"\n  Run name: {run_name}")
    print(f"  Force mode: {force_mode}")
    print(f"  Max length: {max_length}")
    print(f"  HPO trials: {hpo_trials}, HPO epochs/trial: {hpo_epochs}")
    print(f"  Fold epochs: {fold_epochs}, ES patience: {es_patience}")
    print(f"  GL exclusions: {gl_excl}")
    print(f"  PH exclusions: {ph_excl}")
    print(f"  Bilateral flip: {enable_flip}")
    print(f"  N folds: {n_folds}")

    # ── Output directories ──────────────────────────────────────────────
    run_root = Path("outputs") / run_name
    results_dir = run_root / "results"
    models_dir = run_root / "fold_models"
    studies_dir = run_root / "studies"
    for d in [results_dir, models_dir, studies_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Create adapters with exclusions ─────────────────────────────────
    gl_adapter = GroundLinkAdapter()
    gl_adapter.EXCLUDED_SUBJECTS = gl_excl
    ph_adapter = PatientHandlingAdapter()
    ph_adapter.EXCLUDED_SUBJECTS = ph_excl

    print(f"\n{'─' * 50}")
    print("Loading metadata...")

    gl_meta_df, gl_meta_cfg = gl_adapter.load_metadata()
    ph_meta_df, ph_meta_cfg = ph_adapter.load_metadata()

    # Merge metadata with namespace prefixes
    frames = []
    if gl_meta_df is not None:
        gl_m = gl_meta_df.copy()
        gl_m.index = [f"gl_{idx}" for idx in gl_meta_df.index]
        gl_m = gl_m.rename(columns={
            gl_meta_cfg.weight_col: "Mass (kg)",
            gl_meta_cfg.height_col: "Height (m)",
        })
        frames.append(gl_m[["Mass (kg)", "Height (m)"]])
    if ph_meta_df is not None:
        ph_m = ph_meta_df.copy()
        ph_m.index = [f"ph_{idx}" for idx in ph_meta_df.index]
        ph_m = ph_m.rename(columns={
            ph_meta_cfg.weight_col: "Mass (kg)",
            ph_meta_cfg.height_col: "Height (m)",
        })
        frames.append(ph_m[["Mass (kg)", "Height (m)"]])

    combined_meta = pd.concat(frames) if frames else None
    print(f"  Combined metadata: {len(combined_meta)} subjects")
    print(f"  Index: {sorted(combined_meta.index.tolist())}")

    # ── Load all trials ─────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("Loading all trials...")

    canonical_cols = build_canonical_columns(ESSENTIAL_MARKER_SHORT_NAMES_COMBINED)

    trials_by_subject, actual_max_len = load_combined_trials(
        gl_adapter=gl_adapter,
        ph_adapter=ph_adapter,
        metadata_df=combined_meta,
        meta_weight_col="Mass (kg)",
        meta_height_col="Height (m)",
        meta_subject_col="participant_key",
        canonical_cols=canonical_cols,
        essential_markers=ESSENTIAL_MARKER_SHORT_NAMES_COMBINED,
        max_length=max_length,
        force_target_mode=force_mode,
        enable_biomech=enable_biomech,
    )

    # Separate GL and PH subjects
    gl_subjects = sorted([s for s in trials_by_subject if s.startswith("gl_")])
    ph_subjects = sorted([s for s in trials_by_subject if s.startswith("ph_")])

    print(f"\n  GL subjects ({len(gl_subjects)}): {gl_subjects}")
    print(f"  PH subjects ({len(ph_subjects)}): {ph_subjects}")
    for sid, trials in sorted(trials_by_subject.items()):
        print(f"    {sid}: {len(trials)} trials")

    total_trials = sum(len(v) for v in trials_by_subject.values())
    print(f"  Total valid trials: {total_trials}")

    # Determine input/output sizes from first trial
    first_trial = next(iter(trials_by_subject.values()))[0]
    input_size = first_trial[0].shape[1]
    output_size = first_trial[2].shape[1]
    print(f"  Input size: {input_size}, Output size: {output_size}")
    print(f"  Max length: {actual_max_len}")

    # ── Build fold schedule ─────────────────────────────────────────────
    folds = build_dual_holdout_folds(gl_subjects, ph_subjects, n_folds=n_folds)
    print(f"\n{'─' * 50}")
    print(f"Dual-holdout CV: {len(folds)} folds")
    for i, (gl_h, ph_h) in enumerate(folds):
        n_gl_h = len(trials_by_subject.get(gl_h, []))
        n_ph_h = len(trials_by_subject.get(ph_h, []))
        print(f"  Fold {i+1}: hold-out GL={gl_h} ({n_gl_h} trials), PH={ph_h} ({n_ph_h} trials)")

    # ── HPO on fold 1's training data ───────────────────────────────────
    if args.best_params:
        # Skip HPO — load params from JSON file
        print(f"\n{'=' * 50}")
        print(f"Skipping HPO — loading params from {args.best_params}")
        print(f"{'=' * 50}")
        with open(args.best_params) as f:
            best_params = json.load(f)
        print(f"  Loaded params: {json.dumps(best_params, indent=2)}")
        with open(studies_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
    elif HAS_OPTUNA:
        print(f"\n{'=' * 50}")
        print(f"HPO: {hpo_trials} trials on fold-1 training split")
        print(f"{'=' * 50}")

        fold1_gl_held, fold1_ph_held = folds[0]
        hpo_held_out = {fold1_gl_held, fold1_ph_held}

        # Build HPO training set (all subjects NOT held out in fold 1)
        hpo_train_trials = []
        hpo_val_trials = []  # Use fold-1 held-out as HPO validation
        for sid, trials in trials_by_subject.items():
            if sid in hpo_held_out:
                hpo_val_trials.extend(trials)
            else:
                hpo_train_trials.extend(trials)

        # Fit target normalizer on HPO train data
        hpo_normalizer = fit_target_normalizer(
            [t[2] for t in hpo_train_trials],  # target tensors
            [t[3] for t in hpo_train_trials],  # mask tensors
            TARGET_COLUMNS,
        )

        # Apply augmentation + normalization
        if enable_flip:
            hpo_train_trials = augment_bilateral_flip(hpo_train_trials, canonical_cols)
        hpo_train_trials = apply_znorm_to_trials(hpo_train_trials, hpo_normalizer)
        hpo_val_trials = apply_znorm_to_trials(hpo_val_trials, hpo_normalizer)

        hpo_train_ds = trials_to_tensordataset(hpo_train_trials)
        hpo_val_ds = trials_to_tensordataset(hpo_val_trials)

        print(f"  HPO train: {len(hpo_train_ds)} samples, HPO val: {len(hpo_val_ds)} samples")

        # Run Optuna HPO
        objective = make_combined_hpo_objective(
            hpo_train_ds, hpo_val_ds,
            input_size, output_size, actual_max_len,
            hpo_epochs=hpo_epochs,
            model_name=MODEL_NAME,
        )

        # Optional W&B callback
        wandbc = None
        try:
            from optuna_integration.wandb import WeightsAndBiasesCallback
            if HAS_WANDB:
                wandbc = WeightsAndBiasesCallback(
                    metric_name="val_loss",
                    wandb_kwargs=dict(
                        project="metahuman-grf",
                        group=run_name,
                        job_type="HPO-Combined",
                        tags=["HPO", "COMBINED", "dual-holdout"],
                    ),
                    as_multirun=True,
                )
                raw_obj = objective
                def _named_obj(trial):
                    if wandb.run is not None:
                        wandb.run.name = f"hpo/trial_{trial.number:03d}"
                    return raw_obj(trial)
                objective = wandbc.track_in_wandb()(_named_obj)
        except (ImportError, Exception) as e:
            print(f"  W&B HPO callback unavailable: {e}")

        study = optuna.create_study(
            direction="minimize",
            study_name=f"{run_name}_HPO",
        )
        study.optimize(objective, n_trials=hpo_trials, callbacks=[wandbc] if wandbc else [])

        best_params = dict(study.best_params)
        # Inject fixed architecture params that were not searched (ICF only)
        if MODEL_NAME == "ImprovedConvFormer":
            best_params.setdefault("batch_size", 16)
            best_params.setdefault("num_encoder_layers", 2)
            best_params.setdefault("nhead", 8)
            best_params.setdefault("num_cnn_filters", 256)
            best_params.setdefault("dim_feedforward", 512)
        print(f"\n  Best HPO params: {json.dumps(best_params, indent=2)}")
        print(f"  Best val loss: {study.best_value:.6f}")

        # Save study
        with open(studies_dir / "combined_hpo_study.pkl", "wb") as f:
            pickle.dump(study, f)
        with open(studies_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
    else:
        # Fallback defaults
        if MODEL_NAME == "CNN_LSTM":
            best_params = {
                "learning_rate": 1e-3,
                "batch_size": 16,
                "dropout": 0.15,
                "hidden_size": 512,
                "num_layers": 1,
                "num_filters": 256,
                "kernel_size": 3,
                "weight_decay": 1e-3,
            }
        else:
            best_params = {
                "learning_rate": 4.8e-4,
                "batch_size": 16,
                "dropout": 0.10,
                "d_model": 128,
                "num_encoder_layers": 2,
                "nhead": 8,
                "weight_decay": 1e-3,
                "num_cnn_filters": 256,
                "dim_feedforward": 512,
            }
        print(f"  Using fallback params (no Optuna): {best_params}")

    # ── Fold CV loop ────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"Starting {len(folds)}-fold dual-holdout CV")
    print(f"Epochs per fold: {fold_epochs}, ES patience: {es_patience}")
    print(f"{'=' * 50}")

    fold_metrics: List[Dict] = []
    all_fold_details: List[Dict] = []

    dm = best_params.get("d_model", 128)
    nhead = best_params.get("nhead", 8)
    nl = best_params.get("num_encoder_layers", 2)
    ffn = best_params.get("dim_feedforward", 512)
    dr = best_params.get("dropout", 0.10)
    lr = best_params.get("learning_rate", 4.8e-4)
    bs = best_params.get("batch_size", 16)
    wd = best_params.get("weight_decay", 1e-3)
    cnn = best_params.get("num_cnn_filters", 256)

    for fold_idx, (gl_held, ph_held) in enumerate(folds):
        fold_num = fold_idx + 1
        print(f"\n{'─' * 50}")
        print(f"FOLD {fold_num}/{len(folds)}: hold-out GL={gl_held}, PH={ph_held}")
        print(f"{'─' * 50}")

        # W&B fold run
        wb_run = None
        if HAS_WANDB:
            try:
                wb_run = wandb.init(
                    project="metahuman-grf",
                    group=run_name,
                    name=f"fold_{fold_num}_gl={gl_held}_ph={ph_held}",
                    config={
                        "fold": fold_num,
                        "gl_held_out": gl_held,
                        "ph_held_out": ph_held,
                        "dataset": "COMBINED",
                        **best_params,
                    },
                    tags=["COMBINED", "dual-holdout", f"fold_{fold_num}"],
                    reinit=True,
                    job_type="CV-fold",
                )
            except Exception as e:
                print(f"  W&B init skipped: {e}")

        # Split trials: held-out subjects → validation, rest → training
        held_out_set = {gl_held, ph_held}
        train_trials = []
        val_gl_trials = []
        val_ph_trials = []

        for sid, trials in trials_by_subject.items():
            if sid in held_out_set:
                if sid.startswith("gl_"):
                    val_gl_trials.extend(trials)
                else:
                    val_ph_trials.extend(trials)
            else:
                train_trials.extend(trials)

        n_train_raw = len(train_trials)
        n_val_gl = len(val_gl_trials)
        n_val_ph = len(val_ph_trials)
        print(f"  Train: {n_train_raw} trials, Val GL: {n_val_gl}, Val PH: {n_val_ph}")

        # Per-fold target normalization (fit on training data only)
        fold_normalizer = fit_target_normalizer(
            [t[2] for t in train_trials],
            [t[3] for t in train_trials],
            TARGET_COLUMNS,
        )

        # Augment training data with bilateral flips
        if enable_flip:
            train_trials = augment_bilateral_flip(train_trials, canonical_cols)
            print(f"  After bilateral flip: {len(train_trials)} train trials")

        # Apply z-score normalization
        train_trials = apply_znorm_to_trials(train_trials, fold_normalizer)
        val_gl_trials_n = apply_znorm_to_trials(val_gl_trials, fold_normalizer)
        val_ph_trials_n = apply_znorm_to_trials(val_ph_trials, fold_normalizer)

        # Per-fold INPUT normalization (M6): when INPUT_NORM_MODE=per_fold the
        # pipeline emits un-normalized inputs; fit on this fold's training trials
        # and apply to train + both held-out domains.
        if os.getenv("INPUT_NORM_MODE", "per_trial").strip().lower() == "per_fold":
            fold_inorm = fit_input_normalizer([t[0] for t in train_trials], [t[3] for t in train_trials])
            train_trials = _apply_input_norm_to_trials(train_trials, fold_inorm)
            val_gl_trials_n = _apply_input_norm_to_trials(val_gl_trials_n, fold_inorm)
            val_ph_trials_n = _apply_input_norm_to_trials(val_ph_trials_n, fold_inorm)

        # Build DataSets
        train_ds = trials_to_tensordataset(train_trials)
        val_all = val_gl_trials_n + val_ph_trials_n
        val_ds = trials_to_tensordataset(val_all) if val_all else None

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False) if val_ds else None

        # Build model
        model = build_model(MODEL_NAME, best_params, input_size, output_size, actual_max_len, DEVICE)

        criterion = NormalizedPeakWeightedLoss(
            peak_weight_multiplier=2.0,
            feature_weights=[1.0] * output_size,
        )
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # Train
        t_fold_start = time.time()
        tl, vl, tm, vm, tr2, vr2 = train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=fold_epochs,
            device=DEVICE,
            diagnostics=False,
            target_feature_names=TARGET_COLUMNS,
            early_stopping_patience=es_patience,
            early_stopping_min_delta=es_min_delta,
        )
        fold_time = time.time() - t_fold_start
        print(f"  Fold training time: {fold_time:.1f}s ({fold_time/60:.1f} min)")

        # Log training curves to W&B
        if wb_run and tl:
            for ep in range(len(tl)):
                log_d = {"epoch": ep + 1, "train/loss": tl[ep], "train/mae": tm[ep], "train/r2": tr2[ep]}
                if vl and ep < len(vl):
                    log_d.update({"val/loss": vl[ep], "val/mae": vm[ep], "val/r2": vr2[ep]})
                wandb.log(log_d)

        # Load best model state if available (from early stopping)
        best_state = getattr(train_and_evaluate, "_best_state", None)
        # The best state is stored on the model itself after training
        # (train_and_evaluate restores it internally via early stopping)

        # Evaluate domain-split
        print(f"\n  Evaluating fold {fold_num}...")
        fold_metric = {
            "fold": fold_num,
            "gl_held_out": gl_held,
            "ph_held_out": ph_held,
            "n_train": n_train_raw,
            "n_val_gl": n_val_gl,
            "n_val_ph": n_val_ph,
            "train_time_s": fold_time,
        }

        # Add final training metrics
        if tl:
            fold_metric["final_train_loss"] = tl[-1]
            fold_metric["final_train_r2"] = tr2[-1]
        if vl:
            fold_metric["final_val_loss"] = vl[-1]
            fold_metric["final_val_r2"] = vr2[-1]

        # Domain-split evaluation on held-out subjects
        domain_metrics = evaluate_domain_split(
            model, val_gl_trials_n, val_ph_trials_n, bs, TARGET_COLUMNS,
        )
        fold_metric.update(domain_metrics)

        # Print fold summary
        gl_r2 = fold_metric.get("gl_r2", float("nan"))
        ph_r2 = fold_metric.get("ph_r2", float("nan"))
        comb_r2 = fold_metric.get("combined_r2", float("nan"))
        print(f"  GL  held-out R²: {gl_r2:.4f}")
        print(f"  PH  held-out R²: {ph_r2:.4f}")
        print(f"  Combined    R²: {comb_r2:.4f}")

        # Print AP force tracking
        for tag in ["gl", "ph"]:
            for plate in [1, 2]:
                key = f"{tag}_r2_ground_force_{plate}_ap"
                if key in fold_metric:
                    print(f"    {tag.upper()} F{plate}_AP R²: {fold_metric[key]:.4f}")

        fold_metrics.append(fold_metric)

        # W&B finish
        if wb_run:
            wandb.summary.update(fold_metric)
            wandb.finish()

        # Save fold model
        torch.save(model.state_dict(), models_dir / f"fold_{fold_num}_gl={gl_held}_ph={ph_held}.pt")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("COMBINED DUAL-HOLDOUT CV — SUMMARY")
    print(f"{'=' * 70}")

    results_df = pd.DataFrame(fold_metrics)
    results_df.to_csv(results_dir / "combined_dual_holdout_results.csv", index=False)

    # Domain-split summary
    for tag, label in [("gl", "GroundLink"), ("ph", "Patient Handling")]:
        col = f"{tag}_r2"
        if col in results_df.columns:
            vals = results_df[col].dropna()
            if len(vals) > 0:
                print(f"\n  {label} held-out ({len(vals)} folds):")
                print(f"    Mean R²: {vals.mean():.4f} ± {vals.std():.4f}")
                print(f"    Range:   [{vals.min():.4f}, {vals.max():.4f}]")
                best_fold = results_df.loc[vals.idxmax()]
                worst_fold = results_df.loc[vals.idxmin()]
                print(f"    Best:    fold {int(best_fold['fold'])} (gl={best_fold['gl_held_out']}, ph={best_fold['ph_held_out']})")
                print(f"    Worst:   fold {int(worst_fold['fold'])} (gl={worst_fold['gl_held_out']}, ph={worst_fold['ph_held_out']})")

    # Combined R2 summary
    if "combined_r2" in results_df.columns:
        cvals = results_df["combined_r2"].dropna()
        if len(cvals) > 0:
            print(f"\n  Combined (trial-weighted):")
            print(f"    Mean R²: {cvals.mean():.4f} ± {cvals.std():.4f}")

    # Per-feature R2 table
    print(f"\n  Per-feature R² (mean across folds):")
    print(f"  {'Feature':<30} {'GL R²':>8} {'PH R²':>8} {'Notes':>20}")
    print(f"  {'─' * 70}")
    for feat in TARGET_COLUMNS:
        gl_vals = [m.get(f"gl_r2_{feat}", float("nan")) for m in fold_metrics]
        ph_vals = [m.get(f"ph_r2_{feat}", float("nan")) for m in fold_metrics]
        gl_mean = np.nanmean(gl_vals) if gl_vals else float("nan")
        ph_mean = np.nanmean(ph_vals) if ph_vals else float("nan")
        notes = ""
        if "_ap" in feat:
            notes = "← AP TRACKING"
        elif "_vert" in feat:
            notes = "← PRIMARY"
        print(f"  {feat:<30} {gl_mean:>8.4f} {ph_mean:>8.4f} {notes:>20}")

    # Save best params alongside results
    with open(results_dir / "best_hpo_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Total runtime: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {results_dir}")
    print(f"Models saved to:  {models_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
