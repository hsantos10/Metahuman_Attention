# -*- coding: utf-8 -*-
"""
LOSO (leave-one-subject-out) cross-validation pipeline.

Includes:
  - Pre-LOSO HPO
  - Cross-lab test data loading
  - Per-fold training / evaluation loop
  - Results aggregation and saving
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import ESSENTIAL_MARKER_SHORT_NAMES_COMBINED, MARKER_NAME_MAP, ProcessingConfig
from src.data.data_pipeline import load_and_process_trial_data, prepare_dataset_for_modeling
from src.data.dataset_adapters import GroundLinkAdapter, PatientHandlingAdapter
from src.loss import NormalizedPeakWeightedLoss
from src.train import analyze_results, train_and_evaluate, validate_model
from src.utils.input_normalization import (
    apply_input_normalizer,
    apply_input_normalizer_to_batch,
    fit_input_normalizer,
)
from src.utils.main_script_helpers import ensure_metadata_split_tuple, env_int
from src.utils.target_normalization import apply_target_normalizer, fit_target_normalizer
from src.utils.utils import extract_subject_id_from_filepath as _extract_sid

from src.pipeline.config_loader import RunConfig
from chapter1.pipeline.hpo import make_loso_hpo_objective, run_study
from chapter1.pipeline.loso_plots import generate_loso_plots
from src.pipeline.model_factory import build_model, maybe_compile

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ── Cross-lab trial loading ─────────────────────────────────────────────────

def _build_canonical_columns(essential_markers: List[str]) -> List[str]:
    """Reconstruct the canonical feature column list used by the data pipeline."""
    canon_temp = []
    for sn in essential_markers:
        fn = MARKER_NAME_MAP.get(sn, sn)
        canon_temp.extend([f"{fn}_X", f"{fn}_Y", f"{fn}_Z"])
    canon_base = sorted(canon_temp)
    canon_vel_acc = []
    for c in canon_base:
        canon_vel_acc.extend([c, f"{c}_vel", f"{c}_acc"])
    marker_pairs = [
        ("LSHO", "LELBL"), ("LELBL", "LWRL"), ("RSHO", "RELBL"), ("RELBL", "RWRL"),
        ("LKNE", "LANKL"), ("LANKL", "LTOE"), ("RKNE", "RANKM"), ("RANKM", "RTOE"),
        ("LSHO", "RSHO"), ("LASI", "LSHO"),
    ]
    canon_rel = [f"rel_{a}_{b}_{ax}" for a, b in marker_pairs for ax in "XYZ"]
    return canon_vel_acc + canon_rel


def _load_cross_lab_trials(
    cfg: RunConfig,
    essential_markers: List[str],
    canonical_cols: List[str],
    target_columns: List[str],
    max_length: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load trials from the *other* lab (PH↔GL) for cross-lab evaluation."""

    trials: List[Tuple] = []
    try:
        if cfg.selected_dataset == "PATIENT_HANDLING":
            xlab_adapter = GroundLinkAdapter()
            xlab_apply_corrections = False
            xlab_suffix = ".c3d"
        else:
            xlab_adapter = PatientHandlingAdapter()
            xlab_adapter.EXCLUDED_SUBJECTS = ["subj13", "subj14"]   # plate-swapped, excluded in the paper (matters for CROSS_LAB_SUBJECT=all)
            xlab_apply_corrections = True
            xlab_suffix = ".trc"

        xlab_pairs = xlab_adapter.pair_files()
        xlab_meta_df_raw, xlab_meta_cfg_raw = xlab_adapter.load_metadata()

        # Prefix metadata index to match subject-id convention
        if xlab_meta_df_raw is not None:
            xlab_meta = xlab_meta_df_raw.copy()
            prefix = "gl_" if cfg.selected_dataset == "PATIENT_HANDLING" else "ph_"
            xlab_meta.index = [f"{prefix}{idx}" for idx in xlab_meta_df_raw.index]
        else:
            xlab_meta = None

        all_sids = [_extract_sid(p) for p in xlab_pairs.trc_files]
        # CROSS_LAB_SUBJECT=all evaluates on every subject of the other laboratory (the paper's Table V protocol)
        subj_idx = [i for i, sid in enumerate(all_sids) if cfg.cross_lab_subject == "all" or sid == cfg.cross_lab_subject]
        trc_paths = [xlab_pairs.trc_files[i] for i in subj_idx]
        dat_paths = [xlab_pairs.data_files[i] for i in subj_idx]
        print(f"\nCross-lab subject={cfg.cross_lab_subject}: {len(trc_paths)} trials found")

        for trc_fp, dat_fp in zip(trc_paths, dat_paths):
            proc_cfg = ProcessingConfig(
                trc_filepath=trc_fp,
                data_filepath=dat_fp,
                use_metadata=xlab_meta is not None,
                metadata_df=xlab_meta,
                max_length=max_length,
                data_columns_to_use=target_columns,
                split_name="CrossLabTest",
                trc_file_prefix="",
                trc_file_suffix=xlab_suffix,
                subject_id_col_in_metadata="participant_key",
                weight_col_in_metadata=xlab_meta_cfg_raw.weight_col if xlab_meta_cfg_raw else "Weight",
                height_col_in_metadata=xlab_meta_cfg_raw.height_col if xlab_meta_cfg_raw else "Height",
                canonical_marker_columns=canonical_cols,
                essential_marker_short_names_list=ESSENTIAL_MARKER_SHORT_NAMES_COMBINED,
                nan_threshold=0.10,
                apply_coordinate_corrections=xlab_apply_corrections,
                strict_target_columns=False,
                enable_data_diagnostics=False,
                force_target_mode=cfg.force_target_mode,
                enable_biomech_preprocessing=cfg.enable_biomech_preprocessing,
            )
            tt, mt, tgt, mk, probs = load_and_process_trial_data(proc_cfg)
            if tt is not None:
                mt_safe = mt if mt is not None else torch.zeros(2, dtype=torch.float32)
                trials.append((tt, mt_safe, tgt, mk))
            elif probs:
                print(f"  Cross-lab trial skipped: {trc_fp} — {probs[:2]}")

        print(f"  Loaded {len(trials)} valid cross-lab trials")
    except Exception as exc:
        print(f"  WARNING: Could not load cross-lab data: {exc}")
        import traceback; traceback.print_exc()

    return trials


# ── Single-fold evaluation helpers ──────────────────────────────────────────

def _evaluate_primary(
    fold_model, fold_test_ds, batch_size, device, target_columns, tag,
) -> Dict:
    """Evaluate on the held-out subject from the primary dataset."""
    loader = DataLoader(fold_test_ds, batch_size=batch_size, shuffle=False)
    preds, actuals, masks = validate_model(fold_model, loader, device)
    res = analyze_results(preds, actuals, target_columns, masks)
    r2 = float(res["R2"].mean())
    mae = float(res["MAE"].mean())
    metric: Dict = {f"{tag}_r2": r2, f"{tag}_mae": mae}
    for _, row in res.iterrows():
        metric[f"{tag}_r2_{row['Feature']}"] = float(row["R2"])
        metric[f"{tag}_mae_{row['Feature']}"] = float(row["MAE"])
    return metric, r2, mae


def _evaluate_cross_lab(
    fold_model,
    cross_lab_trials,
    fold_normalizer,
    batch_size,
    device,
    target_columns,
    tag,
    input_normalizer=None,
) -> Dict:
    """Evaluate on the cross-lab subject, applying fold normalisation.

    ``input_normalizer`` (set when INPUT_NORM_MODE=per_fold) standardizes the
    cross-lab inputs with the current fold's training-fit statistics, matching
    what the model saw in-domain.
    """
    trc_stack  = torch.stack([t[0] for t in cross_lab_trials])
    meta_stack = torch.stack([t[1] for t in cross_lab_trials])
    tgt_stack  = torch.stack([t[2] for t in cross_lab_trials])
    mask_stack = torch.stack([t[3] for t in cross_lab_trials])

    if input_normalizer is not None:
        trc_stack = apply_input_normalizer_to_batch(trc_stack, mask_stack, input_normalizer)

    if fold_normalizer is not None:
        tgt_np = tgt_stack.numpy().copy()
        norm_cols = fold_normalizer.get("target_columns", [])
        means = fold_normalizer.get("mean", [])
        stds = fold_normalizer.get("std", [])
        for col_idx, col_name in enumerate(target_columns):
            if col_name in norm_cols:
                idx = norm_cols.index(col_name)
                if stds[idx] > 1e-9:
                    tgt_np[:, :, col_idx] = (tgt_np[:, :, col_idx] - means[idx]) / stds[idx]
        tgt_stack = torch.tensor(tgt_np, dtype=torch.float32)

    ds = TensorDataset(trc_stack, meta_stack, tgt_stack, mask_stack)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, actuals, masks = validate_model(fold_model, loader, device)
    res = analyze_results(preds, actuals, target_columns, masks)
    r2 = float(res["R2"].mean())
    mae = float(res["MAE"].mean())
    metric: Dict = {f"{tag}_r2": r2, f"{tag}_mae": mae}
    for _, row in res.iterrows():
        metric[f"{tag}_r2_{row['Feature']}"] = float(row["R2"])
        metric[f"{tag}_mae_{row['Feature']}"] = float(row["MAE"])
    return metric, r2, mae


def _load_best_params_override() -> Optional[Dict]:
    """Return a fixed hyperparameter dict when ``LOSO_BEST_PARAMS`` is set.

    Lets M6 re-runs skip the (stale-after-preprocessing-change) pre-LOSO HPO and
    reuse tuned params. The value may be a path to a JSON file or a literal JSON
    object. Returns ``None`` (→ run HPO as usual) when unset or unparseable.
    """
    raw = os.getenv("LOSO_BEST_PARAMS", "").strip()
    if not raw:
        return None
    try:
        p = Path(raw)
        params = json.load(open(p, "r", encoding="utf-8")) if p.exists() else json.loads(raw)
    except Exception as exc:
        print(f"  [best-params] failed to parse LOSO_BEST_PARAMS: {exc}")
        return None
    if not isinstance(params, dict):
        print(f"  [best-params] ignoring non-dict LOSO_BEST_PARAMS ({type(params).__name__})")
        return None
    return params


# ── Main entry point ────────────────────────────────────────────────────────

def run_loso_pipeline(
    cfg: RunConfig,
    prepare_args: dict,
    train_data_tuple,
    val_data_tuple,
    paired_trc_files: List[str],
    input_size: int,
    output_size: int,
    max_length: int,
    essential_markers: List[str],
    use_metadata: bool,
) -> None:
    """Execute the full LOSO cross-validation pipeline and exit."""

    target_columns = cfg.target_columns
    device = cfg.device
    model_name = os.getenv("MODEL_NAME", "ImprovedConvFormer")
    # When per_fold, the data pipeline emits un-normalized inputs and we refit a
    # single input normalizer on each fold's training frames (M6). Otherwise the
    # pipeline's per-trial z-scoring already ran.
    input_norm_mode = os.getenv("INPUT_NORM_MODE", "per_trial").strip().lower()

    # ── Output directories ──────────────────────────────────────────────
    run_root    = Path(cfg.output_root) / cfg.run_name
    results_dir = run_root / "results"
    models_dir  = run_root / "loso_models"
    studies_dir = run_root / "studies"
    logs_dir    = run_root / "logs"
    plots_dir   = run_root / "plots"
    for d in [results_dir, models_dir, studies_dir, logs_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Pre-LOSO HPO (or fixed best-params override) ─────────────────────
    best_hpo = _load_best_params_override()
    if best_hpo is not None:
        print(f"\n{'=' * 50}")
        print(f"Skipping pre-LOSO HPO — using fixed best params (LOSO_BEST_PARAMS):")
        print(f"  {best_hpo}")
        print(f"{'=' * 50}")
    else:
        print(f"\n{'=' * 50}")
        print(f"Pre-LOSO HPO: {cfg.hpo_trials} trials on full {cfg.selected_dataset} dataset")
        print(f"{'=' * 50}")

        hpo_train = ensure_metadata_split_tuple(train_data_tuple, metadata_dim=2)
        hpo_val   = ensure_metadata_split_tuple(val_data_tuple, metadata_dim=2) if val_data_tuple[0] else None

        hpo_train_ds = TensorDataset(*(torch.stack(t) for t in hpo_train))
        hpo_val_ds   = TensorDataset(*(torch.stack(t) for t in hpo_val)) if hpo_val and hpo_val[0] else None

        objective = make_loso_hpo_objective(
            cfg, hpo_train_ds, hpo_val_ds,
            input_size, output_size, max_length, target_columns,
            model_name=model_name,
        )

        study = run_study(
            objective, cfg.hpo_trials,
            wandb_group=cfg.run_name,
            wandb_tags=["HPO", "LOSO", cfg.selected_dataset],
            wandb_job_type=f"HPO-LOSO-{cfg.selected_dataset}",
            study_name=f"{cfg.run_name}_LOSO_preHPO",
        )
        best_hpo = study.best_params
        print(f"Best pre-LOSO HPO params: {best_hpo}")

        with open(studies_dir / "loso_pre_hpo.pkl", "wb") as f:
            pickle.dump(study, f)

    # ── Cross-lab data ──────────────────────────────────────────────────
    canonical_cols = _build_canonical_columns(essential_markers)
    cross_lab_trials = _load_cross_lab_trials(
        cfg, essential_markers, canonical_cols, target_columns, max_length,
    )

    # ── LOSO fold loop ──────────────────────────────────────────────────
    all_sids = [_extract_sid(p) for p in paired_trc_files]
    unique_subjects = list(dict.fromkeys(all_sids))
    print(f"\nLOSO: {len(unique_subjects)} unique subjects: {unique_subjects}")

    primary_tag = "ph" if cfg.selected_dataset == "PATIENT_HANDLING" else "gl"
    xlab_tag    = "gl" if cfg.selected_dataset == "PATIENT_HANDLING" else "ph"
    batch_size  = best_hpo.get("batch_size", env_int("LOSO_BATCH_SIZE", 8))
    loso_epochs = cfg.effective_loso_epochs

    active_exclusions = None
    adapter = None  # we don't need the adapter object here but check yaml
    _excl_cfg = (cfg.yaml_cfg.get("dataset", {}) or {}).get("excluded_subjects", None)
    if _excl_cfg:
        active_exclusions = list(_excl_cfg)

    fold_metrics: List[Dict] = []

    for held_out in unique_subjects:
        print(f"\n{'=' * 20} LOSO Fold: held-out={held_out} {'=' * 20}")

        # W&B fold run
        wb_run = None
        if HAS_WANDB:
            try:
                wb_config = {
                    "held_out": held_out,
                    "dataset": cfg.selected_dataset,
                    "excluded_subjects": active_exclusions or [],
                    "n_loso_folds": len(unique_subjects),
                    **best_hpo,
                }
                wb_tags = ["LOSO"]
                if active_exclusions:
                    wb_tags.append("no-gl1")
                wb_run = wandb.init(
                    project="metahuman-grf", group=cfg.run_name,
                    name=f"loso/fold_{held_out}", config=wb_config,
                    tags=wb_tags, reinit=True,
                    job_type="LOSO-fold",
                )
            except Exception as exc:
                print(f"  W&B init skipped: {exc}")

        # Prepare fold data
        fold_args = {**prepare_args, "held_out_subject": held_out}
        try:
            fold_tr, fold_va, fold_te, fold_ml, fold_is, fold_os, _ = prepare_dataset_for_modeling(**fold_args)
        except Exception as exc:
            print(f"  Fold {held_out}: data prep failed — {exc}")
            continue

        if not fold_tr[0]:
            print(f"  Fold {held_out}: no training data, skipping."); continue
        if not fold_te[0]:
            print(f"  Fold {held_out}: no test data, skipping."); continue

        # Target normalisation (fit on fold train only)
        target_idx = 2 if use_metadata else 1
        mask_idx   = 3 if use_metadata else 2
        fold_normalizer = fit_target_normalizer(fold_tr[target_idx], fold_tr[mask_idx], target_columns)
        if fold_normalizer is not None:
            fold_tr = apply_target_normalizer(fold_tr, use_metadata, fold_normalizer)
            fold_va = apply_target_normalizer(fold_va, use_metadata, fold_normalizer)
            fold_te = apply_target_normalizer(fold_te, use_metadata, fold_normalizer)

        # Per-fold INPUT normalisation (M6): fit on fold-train frames only, then
        # apply to train/val/test here and to the cross-lab trials at eval time.
        fold_input_norm = None
        if input_norm_mode == "per_fold":
            fold_input_norm = fit_input_normalizer(fold_tr[0], fold_tr[mask_idx])
            fold_tr = (apply_input_normalizer(fold_tr[0], fold_tr[mask_idx], fold_input_norm),) + tuple(fold_tr[1:])
            if fold_va[0]:
                fold_va = (apply_input_normalizer(fold_va[0], fold_va[mask_idx], fold_input_norm),) + tuple(fold_va[1:])
            if fold_te[0]:
                fold_te = (apply_input_normalizer(fold_te[0], fold_te[mask_idx], fold_input_norm),) + tuple(fold_te[1:])

        fold_tr = ensure_metadata_split_tuple(fold_tr, metadata_dim=2)
        fold_va = ensure_metadata_split_tuple(fold_va, metadata_dim=2)
        fold_te = ensure_metadata_split_tuple(fold_te, metadata_dim=2)

        train_ds = TensorDataset(*(torch.stack(t) for t in fold_tr))
        val_ds   = TensorDataset(*(torch.stack(t) for t in fold_va)) if fold_va[0] else None
        test_ds  = TensorDataset(*(torch.stack(t) for t in fold_te)) if fold_te[0] else None

        # Build model
        model = build_model(model_name, best_hpo, fold_is, fold_os, fold_ml, device)
        eval_only_dir = os.getenv("XLAB_EVAL_ONLY", "").strip()   # folder of fold_<subject>.pt from an earlier run: no training
        if eval_only_dir:
            sd = torch.load(Path(eval_only_dir) / f"fold_{held_out}.pt", map_location=device)
            model.load_state_dict({k.replace("_orig_mod.", "", 1): v for k, v in sd.items()})
            print(f"  [eval-only] loaded {eval_only_dir}/fold_{held_out}.pt")
        model = maybe_compile(model)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
        criterion    = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=[1.0] * fold_os)
        optimizer    = optim.AdamW(
            model.parameters(),
            lr=best_hpo.get("learning_rate", float(os.getenv("LOSO_LR", "1e-4"))),
            weight_decay=best_hpo.get("weight_decay", 0.01),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # Train (skipped in eval-only mode)
        tl = vl = tm = vm = tr2 = vr2 = []
        if not eval_only_dir:
          tl, vl, tm, vm, tr2, vr2 = train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=loso_epochs, device=device,
            diagnostics=cfg.train_diagnostics,
            target_feature_names=target_columns,
            early_stopping_patience=cfg.early_stopping_patience,
            early_stopping_min_delta=cfg.early_stopping_min_delta,
        )

        # Log to W&B
        if wb_run and tl:
            for ep in range(len(tl)):
                log_d = {"epoch": ep + 1, "train/loss": tl[ep], "train/mae": tm[ep], "train/r2": tr2[ep]}
                if vl and ep < len(vl):
                    log_d.update({"val/loss": vl[ep], "val/mae": vm[ep], "val/r2": vr2[ep]})
                wandb.log(log_d)

        # Evaluate on held-out subject
        fold_metric: Dict = {"subject": held_out}
        if test_ds:
            m, r2, mae = _evaluate_primary(model, test_ds, batch_size, device, target_columns, primary_tag)
            fold_metric.update(m)
            print(f"  {primary_tag.upper()} held-out {held_out}: mean_R²={r2:.4f}  mean_MAE={mae:.4f}")

        # Evaluate on cross-lab subject
        if cross_lab_trials:
            try:
                m, r2, mae = _evaluate_cross_lab(
                    model, cross_lab_trials, fold_normalizer,
                    batch_size, device, target_columns, xlab_tag,
                    input_normalizer=fold_input_norm,
                )
                fold_metric.update(m)
                print(f"  {xlab_tag.upper()} cross-lab {cfg.cross_lab_subject}: mean_R²={r2:.4f}  mean_MAE={mae:.4f}")
            except Exception as exc:
                print(f"  Cross-lab eval failed for fold {held_out}: {exc}")

        fold_metrics.append(fold_metric)

        # W&B finish
        if wb_run:
            wandb.summary.update(fold_metric)
            wandb.finish()

        # Save fold model
        if not eval_only_dir:
            torch.save(model.state_dict(), models_dir / f"fold_{held_out}.pt")

    # ── Summary ─────────────────────────────────────────────────────────
    if fold_metrics:
        loso_df = pd.DataFrame(fold_metrics)
        loso_df.to_csv(results_dir / "loso_results.csv", index=False)
        print(f"\n{'=' * 50}")
        print(f"LOSO Summary — {len(fold_metrics)} folds (primary={primary_tag.upper()}, cross-lab={xlab_tag.upper()})")
        if f"{primary_tag}_r2" in loso_df.columns:
            print(f"  {primary_tag.upper()} held-out : mean_R²={loso_df[f'{primary_tag}_r2'].mean():.4f}"
                  f" ± {loso_df[f'{primary_tag}_r2'].std():.4f}"
                  f"  mean_MAE={loso_df[f'{primary_tag}_mae'].mean():.4f}"
                  f" ± {loso_df[f'{primary_tag}_mae'].std():.4f}")
        if f"{xlab_tag}_r2" in loso_df.columns:
            print(f"  {xlab_tag.upper()} cross-lab: mean_R²={loso_df[f'{xlab_tag}_r2'].mean():.4f}"
                  f" ± {loso_df[f'{xlab_tag}_r2'].std():.4f}"
                  f"  mean_MAE={loso_df[f'{xlab_tag}_mae'].mean():.4f}"
                  f" ± {loso_df[f'{xlab_tag}_mae'].std():.4f}")
        print(f"{'=' * 50}")
        print(f"Results saved to {results_dir / 'loso_results.csv'}")
    else:
        print("LOSO: no folds completed successfully.")

    # ── Post-analysis plots ─────────────────────────────────────────────
    generate_loso_plots(results_dir, plots_dir, cfg.selected_dataset)

    print("\nLOSO run complete — skipping standard pipeline.")
    sys.exit(0)
