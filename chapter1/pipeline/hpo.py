# -*- coding: utf-8 -*-
"""
Hyperparameter optimisation helpers.

Provides Optuna objective factories for both the LOSO pre-HPO and the
standard per-model HPO loop.  Also handles study persistence and optional
W&B callback wiring.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.loss import NormalizedPeakWeightedLoss
from src.model import ImprovedConvFormer
from src.train import train_and_evaluate
from src.pipeline.model_factory import build_model

from src.pipeline.config_loader import RunConfig

# Optional integrations
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from optuna_integration.wandb import WeightsAndBiasesCallback
    HAS_OPTUNA_WANDB = True
except ImportError:
    HAS_OPTUNA_WANDB = False


# ── LOSO pre-HPO objective ──────────────────────────────────────────────────

def make_loso_hpo_objective(
    cfg: RunConfig,
    train_ds: TensorDataset,
    val_ds: Optional[TensorDataset],
    input_size: int,
    output_size: int,
    max_length: int,
    target_columns: List[str],
    model_name: str = "ImprovedConvFormer",
):
    """Return a callable ``objective(trial)`` for the pre-LOSO HPO study."""

    def objective(trial: optuna.Trial) -> float:
        params: dict = {}

        if model_name == "CNN":
            lr   = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            bs   = trial.suggest_categorical("batch_size", [8, 16, 32])
            dr   = trial.suggest_float("dropout", 0.05, 0.40)
            wd   = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
            params["num_filters"] = trial.suggest_categorical("num_filters", [128, 256, 512])
            params["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 7, 15])
            params["dropout"]     = dr
        elif model_name == "CNN_LSTM":
            lr   = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
            bs   = trial.suggest_categorical("batch_size", [8, 16, 32])
            dr   = trial.suggest_float("dropout", 0.05, 0.40)
            wd   = trial.suggest_float("weight_decay", 0.001, 0.05, log=True)
            params["hidden_size"] = trial.suggest_categorical("hidden_size", [256, 512])
            params["num_layers"]  = trial.suggest_categorical("num_layers", [1, 2])
            params["num_filters"] = trial.suggest_categorical("num_filters", [128, 256])
            params["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 7])
            params["dropout"]     = dr
        elif cfg.selected_dataset == "PATIENT_HANDLING":
            # Broad HPO search space for Patient Handling — prioritises
            # regularisation to combat overfitting (train R²~0.92 vs val 0.14).
            lr   = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
            bs   = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
            dr   = trial.suggest_float("dropout", 0.15, 0.50)
            dm   = trial.suggest_categorical("d_model", [64, 128, 256, 512])
            nl   = trial.suggest_categorical("num_encoder_layers", [1, 2, 3, 4, 6])
            ffn  = trial.suggest_categorical("dim_feedforward", [256, 512, 1024, 2048])
            cnn  = trial.suggest_categorical("num_cnn_filters", [64, 128, 256])
            nhead = trial.suggest_categorical("nhead", [4, 8])
            wd   = trial.suggest_float("weight_decay", 0.005, 0.1, log=True)
            params.update(dict(d_model=dm, nhead=nhead, num_encoder_layers=nl,
                               dim_feedforward=ffn, num_cnn_filters=cnn, dropout=dr))
        else:
            lr   = trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True)
            bs   = 8
            dr   = trial.suggest_float("dropout", 0.05, 0.20)
            dm   = 128
            nl   = trial.suggest_categorical("num_encoder_layers", [2, 3, 4])
            ffn  = 1024
            cnn  = 128
            nhead = 8
            wd   = 0.01
            params.update(dict(d_model=dm, nhead=nhead, num_encoder_layers=nl,
                               dim_feedforward=ffn, num_cnn_filters=cnn, dropout=dr))

        # d_model / nhead divisibility only relevant for Transformer models
        if model_name == "ImprovedConvFormer" and params.get("d_model", 128) % params.get("nhead", 8) != 0:
            raise optuna.exceptions.TrialPruned()

        params["learning_rate"] = lr
        params["batch_size"]    = bs
        params["dropout"]       = dr

        model = build_model(model_name, params, input_size, output_size, max_length, cfg.device)

        _wd = wd
        tl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        vl = DataLoader(val_ds, batch_size=bs, shuffle=False) if val_ds else None
        cr = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=[1.0] * output_size)
        op = optim.AdamW(model.parameters(), lr=lr, weight_decay=_wd)
        sc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(op, T_0=10, T_mult=2)

        tl_losses, vl_losses, tl_maes, vl_maes, tl_r2s, vl_r2s = train_and_evaluate(
            model, tl, vl, cr, op, sc,
            num_epochs=cfg.hpo_epochs, device=cfg.device,
            diagnostics=False, target_feature_names=target_columns,
        )

        # Log per-epoch metrics to W&B (custom metric tracking)
        if HAS_WANDB and wandb.run is not None:
            for ep in range(len(vl_losses)):
                log_d = {"epoch": ep + 1, "hpo/val_loss": vl_losses[ep]}
                if tl_losses and ep < len(tl_losses):
                    log_d["hpo/train_loss"] = tl_losses[ep]
                if vl_maes and ep < len(vl_maes):
                    log_d["hpo/val_mae"] = vl_maes[ep]
                if vl_r2s and ep < len(vl_r2s):
                    log_d["hpo/val_r2"] = vl_r2s[ep]
                wandb.log(log_d)

        return vl_losses[-1] if vl_losses else float("inf")

    return objective


# ── Standard per-model HPO objective ────────────────────────────────────────

def make_standard_hpo_objective(
    cfg: RunConfig,
    model_name: str,
    train_ds: TensorDataset,
    val_ds: Optional[TensorDataset],
    input_size: int,
    output_size: int,
    max_length: int,
    target_columns: List[str],
    feature_loss_weights: List[float],
):
    """Return a callable ``objective(trial)`` for the standard HPO study."""
    from src.pipeline.model_factory import build_model  # local to avoid circular

    def objective(trial: optuna.Trial) -> float:
        params: dict = {}

        if model_name == "LSTM_ATTENTION":
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.002, 0.003)
            params["batch_size"] = trial.suggest_categorical("batch_size", [4])
            params["dropout"] = trial.suggest_float("dropout", 0.2, 0.3)
            params["hidden_size"] = trial.suggest_categorical("hidden_size", [128])
            params["num_layers"] = trial.suggest_categorical("num_layers", [1])

        elif model_name == "CNN":
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.003, 0.004)
            params["batch_size"] = trial.suggest_categorical("batch_size", [4])
            params["dropout"] = trial.suggest_float("dropout", 0.08, 0.12)
            params["num_filters"] = trial.suggest_categorical("num_filters", [512])
            params["kernel_size"] = trial.suggest_categorical("kernel_size", [7])

        elif model_name == "CNN_LSTM":
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.0015, 0.0025)
            params["batch_size"] = trial.suggest_categorical("batch_size", [4])
            params["dropout"] = trial.suggest_float("dropout", 0.1, 0.15)
            params["hidden_size"] = trial.suggest_categorical("hidden_size", [512])
            params["num_layers"] = trial.suggest_categorical("num_layers", [1])
            params["num_filters"] = trial.suggest_categorical("num_filters", [256])
            params["kernel_size"] = trial.suggest_categorical("kernel_size", [3])

        elif model_name == "ImprovedConvFormer":
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            params["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16])
            params["dropout"] = trial.suggest_float("dropout", 0.05, 0.5)
            params["d_model"] = trial.suggest_categorical("d_model", [128, 256, 512])
            params["num_encoder_layers"] = trial.suggest_int("num_encoder_layers", 2, 6)
            params["nhead"] = 8
            params["dim_feedforward"] = trial.suggest_categorical("dim_feedforward", [512, 1024, 2048])
            params["num_cnn_filters"] = trial.suggest_categorical("num_cnn_filters", [64, 128, 256, 512])

            if params["d_model"] % params["nhead"] != 0:
                raise optuna.exceptions.TrialPruned()
        else:
            raise ValueError(f"Model {model_name} not configured in HPO loop.")

        model = build_model(model_name, params, input_size, output_size, max_length, cfg.device)
        tl = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        vl = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False) if val_ds else None
        cr = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=feature_loss_weights)
        op = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=0.01)
        sc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(op, T_0=10, T_mult=2)

        tl_losses, val_losses, tl_maes, val_maes, tl_r2s, val_r2s = train_and_evaluate(
            model, tl, vl, cr, op, sc,
            num_epochs=cfg.hpo_epochs, device=cfg.device,
            diagnostics=cfg.train_diagnostics,
            target_feature_names=target_columns,
        )

        # Log per-epoch metrics to W&B (custom metric tracking)
        if HAS_WANDB and wandb.run is not None:
            for ep in range(len(val_losses)):
                log_d = {"epoch": ep + 1, "hpo/val_loss": val_losses[ep]}
                if tl_losses and ep < len(tl_losses):
                    log_d["hpo/train_loss"] = tl_losses[ep]
                if val_maes and ep < len(val_maes):
                    log_d["hpo/val_mae"] = val_maes[ep]
                if val_r2s and ep < len(val_r2s):
                    log_d["hpo/val_r2"] = val_r2s[ep]
                wandb.log(log_d)

        trial.set_user_attr("val_mae", val_maes[-1])
        trial.set_user_attr("val_r2", val_r2s[-1])
        return val_losses[-1]

    return objective


# ── Run a study with optional W&B ──────────────────────────────────────────

def run_study(
    objective,
    n_trials: int,
    *,
    wandb_project: str = "metahuman-grf",
    wandb_group: str = "",
    wandb_tags: Optional[List[str]] = None,
    wandb_job_type: Optional[str] = None,
    study_name: Optional[str] = None,
) -> optuna.Study:
    """Create + optimise an Optuna study, optionally logging to W&B.

    Each trial is logged as a separate W&B run (``as_multirun=True``) with a
    unique name derived from the study name and trial number.  The
    ``@wandbc.track_in_wandb()`` decorator is applied to the objective so that
    custom metrics (training loss, accuracy, etc.) can be logged directly to
    W&B from inside the objective via ``wandb.log()``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = study_name or f"hpo_{timestamp}"

    wandbc = None
    if HAS_OPTUNA_WANDB and HAS_WANDB:
        wandbc = WeightsAndBiasesCallback(
            metric_name="val_loss",
            wandb_kwargs=dict(
                project=wandb_project,
                group=wandb_group,
                job_type=wandb_job_type,
                tags=wandb_tags or [],
            ),
            as_multirun=True,
        )

        # Wrap the objective so each trial's W&B run gets a unique name.
        # track_in_wandb() calls wandb.init() before the objective and
        # wandb.finish() after, so the rename happens while the run is active.
        raw_objective = objective

        def _named_objective(trial: optuna.Trial) -> float:
            if wandb.run is not None:
                wandb.run.name = f"hpo/trial_{trial.number:03d}"
            return raw_objective(trial)

        objective = wandbc.track_in_wandb()(_named_objective)
        print(f"  [W&B] Optuna WeightsAndBiasesCallback active "
              f"(project={wandb_project}, study={base_name})")

    study = optuna.create_study(direction="minimize", study_name=base_name)
    study.optimize(objective, n_trials=n_trials, callbacks=[wandbc] if wandbc else [])
    return study


# ── Persistence helpers ─────────────────────────────────────────────────────

def save_study(study: optuna.Study, output_dir: Path, model_name: str) -> None:
    """Persist the Optuna study as pickle, CSV summary, and best-params JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"{model_name}_optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / f"{model_name}_trials_summary.csv", index=False)

    best_params = dict(study.best_trial.params)
    if model_name == "ImprovedConvFormer":
        best_params.setdefault("num_encoder_layers", 2)
        best_params.setdefault("nhead", 8)

    with open(output_dir / f"{model_name}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"  Study artefacts saved to {output_dir}")
    return best_params
