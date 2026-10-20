# -*- coding: utf-8 -*-
"""
Standard (non-LOSO) training pipeline.

Steps:
  1. HPO per model (Optuna)
  2. Sanity check (short training to detect bad scaling)
  3. Final training with best hyperparams
  4. Evaluation on held-out test set
  5. Checkpoint, config, and result saving
  6. Post-analysis
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.loss import NormalizedPeakWeightedLoss
from src.train import analyze_results, train_and_evaluate, validate_model
from src.utils.main_script_helpers import (
    build_output_feature_weights,
    ensure_metadata_split_tuple,
    force_feature_indices,
    convert_nbw_targets_to_newtons,
)
from src.utils.target_normalization import inverse_target_normalization

from src.pipeline.config_loader import RunConfig
from chapter1.pipeline.hpo import make_standard_hpo_objective, run_study, save_study
from src.pipeline.model_factory import build_and_compile

try:
    from chapter1.analysis.visualization import (
        plot_model_comparison,
        plot_optuna_history,
        plot_optuna_param_scatter,
        plot_optuna_trial_scatter,
    )
    HAS_PLOTLY_VIZ = True
except ImportError:
    HAS_PLOTLY_VIZ = False


# ── Plotting helpers ────────────────────────────────────────────────────────

def _save_optuna_plots(study, trials_df, model_name: str, output_dir: Path) -> None:
    """Generate matplotlib/Plotly plots for an Optuna study."""
    try:
        if HAS_PLOTLY_VIZ:
            plot_optuna_history(trials_df, study.best_value, model_name, save_dir=str(output_dir))
            plot_optuna_trial_scatter(trials_df, study.best_value, model_name, save_dir=str(output_dir))
            plot_optuna_param_scatter(trials_df, model_name, save_dir=str(output_dir))
            print(f"  Plotly Optuna plots saved to {output_dir}")
        else:
            import matplotlib.pyplot as plt

        import matplotlib.pyplot as plt

        # Optimisation history
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(trials_df.index, trials_df["value"], marker="o", linewidth=2)
        ax.set(xlabel="Trial Number", ylabel="Validation Loss",
               title=f"{model_name} — Optimization History")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_optimization_history.png", dpi=150)
        plt.close()

        # Trial scatter
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["green" if t.value == study.best_value else "blue" for t in study.trials]
        ax.scatter(trials_df.index, trials_df["value"], c=colors, alpha=0.6, s=100)
        ax.axhline(y=study.best_value, color="red", linestyle="--", linewidth=2,
                    label=f"Best: {study.best_value:.4f}")
        ax.set(xlabel="Trial Number", ylabel="Validation Loss",
               title=f"{model_name} — Trial Values Distribution")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_trial_values.png", dpi=150)
        plt.close()

        # Param-vs-loss scatter grid
        param_cols = [c for c in trials_df.columns if c.startswith("params_")]
        numeric_cols = [
            c for c in param_cols
            if pd.to_numeric(trials_df[c], errors="coerce").notna().sum() > 0
        ]
        if numeric_cols:
            n = len(numeric_cols)
            n_cols = 2
            n_rows = (n + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
            axes = np.array(axes).reshape(-1) if n_rows * n_cols > 1 else [axes]
            for idx, col in enumerate(numeric_cols):
                ax = axes[idx]
                x = pd.to_numeric(trials_df[col], errors="coerce")
                y = trials_df["value"]
                valid = x.notna() & y.notna()
                ax.scatter(x[valid], y[valid], alpha=0.75, s=50)
                ax.set(xlabel=col.replace("params_", ""), ylabel="Validation Loss",
                       title=f"{col.replace('params_', '')} vs Loss")
                ax.grid(True, alpha=0.3)
            for j in range(n, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            plt.savefig(output_dir / f"{model_name}_param_vs_loss_scatter.png", dpi=150)
            plt.close()

        print(f"  Matplotlib Optuna plots saved to {output_dir}")
    except Exception as e:
        print(f"  Warning: Could not generate all Optuna plots: {e}")


def _save_comparison_plots(results_df: pd.DataFrame, plots_dir: Path) -> None:
    """Save final model comparison bar charts."""
    try:
        if HAS_PLOTLY_VIZ:
            plot_model_comparison(results_df, save_dir=str(plots_dir))
            print(f"  Plotly comparison plots saved to {plots_dir}")
        else:
            import matplotlib.pyplot as plt
            import seaborn as sns

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        sns.barplot(x="model", y="test_mae", data=results_df, ax=axes[0])
        axes[0].set_title("Test MAE Comparison"); axes[0].tick_params(axis="x", rotation=45)
        sns.barplot(x="model", y="test_r2", data=results_df, ax=axes[1])
        axes[1].set_title("Test R² Comparison"); axes[1].tick_params(axis="x", rotation=45)
        sns.barplot(x="model", y="training_time_sec", data=results_df, ax=axes[2])
        axes[2].set_title("Training Time (s)"); axes[2].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / "model_comparison_plots.png")
        plt.close()
        print(f"  Comparison plots saved to {plots_dir / 'model_comparison_plots.png'}")
    except ImportError:
        print("  Could not generate comparison plots (matplotlib/seaborn missing).")


# ── Main entry point ────────────────────────────────────────────────────────

def run_standard_pipeline(
    cfg: RunConfig,
    train_dataset: TensorDataset,
    val_dataset: Optional[TensorDataset],
    test_dataset: Optional[TensorDataset],
    test_data_tuple,
    input_size: int,
    output_size: int,
    max_length: int,
    input_feature_names: List[str],
    feature_loss_weights: List[float],
    target_normalizer: Optional[Dict],
    use_metadata: bool,
) -> None:
    """Execute HPO → sanity → final training → evaluation for each model."""

    target_columns = cfg.target_columns
    device = cfg.device

    # ── Output directories ──────────────────────────────────────────────
    run_root    = Path(cfg.output_root) / cfg.run_name
    results_dir = run_root / "results"
    logs_dir    = run_root / "logs"
    plots_dir   = run_root / "plots"
    data_dir    = run_root / "data"
    studies_dir = run_root / "studies"
    models_dir  = run_root / "models"
    for d in [results_dir, logs_dir, plots_dir, data_dir, studies_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "model_comparison_results.csv"

    # Resume from previous run if results file exists
    if results_file.exists():
        print(f"Loading existing results from {results_file}...")
        prev_df = pd.read_csv(results_file)
        all_results: List[Dict] = prev_df.to_dict("records")
        completed_models = prev_df["model"].tolist()
    else:
        all_results = []
        completed_models: List[str] = []

    # ── Per-model experiment loop ───────────────────────────────────────
    for model_name in cfg.models_to_run:
        if cfg.skip_completed_models and model_name in completed_models:
            print(f"\n{'=' * 20} Skipping already completed model: {model_name} {'=' * 20}")
            continue

        print(f"\n{'=' * 20} Running Experiment for: {model_name} {'=' * 20}")

        # ── HPO ─────────────────────────────────────────────────────────
        objective = make_standard_hpo_objective(
            cfg, model_name, train_dataset, val_dataset,
            input_size, output_size, max_length,
            target_columns, feature_loss_weights,
        )
        study = run_study(
            objective, cfg.hpo_trials,
            wandb_group=cfg.run_name,
            wandb_tags=["HPO", "standard", cfg.selected_dataset, model_name],
            wandb_job_type=f"HPO-{model_name}",
            study_name=f"{cfg.run_name}_{model_name}",
        )

        study_dir = studies_dir / model_name
        best_params = save_study(study, study_dir, model_name)
        _save_optuna_plots(study, study.trials_dataframe(), model_name, study_dir)

        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        if model_name == "ImprovedConvFormer":
            best_params.setdefault("num_encoder_layers", 2)
            best_params.setdefault("nhead", 8)

        print(f"  Best trial — val_loss={best_trial.value:.4f}  params={best_params}")

        # ── Dataloaders with best batch size ────────────────────────────
        bs = best_params["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=bs, shuffle=False) if val_dataset else None
        test_loader  = DataLoader(test_dataset, batch_size=bs, shuffle=False) if test_dataset else None

        # ── Sanity check ────────────────────────────────────────────────
        print(f"--- Sanity check training ({cfg.sanity_epochs} epochs) for {model_name} ---")
        sanity_model = build_and_compile(model_name, best_params, input_size, output_size, max_length, device)
        sanity_crit  = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=feature_loss_weights)
        sanity_opt   = optim.AdamW(sanity_model.parameters(), lr=best_params["learning_rate"], weight_decay=0.01)
        sanity_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(sanity_opt, T_0=10, T_mult=2)

        sanity_tl, *_ = train_and_evaluate(
            sanity_model, train_loader, val_loader,
            sanity_crit, sanity_opt, sanity_sched,
            num_epochs=cfg.sanity_epochs, device=device,
            diagnostics=cfg.train_diagnostics,
            target_feature_names=target_columns,
        )
        if not sanity_tl:
            raise RuntimeError("Sanity check failed: no training loss values produced.")

        window = max(1, min(cfg.sanity_abort_epoch_window, len(sanity_tl)))
        best_early_loss = min(sanity_tl[:window])
        if best_early_loss > cfg.sanity_loss_abort_threshold:
            raise RuntimeError(
                f"Sanity aborted: best train loss in first {window} epoch(s) = {best_early_loss:.4f} "
                f"(threshold {cfg.sanity_loss_abort_threshold:.1f})."
            )
        print(f"  Sanity passed — best early loss = {best_early_loss:.4f}")

        # ── Final training ──────────────────────────────────────────────
        print(f"--- Final training ({cfg.final_epochs} epochs) for {model_name} ---")
        final_model = build_and_compile(model_name, best_params, input_size, output_size, max_length, device)
        criterion   = NormalizedPeakWeightedLoss(peak_weight_multiplier=2.0, feature_weights=feature_loss_weights)
        optimizer   = optim.AdamW(final_model.parameters(), lr=best_params["learning_rate"], weight_decay=0.01)
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        t0 = time.time()
        train_and_evaluate(
            final_model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            num_epochs=cfg.final_epochs, device=device,
            early_stopping_patience=cfg.early_stopping_patience,
            early_stopping_min_delta=cfg.early_stopping_min_delta,
            diagnostics=cfg.train_diagnostics,
            target_feature_names=target_columns,
        )
        training_time = time.time() - t0

        # ── Test evaluation ─────────────────────────────────────────────
        test_loss, test_mae, test_r2 = -1.0, -1.0, -1.0
        test_metrics_per_feature: Any = {}
        if test_loader:
            preds, actuals, masks = validate_model(final_model, test_loader, device)
            preds   = inverse_target_normalization(preds, masks, target_normalizer)
            actuals = inverse_target_normalization(actuals, masks, target_normalizer)

            if cfg.force_target_mode == "nbw":
                meta_np = (
                    torch.stack(test_data_tuple[1]).cpu().numpy()
                    if test_data_tuple and len(test_data_tuple) > 1 and test_data_tuple[1]
                    else None
                )
                fi = force_feature_indices(target_columns)
                preds   = convert_nbw_targets_to_newtons(preds, meta_np, fi)
                actuals = convert_nbw_targets_to_newtons(actuals, meta_np, fi)

            if preds.size > 0:
                res_df = analyze_results(preds, actuals, target_columns, masks)
                test_loss = float(res_df["MSE"].mean())
                test_mae  = float(res_df["MAE"].mean())
                test_r2   = float(res_df["R2"].mean())
                test_metrics_per_feature = res_df.to_dict("records")

        # ── Save checkpoint + config ────────────────────────────────────
        _save_checkpoint(
            final_model, best_params, model_name, cfg,
            input_size, output_size, max_length, input_feature_names,
            target_normalizer, best_trial, test_loss, test_mae, test_r2,
            training_time, models_dir,
        )

        # ── Accumulate results ──────────────────────────────────────────
        num_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
        all_results.append({
            "model": model_name,
            "best_val_loss": best_trial.value,
            "best_val_mae": best_trial.user_attrs.get("val_mae", -1),
            "best_val_r2": best_trial.user_attrs.get("val_r2", -1),
            "test_loss": test_loss,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "test_metrics_per_feature": str(test_metrics_per_feature),
            "num_params": num_params,
            "training_time_sec": training_time,
            "best_hyperparameters": str(best_params),
        })
        pd.DataFrame(all_results).to_csv(results_file, index=False)
        print(f"  Results for {model_name} saved to {results_file}")

    # ── Aggregate results ───────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    print("\n\n--- Model Comparison Results ---")
    print(results_df)
    results_df.to_csv(results_file, index=False)

    md_path = run_root / "model_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Model Comparison\n\n")
        f.write(results_df.to_markdown(index=False))
    print(f"  Markdown table: {md_path}")

    _save_comparison_plots(results_df, plots_dir)

    # ── Save analysis data ──────────────────────────────────────────────
    analysis_data = {
        "test_dataset": test_dataset,
        "target_feature_names": target_columns,
        "input_feature_names": input_feature_names,
        "models_to_run": cfg.models_to_run,
        "selected_dataset": cfg.selected_dataset,
        "input_size": input_size,
        "output_size": output_size,
        "max_length": max_length,
        "target_normalizer": target_normalizer,
        "force_target_mode": cfg.force_target_mode,
    }
    analysis_path = data_dir / "analysis_data.pt"
    torch.save(analysis_data, analysis_path)
    print(f"  Analysis data saved to {analysis_path}")

    # ── Auto post-analysis ──────────────────────────────────────────────
    try:
        from chapter1.analysis import post_analysis
        print("\n--- Running post-analysis ---")
        post_analysis.main(
            run_root=str(run_root),
            results_file=str(results_file),
            model_dir=str(models_dir),
            analysis_data_path=str(analysis_path),
        )
    except Exception as e:
        print(f"  Warning: Post-analysis failed: {e}")

    print("\n--- All experiments complete. ---")


# ── Private helpers ─────────────────────────────────────────────────────────

def _save_checkpoint(
    model, params, model_name, cfg: RunConfig,
    input_size, output_size, max_length, input_feature_names,
    target_normalizer, best_trial, test_loss, test_mae, test_r2,
    training_time, models_dir: Path,
) -> None:
    """Write the model checkpoint (.pt) and companion config (.json)."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": params,
        "model_architecture": model_name,
        "input_size": input_size,
        "output_size": output_size,
        "max_length": max_length,
        "num_epochs_trained": cfg.final_epochs,
        "best_val_loss": best_trial.value,
        "best_val_mae": best_trial.user_attrs.get("val_mae", -1),
        "best_val_r2": best_trial.user_attrs.get("val_r2", -1),
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "training_time_sec": training_time,
        "dataset": cfg.selected_dataset,
        "target_columns": cfg.target_columns,
        "input_feature_names": input_feature_names,
        "target_normalizer": target_normalizer,
    }
    torch.save(checkpoint, models_dir / f"{model_name}_model.pt")

    config_dict = {
        "model_architecture": model_name,
        "hyperparameters": params,
        "data_configuration": {
            "input_size": input_size,
            "output_size": output_size,
            "max_sequence_length": max_length,
            "dataset": cfg.selected_dataset,
            "target_columns": cfg.target_columns,
            "num_input_features": len(input_feature_names),
            "target_normalizer": target_normalizer,
        },
        "training_configuration": {
            "sanity_num_epochs": cfg.sanity_epochs,
            "final_num_epochs": cfg.final_epochs,
            "num_optuna_trials": cfg.hpo_trials,
            "num_hpo_epochs": cfg.hpo_epochs,
            "early_stopping_patience": cfg.early_stopping_patience,
            "early_stopping_min_delta": cfg.early_stopping_min_delta,
            "optimizer": "AdamW",
            "loss_function": "NormalizedPeakWeightedLoss",
            "scheduler": "CosineAnnealingWarmRestarts",
        },
        "performance_metrics": {
            "best_validation_loss": float(best_trial.value),
            "best_validation_mae": float(best_trial.user_attrs.get("val_mae", -1)),
            "best_validation_r2": float(best_trial.user_attrs.get("val_r2", -1)),
            "test_loss": float(test_loss),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
        },
        "model_info": {
            "num_parameters": num_params,
            "training_time_seconds": training_time,
        },
    }
    with open(models_dir / f"{model_name}_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"  Checkpoint + config saved to {models_dir}")
