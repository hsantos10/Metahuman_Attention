# -*- coding: utf-8 -*-
"""
Main training script — lab-agnostic GRF prediction.

Workflow:
  1. Load config (YAML + .env overrides)              → config_loader
  2. Pair marker/force files via dataset adapter
  3. Prepare data (normalise, split, augment)
  4a. LOSO cross-validation pipeline                   → loso_pipeline
  4b. Standard HPO + final training pipeline            → standard_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as `python chapter1/pipeline/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import TensorDataset

from src.utils.main_script_helpers import ensure_metadata_split_tuple
from src.pipeline.config_loader import load_config, setup_dataset_adapter
from src.pipeline.prepare_data import _prepare_data


# ── Data preparation lives in src/pipeline/prepare_data.py (shared with Chapter 2) ──


def _make_datasets(train_tuple, val_tuple, test_tuple, use_metadata):
    """Convert split tuples → TensorDatasets (injecting dummy metadata if needed)."""
    if not use_metadata:
        train_tuple = ensure_metadata_split_tuple(train_tuple, metadata_dim=2)
        val_tuple   = ensure_metadata_split_tuple(val_tuple, metadata_dim=2)
        test_tuple  = ensure_metadata_split_tuple(test_tuple, metadata_dim=2)

    def _to_ds(t):
        if not t or not t[0]:
            return None
        return TensorDataset(*(torch.stack(x) for x in t))

    return _to_ds(train_tuple), _to_ds(val_tuple), _to_ds(test_tuple)


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Config
    cfg = load_config()
    adapter = setup_dataset_adapter(cfg)

    print(f"Dataset: {cfg.selected_dataset}  |  Adapter: {adapter.__class__.__name__}")
    print(
        f"Run: {cfg.run_name}  |  force_mode={cfg.force_target_mode}  "
        f"|  target_norm={cfg.target_normalization_mode}  "
        f"|  diagnostics={cfg.train_diagnostics}  "
        f"|  skip_completed={cfg.skip_completed_models}"
    )

    # 2. Data
    data = _prepare_data(cfg, adapter)

    # 3. Route to pipeline
    if cfg.use_loso:
        from chapter1.pipeline.loso_pipeline import run_loso_pipeline
        run_loso_pipeline(
            cfg,
            prepare_args=data["prepare_args"],
            train_data_tuple=data["train_tuple"],
            val_data_tuple=data["val_tuple"],
            paired_trc_files=data["trc_files"],
            input_size=data["input_size"],
            output_size=data["output_size"],
            max_length=data["max_length"],
            essential_markers=data["essential_markers"],
            use_metadata=data["use_metadata"],
        )
        # run_loso_pipeline calls sys.exit(0) internally
    else:
        from chapter1.pipeline.standard_pipeline import run_standard_pipeline

        train_ds, val_ds, test_ds = _make_datasets(
            data["train_tuple"], data["val_tuple"], data["test_tuple"],
            data["use_metadata"],
        )
        run_standard_pipeline(
            cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            test_data_tuple=data["test_tuple"],
            input_size=data["input_size"],
            output_size=data["output_size"],
            max_length=data["max_length"],
            input_feature_names=data["input_feature_names"],
            feature_loss_weights=data["feature_weights"],
            target_normalizer=data["target_normalizer"],
            use_metadata=data["use_metadata"],
        )


if __name__ == "__main__":
    main()
