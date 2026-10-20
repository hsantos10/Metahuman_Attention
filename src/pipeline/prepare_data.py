# -*- coding: utf-8 -*-
"""Pair files, load every trial, split, and normalise targets — shared by both chapters.

`_prepare_data(cfg, adapter)` is the first step of every training script: Chapter 1's
`chapter1/pipeline/main.py` and the Chapter 2 driver `chapter2/a0_wt_loso.py` both call it.
It lived in `scripts/main.py` until 2026-10-02 and was moved here verbatim so that Chapter 2
does not import Chapter 1 code.
"""

from __future__ import annotations

import sys

import pandas as pd

from src.data.data_pipeline import prepare_dataset_for_modeling
from src.data.groundlink_adapter import set_groundlink_mass_map
from src.utils.main_script_helpers import env_int
from src.utils.target_normalization import (
    apply_target_normalizer,
    fit_target_normalizer,
)
from src.pipeline.config_loader import RunConfig


def _prepare_data(cfg: RunConfig, adapter):
    """Pair files, load trials, prepare train/val/test splits, and normalise targets.

    Returns a dict with everything downstream pipelines need.
    """
    # ── Pair files ──────────────────────────────────────────────────────
    pairing = adapter.pair_files()
    trc_files  = pairing.trc_files
    data_files = pairing.data_files

    max_paired = env_int("MAX_PAIRED_FILES_OVERRIDE", 0)
    if max_paired > 0:
        trc_files  = trc_files[:max_paired]
        data_files = data_files[:max_paired]
        print(f"INFO: MAX_PAIRED_FILES_OVERRIDE → first {len(trc_files)} pairs")

    print(f"Paired files: {len(trc_files)}")
    if not trc_files:
        sys.exit("No paired TRC/DATA files found for the selected dataset.")

    # ── Metadata ────────────────────────────────────────────────────────
    use_metadata = adapter.use_metadata
    essential_markers = adapter.essential_markers

    metadata_df, metadata_cfg = adapter.load_metadata()
    if use_metadata and metadata_df is None:
        print("Warning: Metadata requested but unavailable — continuing without.")
        use_metadata = False

    if cfg.selected_dataset == "GROUNDLINK" and metadata_df is not None:
        metadata_df = metadata_df.copy()
        metadata_df.index = [f"gl_{idx}" for idx in metadata_df.index]
    elif cfg.selected_dataset == "PATIENT_HANDLING" and metadata_df is not None:
        metadata_df = metadata_df.copy()
        metadata_df.index = [f"ph_{idx}" for idx in metadata_df.index]
    elif cfg.selected_dataset == "WIND_TURBINE" and metadata_df is not None:
        metadata_df = metadata_df.copy()
        metadata_df.index = [f"wt_{idx}" for idx in metadata_df.index]

    # Since 2026-10-06 the YAML flag decides (processing.coordinate_corrections -> APPLY_COORD_CORRECTIONS_OVERRIDE);
    # GroundLink never gets the corrections. Before, every non-GroundLink dataset got them (docs/DATA_TRACE.md B5).
    apply_force_corrections = cfg.selected_dataset != "GROUNDLINK" and cfg.apply_coordinate_corrections
    default_max_len = 1750 if cfg.selected_dataset == "GROUNDLINK" else 0
    max_len_val = env_int("GLOBAL_MAX_LEN_OVERRIDE", default_max_len)
    global_max_len = max_len_val if max_len_val > 0 else None

    # Inject GL mass map for NBW calculation inside the adapter
    if cfg.selected_dataset == "GROUNDLINK" and metadata_df is not None and metadata_cfg.weight_col:
        mass_map = {
            idx: float(row[metadata_cfg.weight_col])
            for idx, row in metadata_df.iterrows()
            if metadata_cfg.weight_col in metadata_df.columns and not pd.isna(row[metadata_cfg.weight_col])
        }
        if mass_map:
            set_groundlink_mass_map(mass_map)

    # ── Build prepare_args (reused by both pipelines) ───────────────────
    prepare_args = {
        "trc_file_paths": trc_files,
        "data_file_paths": data_files,
        "data_columns_to_use": cfg.target_columns,
        "essential_marker_names": essential_markers,
        "use_metadata": use_metadata,
        "nan_threshold": 0.10,
        "test_size": 0.1,
        "val_size": 0.1,
        "random_state": 42,
        "selected_dataset": cfg.selected_dataset,
        "apply_force_corrections": apply_force_corrections,
        "global_max_len_override": global_max_len,
        "strict_target_columns": cfg.strict_target_columns if cfg.selected_dataset == "GROUNDLINK" else False,
        "enable_data_diagnostics": cfg.enable_data_diagnostics if cfg.selected_dataset == "GROUNDLINK" else False,
        "force_target_mode": cfg.force_target_mode,
        "enable_biomech_preprocessing": cfg.enable_biomech_preprocessing,
        "use_bilateral_flip": cfg.use_bilateral_flip,
    }
    if use_metadata:
        prepare_args["metadata_df"] = metadata_df
        if metadata_cfg.weight_col:
            prepare_args["weight_col_in_metadata"] = metadata_cfg.weight_col
        if metadata_cfg.height_col:
            prepare_args["height_col_in_metadata"] = metadata_cfg.height_col
        if metadata_cfg.subject_id_col:
            prepare_args["subject_id_col_in_metadata"] = metadata_cfg.subject_id_col

    # ── Split dataset ───────────────────────────────────────────────────
    (train_tuple, val_tuple, test_tuple,
     max_length, input_size, output_size,
     input_feature_names) = prepare_dataset_for_modeling(**prepare_args)

    if not train_tuple[0]:
        sys.exit("No training data loaded.")

    # ── Target normalisation + feature weights ──────────────────────────
    # Unified pipeline: both GL and PH use uniform loss weights and the
    # same config-controlled target normalisation (standard z-score).
    target_normalizer = None
    feature_weights = [1.0] * len(cfg.target_columns)

    if cfg.target_normalization_mode == "standard":
        target_normalizer = _fit_and_apply_normalizer(
            train_tuple, val_tuple, test_tuple,
            use_metadata, cfg.target_columns,
        )
        if target_normalizer:
            train_tuple = apply_target_normalizer(train_tuple, use_metadata, target_normalizer)
            val_tuple   = apply_target_normalizer(val_tuple, use_metadata, target_normalizer)
            test_tuple  = apply_target_normalizer(test_tuple, use_metadata, target_normalizer)
        print(f"  {cfg.selected_dataset}: uniform weights + STANDARD target normalisation.")
    else:
        print(f"  {cfg.selected_dataset}: uniform weights, skip target normalisation.")

    print(f"Feature weights: {feature_weights}")
    print(f"Force mode: {cfg.force_target_mode}  |  Biomech: {cfg.enable_biomech_preprocessing}  |  Flip: {cfg.use_bilateral_flip}")

    return {
        "prepare_args": prepare_args,
        "train_tuple": train_tuple,
        "val_tuple": val_tuple,
        "test_tuple": test_tuple,
        "trc_files": trc_files,
        "max_length": max_length,
        "input_size": input_size,
        "output_size": output_size,
        "input_feature_names": input_feature_names,
        "feature_weights": feature_weights,
        "target_normalizer": target_normalizer,
        "use_metadata": use_metadata,
        "essential_markers": essential_markers,
    }


def _fit_and_apply_normalizer(train_tuple, val_tuple, test_tuple, use_metadata, target_columns):
    """Fit a target normaliser on the training split."""
    target_idx = 2 if use_metadata else 1
    mask_idx   = 3 if use_metadata else 2
    normalizer = fit_target_normalizer(
        train_tuple[target_idx], train_tuple[mask_idx], target_columns,
    )
    return normalizer
