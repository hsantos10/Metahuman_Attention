# -*- coding: utf-8 -*-
"""
Configuration loading and resolution.

Handles .env files, YAML configs, CLI args, and env-var overrides.
Produces a single RunConfig dataclass consumed by all pipeline stages.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch

from src.config import ESSENTIAL_MARKER_SHORT_NAMES_COMBINED, TARGET_COLUMNS  # noqa: F401
from src.data.dataset_adapters import get_dataset_adapter
from src.utils.main_script_helpers import env_int, env_float


# ── Defaults ────────────────────────────────────────────────────────────────

_DEFAULT_HPO_TRIALS = 50
_DEFAULT_HPO_EPOCHS = 25
_DEFAULT_SANITY_EPOCHS = 30
_DEFAULT_FINAL_EPOCHS = 500
_DEFAULT_EARLY_STOPPING_PATIENCE = 50
_DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.0
_DEFAULT_SANITY_LOSS_ABORT_THRESHOLD = 10.0
_DEFAULT_SANITY_ABORT_EPOCH_WINDOW = 2


# ── Dataclass ───────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """Immutable bag of all configuration consumed by the training pipelines."""

    # Identity
    run_name: str
    run_date: str
    selected_dataset: str  # GROUNDLINK | PATIENT_HANDLING | COMBINED

    # Targets / normalisation
    force_target_mode: str           # 'nbw' or 'newton'
    target_normalization_mode: str   # 'standard' or 'skip'
    target_columns: List[str] = field(default_factory=lambda: list(TARGET_COLUMNS))

    # Output base for NEW runs (default: runs/). Historical artifacts stay in
    # outputs/. Override with the OUTPUT_ROOT env var.
    output_root: str = "runs"

    # Processing flags
    strict_target_columns: bool = True
    # Plate-frame corrections (negate AP force and AP CoP, tare shear): written for Patient Handling. Read from the
    # YAML processing.coordinate_corrections via APPLY_COORD_CORRECTIONS_OVERRIDE since 2026-10-06 (was decided by
    # dataset != GROUNDLINK, which also applied it to the wind-turbine and AddBiomechanics data; docs/DATA_TRACE.md B5).
    apply_coordinate_corrections: bool = True
    enable_data_diagnostics: bool = False
    train_diagnostics: bool = False
    skip_completed_models: bool = True
    enable_biomech_preprocessing: bool = True
    use_bilateral_flip: bool = True

    # LOSO
    use_loso: bool = False
    cross_lab_subject: Optional[str] = None

    # Models
    models_to_run: List[str] = field(default_factory=lambda: ["ImprovedConvFormer"])

    # HPO
    hpo_trials: int = _DEFAULT_HPO_TRIALS
    hpo_epochs: int = _DEFAULT_HPO_EPOCHS

    # Training
    sanity_epochs: int = _DEFAULT_SANITY_EPOCHS
    final_epochs: int = _DEFAULT_FINAL_EPOCHS
    early_stopping_patience: int = _DEFAULT_EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = _DEFAULT_EARLY_STOPPING_MIN_DELTA
    sanity_loss_abort_threshold: float = _DEFAULT_SANITY_LOSS_ABORT_THRESHOLD
    sanity_abort_epoch_window: int = _DEFAULT_SANITY_ABORT_EPOCH_WINDOW
    loso_epochs: Optional[int] = None  # defaults to final_epochs when None

    # YAML config dict (raw, for adapter-level settings)
    yaml_cfg: dict = field(default_factory=dict)

    # Runtime (derived)
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    @property
    def effective_loso_epochs(self) -> int:
        return self.loso_epochs if self.loso_epochs is not None else self.final_epochs


# ── .env / YAML loaders ────────────────────────────────────────────────────

def _load_env_file(env_file_path: Path) -> None:
    """Read a .env file and set keys into ``os.environ``.

    Env vars already present in ``os.environ`` (e.g. set via the shell) are
    NOT overwritten — shell-level variables always take highest priority.
    """
    if not env_file_path.exists():
        return
    try:
        with open(env_file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip()
                if not key:
                    continue
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                # Only set if not already present (shell env vars take priority)
                if key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f"Warning: Failed to load .env file {env_file_path}: {e}")


def _load_yaml_config(yaml_path: str) -> dict:
    """Load a YAML config and set matching env-var defaults (only if unset)."""
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed — pip install pyyaml")
        return {}

    p = Path(yaml_path)
    if not p.exists():
        print(f"Warning: Config file {yaml_path} not found.")
        return {}

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    _YAML_TO_ENV = {
        ("dataset", "name"): "SELECTED_DATASET",
        ("targets", "force_mode"): "FORCE_TARGET_MODE",
        ("targets", "normalization"): "GROUNDLINK_TARGET_NORMALIZATION_MODE",
        ("processing", "max_length"): "GLOBAL_MAX_LEN_OVERRIDE",
        ("processing", "strict_target_columns"): "GROUNDLINK_STRICT_TARGET_COLUMNS",
        ("processing", "biomech_preprocessing"): "ENABLE_BIOMECH_PREPROCESSING",
        ("processing", "bilateral_flip"): "USE_BILATERAL_FLIP",
        ("processing", "data_diagnostics"): "GROUNDLINK_ENABLE_DATA_DIAGNOSTICS",
        ("processing", "coordinate_corrections"): "APPLY_COORD_CORRECTIONS_OVERRIDE",
        ("hpo", "num_trials"): "HPO_TRIALS_OVERRIDE",
        ("hpo", "num_epochs"): "HPO_EPOCHS_OVERRIDE",
        ("hpo", "epochs_per_trial"): "HPO_EPOCHS_OVERRIDE",
        ("training", "sanity_epochs"): "SANITY_EPOCHS_OVERRIDE",
        ("training", "final_epochs"): "FINAL_EPOCHS_OVERRIDE",
        ("training", "batch_size"): "LOSO_BATCH_SIZE",
        ("training", "learning_rate"): "LOSO_LR",
        ("early_stopping", "patience"): "EARLY_STOPPING_PATIENCE_OVERRIDE",
        ("early_stopping", "min_delta"): "EARLY_STOPPING_MIN_DELTA_OVERRIDE",
        ("loso", "epochs"): "LOSO_EPOCHS_OVERRIDE",
        ("cross_lab", "use_loso"): "USE_LOSO",
        ("cross_lab", "cross_lab_subject"): "CROSS_LAB_SUBJECT",
        ("sanity", "epochs"): "SANITY_EPOCHS_OVERRIDE",
        ("sanity", "loss_abort_threshold"): "SANITY_LOSS_ABORT_THRESHOLD_OVERRIDE",
        ("sanity", "abort_epoch_window"): "SANITY_ABORT_EPOCH_WINDOW_OVERRIDE",
        ("diagnostics", "train_diagnostics"): "TRAIN_ENABLE_DIAGNOSTICS",
        ("diagnostics", "skip_completed"): "SKIP_COMPLETED_MODELS",
    }

    for (section, key), env_key in _YAML_TO_ENV.items():
        if section in cfg and key in cfg[section]:
            val = cfg[section][key]
            if isinstance(val, bool):
                val = "1" if val else "0"
            if env_key not in os.environ:
                os.environ[env_key] = str(val)

    if "model" in cfg and "architectures" in cfg["model"]:
        archs = cfg["model"]["architectures"]
        if isinstance(archs, list) and "MODELS_TO_RUN" not in os.environ:
            os.environ["MODELS_TO_RUN"] = ",".join(archs)

    return cfg


# ── Env-var → bool helper ──────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no"}


# ── Public entry point ──────────────────────────────────────────────────────

def load_config() -> RunConfig:
    """Parse CLI args, load .env + YAML, resolve env vars → ``RunConfig``."""

    # This module lives at <repo>/src/pipeline/config_loader.py, so the repo
    # root is three levels up (pipeline → src → repo).
    repo_root = Path(__file__).resolve().parents[2]
    _load_env_file(repo_root / ".env")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file (e.g. configs/groundlink.yaml)")
    cli_args, _ = parser.parse_known_args()
    yaml_cfg: dict = {}
    if cli_args.config:
        yaml_cfg = _load_yaml_config(cli_args.config)

    run_date = datetime.now().strftime("%d-%m-%Y")

    # --- Dataset ---
    selected_dataset = os.getenv("SELECTED_DATASET", "GROUNDLINK").strip().upper()
    if selected_dataset not in {"PATIENT_HANDLING", "GROUNDLINK", "WIND_TURBINE", "ADDBIO", "COMBINED"}:
        print(f"Warning: Unsupported SELECTED_DATASET={selected_dataset!r}. Falling back to 'GROUNDLINK'.")
        selected_dataset = "GROUNDLINK"

    # --- Force target mode ---
    force_target_mode = os.getenv("FORCE_TARGET_MODE", "nbw").strip().lower()
    if force_target_mode not in {"nbw", "newton"}:
        print(f"Warning: Unsupported FORCE_TARGET_MODE={force_target_mode!r}. Falling back to 'nbw'.")
        force_target_mode = "nbw"

    # --- Target normalisation ---
    target_norm = os.getenv("GROUNDLINK_TARGET_NORMALIZATION_MODE", "standard").strip().lower()
    if target_norm not in {"skip", "standard"}:
        print(f"Warning: Unsupported target normalization mode={target_norm!r}. Falling back to 'skip'.")
        target_norm = "skip"

    # --- Models ---
    models = [m.strip() for m in os.getenv("MODELS_TO_RUN", "ImprovedConvFormer").split(",") if m.strip()]
    if not models:
        models = ["ImprovedConvFormer"]

    # --- LOSO cross-lab subject ---
    default_xlab_subject = "ph_1" if selected_dataset == "GROUNDLINK" else "gl_2"
    cross_lab_subject = os.getenv("CROSS_LAB_SUBJECT", default_xlab_subject)

    loso_epochs_raw = env_int("LOSO_EPOCHS_OVERRIDE", 0)

    return RunConfig(
        run_name=os.getenv("RUN_NAME_OVERRIDE", f"test_11_{run_date}"),
        run_date=run_date,
        selected_dataset=selected_dataset,
        output_root=os.getenv("OUTPUT_ROOT", "runs"),
        force_target_mode=force_target_mode,
        target_normalization_mode=target_norm,
        strict_target_columns=_env_bool("GROUNDLINK_STRICT_TARGET_COLUMNS", True),
        apply_coordinate_corrections=_env_bool("APPLY_COORD_CORRECTIONS_OVERRIDE", True),
        enable_data_diagnostics=_env_bool("GROUNDLINK_ENABLE_DATA_DIAGNOSTICS", False),
        train_diagnostics=_env_bool("TRAIN_ENABLE_DIAGNOSTICS", False),
        skip_completed_models=_env_bool("SKIP_COMPLETED_MODELS", True),
        enable_biomech_preprocessing=_env_bool("ENABLE_BIOMECH_PREPROCESSING", True),
        use_bilateral_flip=_env_bool("USE_BILATERAL_FLIP", True),
        use_loso=_env_bool("USE_LOSO", False),
        cross_lab_subject=cross_lab_subject,
        models_to_run=models,
        hpo_trials=env_int("HPO_TRIALS_OVERRIDE", _DEFAULT_HPO_TRIALS),
        hpo_epochs=env_int("HPO_EPOCHS_OVERRIDE", _DEFAULT_HPO_EPOCHS),
        sanity_epochs=env_int("SANITY_EPOCHS_OVERRIDE", _DEFAULT_SANITY_EPOCHS),
        final_epochs=env_int("FINAL_EPOCHS_OVERRIDE", _DEFAULT_FINAL_EPOCHS),
        early_stopping_patience=env_int("EARLY_STOPPING_PATIENCE_OVERRIDE", _DEFAULT_EARLY_STOPPING_PATIENCE),
        early_stopping_min_delta=env_float("EARLY_STOPPING_MIN_DELTA_OVERRIDE", _DEFAULT_EARLY_STOPPING_MIN_DELTA),
        sanity_loss_abort_threshold=env_float("SANITY_LOSS_ABORT_THRESHOLD_OVERRIDE", _DEFAULT_SANITY_LOSS_ABORT_THRESHOLD),
        sanity_abort_epoch_window=env_int("SANITY_ABORT_EPOCH_WINDOW_OVERRIDE", _DEFAULT_SANITY_ABORT_EPOCH_WINDOW),
        loso_epochs=loso_epochs_raw if loso_epochs_raw > 0 else None,
        yaml_cfg=yaml_cfg,
    )


def setup_dataset_adapter(cfg: RunConfig):
    """Create the dataset adapter and apply YAML-driven exclusions."""
    adapter = get_dataset_adapter(cfg.selected_dataset)

    excluded = (cfg.yaml_cfg.get("dataset", {}) or {}).get("excluded_subjects", None)
    if excluded:
        excl_list = list(excluded)
        if hasattr(adapter, "EXCLUDED_SUBJECTS"):
            adapter.EXCLUDED_SUBJECTS = excl_list
            print(f"  [Config] Excluding subjects: {excl_list}")
        if hasattr(adapter, "_gl"):
            adapter._gl.EXCLUDED_SUBJECTS = excl_list
        if hasattr(adapter, "_ph"):
            adapter._ph.EXCLUDED_SUBJECTS = excl_list

    # Exclusion-sensitivity: re-include the normally-excluded subjects (their
    # documented calibration errors are repaired in the pipeline when
    # RECOVER_EXCLUDED=1). This wins over any YAML exclusions.
    if os.getenv("RECOVER_EXCLUDED", "0") == "1":
        for a in (adapter, getattr(adapter, "_gl", None), getattr(adapter, "_ph", None)):
            if a is not None and hasattr(a, "EXCLUDED_SUBJECTS"):
                a.EXCLUDED_SUBJECTS = []
        print("  [Config] RECOVER_EXCLUDED=1 -> including all subjects (corrections applied in-pipeline)")

    return adapter
