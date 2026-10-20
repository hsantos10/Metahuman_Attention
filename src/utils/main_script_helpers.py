"""Small helpers shared by the training entry points: typed environment-variable readers (`env_int`, ...),
`ensure_metadata_split_tuple` (inserts a dummy metadata list so every split is `(inputs, metadata, targets,
mask)`), and HPO-parameter file helpers."""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.conversions import (
    force_feature_indices,
    metadata_weight_to_bw_newtons as _metadata_weight_to_bw_newtons,
    convert_nbw_targets_to_newtons,
)


# ── Env-var helpers ─────────────────────────────────────────────────────────

def env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Warning: invalid integer for {name}={value!r}; using default {default}")
        return default


def env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Warning: invalid float for {name}={value!r}; using default {default}")
        return default


# ── Feature / loss weight helpers ──────────────────────────────────────────

def build_output_feature_weights(target_columns):
    feature_weights = []
    for column_name in target_columns:
        col = str(column_name).lower()
        if '_vert' in col:
            feature_weights.append(2.0)
        elif '_ap' in col or '_ml' in col:
            feature_weights.append(1.0)
        else:
            feature_weights.append(0.5)
    return feature_weights


# force_feature_indices, _metadata_weight_to_bw_newtons, convert_nbw_targets_to_newtons
# are now imported from src.utils.conversions (canonical single-source implementations).


# ── Tuple helpers ──────────────────────────────────────────────────────────

def ensure_metadata_split_tuple(split_tuple, metadata_dim=2):
    if not split_tuple or not split_tuple[0]:
        return split_tuple

    if len(split_tuple) == 4:
        # Already has metadata: (trc, meta, target, mask)
        return split_tuple
    elif len(split_tuple) == 3:
        # No metadata yet: (trc, target, mask) → inject dummy metadata
        trc, target, mask = split_tuple
        dummy_meta = torch.zeros(metadata_dim, dtype=torch.float32)
        meta_list = [dummy_meta for _ in trc]
        return (trc, meta_list, target, mask)
    else:
        raise ValueError(f"Unexpected split_tuple length {len(split_tuple)}: expected 3 or 4")


# ── HPO / fold-selection helpers ─────────────────────────────────────────

def load_hpo_params(studies_dir: Path) -> dict:
    """Load best HPO params from studies directory (JSON → pkl fallback)."""
    json_path = studies_dir / "best_params.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    json_path2 = studies_dir / "best_hpo_params.json"
    if json_path2.exists():
        with open(json_path2) as f:
            return json.load(f)
    pkl = studies_dir / "loso_pre_hpo.pkl"
    if pkl.exists():
        with open(pkl, 'rb') as f:
            study = pickle.load(f)
        return study.best_params
    return {}


def select_best_fold(loso_csv_path: Path, primary_tag: str = "gl") -> str:
    """Return the fold subject with the highest primary R²."""
    df = pd.read_csv(loso_csv_path)
    r2_col = f'{primary_tag}_r2'
    best_idx = df[r2_col].idxmax()
    return df.loc[best_idx, 'subject']
