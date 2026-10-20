"""
Unit-conversion and feature-index helpers shared across training pipelines.

These were previously duplicated in:
  - src/utils/main_script_helpers.py
  - src/analysis/post_analysis.py
  - src/inference.py (retired to archive/broken/ on 2026-10-02)
"""

from __future__ import annotations

import numpy as np


def metadata_weight_to_bw_newtons(weight_array) -> np.ndarray:
    """Convert a weight array (kg or N) to body-weight in Newtons.

    If the value is > 200 it is assumed to already be in Newtons;
    otherwise it is multiplied by 9.81.
    """
    weights = np.asarray(weight_array, dtype=np.float32)
    bw_newtons = np.where(weights > 200.0, weights, weights * 9.81)
    return np.clip(bw_newtons, a_min=1e-6, a_max=None)


def convert_nbw_targets_to_newtons(array_3d, metadata_2d, force_indices):
    """Scale normalised-body-weight force channels back to Newtons.

    Args:
        array_3d:      (N, T, F) NumPy array of targets.
        metadata_2d:   (N, M) array; column 0 is subject mass (kg or N).
        force_indices: List[int] of output columns that are force channels.

    Returns:
        array_3d with force columns rescaled (copy, original unchanged).
    """
    if array_3d is None or metadata_2d is None or len(force_indices) == 0:
        return array_3d
    converted = np.array(array_3d, copy=True)
    bw = metadata_weight_to_bw_newtons(metadata_2d[:, 0]).reshape(-1, 1, 1)
    converted[:, :, force_indices] = converted[:, :, force_indices] * bw
    return converted


def force_feature_indices(columns) -> list:
    """Return the indices of force (non-CoP) target columns.

    Force columns end in ``_ml``, ``_ap``, or ``_vert``.
    """
    suffixes = ("_ml", "_ap", "_vert")
    return [
        i for i, name in enumerate(columns)
        if isinstance(name, str) and name.lower().endswith(suffixes)
    ]
