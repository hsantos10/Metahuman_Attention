"""
augmentation.py — Data augmentation for biomechanical time series.

Bilateral flip (left/right mirror):
  Instantly doubles the training set and prevents the model from learning
  a dominant-leg or dominant-arm bias.  The operation is exact — it mirrors
  the subject about the sagittal plane (negates X / ML axis), swaps
  left↔right marker columns, then mirrors the force targets accordingly.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.config import MARKER_NAME_MAP, LEFT_RIGHT_MARKER_PAIRS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_col_index(columns: List[str]) -> dict:
    return {col: i for i, col in enumerate(columns)}


def _get_marker_col_indices(col_idx: dict, short_name: str) -> List[int]:
    """Return all column indices that belong to the given marker short name."""
    full_name = MARKER_NAME_MAP.get(short_name, short_name)
    return [v for k, v in col_idx.items() if k.startswith(full_name + "_")]


def _get_rel_col_indices(col_idx: dict, left_short: str, right_short: str) -> Tuple[List[int], List[int]]:
    """
    Return indices for rel_LEFT_* and rel_RIGHT_* columns as well as
    the implicit swap pattern when L and R have matching canonical pairs.
    """
    left_prefix  = f"rel_{left_short}_"
    right_prefix = f"rel_{right_short}_"

    # Find explicit L→ and R→ pairs that mirror each other
    # e.g. rel_LSHO_LELBL_X  ↔  rel_RSHO_RELBL_X
    left_cols  = {k: v for k, v in col_idx.items() if k.startswith(left_prefix)}
    right_cols = {k: v for k, v in col_idx.items() if k.startswith(right_prefix)}
    return left_cols, right_cols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bilateral_flip(
    trc_tensor:    torch.Tensor,
    target_tensor: torch.Tensor,
    mask_tensor:   torch.Tensor,
    canonical_cols: List[str],
    target_cols:    List[str],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Mirror the subject left↔right.

    Input
    -----
    trc_tensor    : (max_len, n_input_features)  — padded & normalised markers
    target_tensor : (max_len, n_targets)         — padded force/CoP targets
    mask_tensor   : (max_len,)                   — 1 for valid frames, 0 for pad
    canonical_cols : list of input feature names  (must match trc_tensor dim 1)
    target_cols    : list of target column names  (must match target_tensor dim 1)

    Returns
    -------
    Flipped (trc_tensor, target_tensor, mask_tensor) or None on error.
    """
    try:
        trc_np    = trc_tensor.numpy().copy()     # (T, F)
        tgt_np    = target_tensor.numpy().copy()  # (T, C)
        mask_np   = mask_tensor.numpy().copy()    # (T,)

        col_idx = _build_col_index(canonical_cols)

        # ------------------------------------------------------------------ #
        # (A) INPUT MARKERS                                                   #
        # ------------------------------------------------------------------ #

        # Step 1: Negate all _X columns (positional, vel, acc)
        x_axis_indices = [
            i for col, i in col_idx.items()
            if col.endswith("_X") or col.endswith("_X_vel") or col.endswith("_X_acc")
        ]
        trc_np[:, x_axis_indices] *= -1.0

        # Step 2: Swap left↔right absolute-marker column groups
        for left_sh, right_sh in LEFT_RIGHT_MARKER_PAIRS:
            l_idxs = _get_marker_col_indices(col_idx, left_sh)
            r_idxs = _get_marker_col_indices(col_idx, right_sh)
            if l_idxs and r_idxs and len(l_idxs) == len(r_idxs):
                # Must sort by suffix order so they match element-wise
                l_sorted = sorted(l_idxs, key=lambda i: canonical_cols[i])
                r_sorted = sorted(r_idxs, key=lambda i: canonical_cols[i])
                trc_np[:, l_sorted], trc_np[:, r_sorted] = (
                    trc_np[:, r_sorted].copy(),
                    trc_np[:, l_sorted].copy(),
                )

        # Step 3: Handle relative position columns
        # Mirror pairs defined in marker_pairs (same list as in data_pipeline.py)
        # The full set of relative pairs and how they map:
        #   rel_LSHO_LELBL ↔ rel_RSHO_RELBL  (X negated via step 1; values swap)
        #   rel_LELBL_LWRL ↔ rel_RELBL_RWRL
        #   rel_LKNE_LANKL ↔ rel_RKNE_RANKM
        #   rel_LANKL_LTOE ↔ rel_RANKM_RTOE
        #   rel_LSHO_RSHO  → stays same slot; X already negated in step 1
        #                     but Y/Z need negate too (see derivation in module docs)
        #   rel_LASI_LSHO  → would map to rel_RASI_RSHO which doesn't exist → skip
        rel_bilateral_pairs = [
            (('LSHO',  'LELBL'), ('RSHO',  'RELBL')),
            (('LELBL', 'LWRL'),  ('RELBL', 'RWRL')),
            (('LKNE',  'LANKL'), ('RKNE',  'RANKM')),
            (('LANKL', 'LTOE'),  ('RANKM', 'RTOE')),
        ]
        for (la, lb), (ra, rb) in rel_bilateral_pairs:
            # swap only if all four markers are in the active set: with the 20-marker set RWRL is absent and
            # rel_RELBL_RWRL is an always-zero column, so swapping would move real data to the right and zeros to
            # the left. An unpairable rel column stays mirrored in place, as an unpaired marker does (2026-10-02).
            if not all(_get_marker_col_indices(col_idx, m) for m in (la, lb, ra, rb)):
                continue
            for ax in ('X', 'Y', 'Z'):
                l_col = f"rel_{la}_{lb}_{ax}"
                r_col = f"rel_{ra}_{rb}_{ax}"
                li = col_idx.get(l_col)
                ri = col_idx.get(r_col)
                if li is not None and ri is not None:
                    trc_np[:, li], trc_np[:, ri] = (
                        trc_np[:, ri].copy(),
                        trc_np[:, li].copy(),
                    )

        # rel_LSHO_RSHO: X stays (negates cancel), Y and Z negate
        for ax in ('Y', 'Z'):
            col = f"rel_LSHO_RSHO_{ax}"
            idx = col_idx.get(col)
            if idx is not None:
                trc_np[:, idx] *= -1.0

        # CoM kinematics: COM_vel_X / COM_acc_X already negated by X-axis step (requires the caller to pass the
        # full column list incl. COM_*; data_pipeline passes input_feature_names). Y and Z need no swap.

        # ------------------------------------------------------------------ #
        # (B) FORCE / CoP TARGETS                                             #
        # ------------------------------------------------------------------ #
        tgt_col_idx = _build_col_index(target_cols)

        # Negate _ml (ML force changes sign)
        for col, i in tgt_col_idx.items():
            if isinstance(col, str) and col.lower().endswith('_ml'):
                tgt_np[:, i] *= -1.0

        # Negate _px (CoP X position mirrors)
        for col, i in tgt_col_idx.items():
            if isinstance(col, str) and col.lower().endswith('_px'):
                tgt_np[:, i] *= -1.0

        # Swap plate 1 ↔ plate 2 (left foot plate ↔ right foot plate)
        plate1_cols = {c: i for c, i in tgt_col_idx.items()
                       if isinstance(c, str) and c.startswith('ground_force_1_')}
        plate2_cols = {c: i for c, i in tgt_col_idx.items()
                       if isinstance(c, str) and c.startswith('ground_force_2_')}
        for p1_col, p1_i in plate1_cols.items():
            p2_col = p1_col.replace('ground_force_1_', 'ground_force_2_')
            p2_i = tgt_col_idx.get(p2_col)
            if p2_i is not None:
                tgt_np[:, p1_i], tgt_np[:, p2_i] = (
                    tgt_np[:, p2_i].copy(),
                    tgt_np[:, p1_i].copy(),
                )

        return (
            torch.tensor(trc_np, dtype=torch.float32),
            torch.tensor(tgt_np, dtype=torch.float32),
            torch.tensor(mask_np, dtype=torch.float32),
        )

    except Exception as e:
        print(f"Warning (bilateral_flip): augmentation failed: {e}")
        return None
