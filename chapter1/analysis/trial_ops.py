"""
Trial-level tensor operations shared across training pipelines.

These were previously duplicated (3–5 copies each) across:
  - chapter1/pipeline/combined_hpo_cv.py
  - archive/ch1_figures_2026-03/{combined_post_analysis,loso_inference,attention_analysis}.py
  - archive/ch1_early/experiment_{fixed_combined,rotation_aug}.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import TensorDataset

from src.config import MARKER_NAME_MAP


def build_canonical_columns(essential_markers: List[str]) -> List[str]:
    """Build the full canonical feature column list (base + vel/acc + rel-pos).

    The column order matches what ``prepare_dataset_for_modeling`` produces:
      - Base XYZ columns (sorted alphabetically)
      - Each base column expanded to: col, col_vel, col_acc
      - 10 relative-position marker pairs × 3 axes

    Args:
        essential_markers: Short marker names (e.g. ``["LSHO", "RSHO", ...]``).

    Returns:
        Ordered list of feature column name strings.
    """
    canon_temp = []
    for sn in essential_markers:
        fn = MARKER_NAME_MAP.get(sn, sn)
        canon_temp.extend([f"{fn}_X", f"{fn}_Y", f"{fn}_Z"])
    canon_base = sorted(canon_temp)

    # Velocity + acceleration columns
    canon_vel_acc: List[str] = []
    for c in canon_base:
        canon_vel_acc.extend([c, f"{c}_vel", f"{c}_acc"])

    # Relative position columns (10 anatomically motivated pairs)
    marker_pairs = [
        ("LSHO", "LELBL"), ("LELBL", "LWRL"), ("RSHO", "RELBL"), ("RELBL", "RWRL"),
        ("LKNE", "LANKL"), ("LANKL", "LTOE"), ("RKNE", "RANKM"), ("RANKM", "RTOE"),
        ("LSHO", "RSHO"), ("LASI", "LSHO"),
    ]
    canon_rel = [f"rel_{a}_{b}_{ax}" for a, b in marker_pairs for ax in "XYZ"]
    return canon_vel_acc + canon_rel


def trials_to_tensordataset(
    trial_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> TensorDataset:
    """Stack a list of ``(trc, meta, target, mask)`` tuples into a TensorDataset.

    Args:
        trial_list: List of 4-tuples where each element is a 1D/2D tensor for a
                    single trial.

    Returns:
        TensorDataset with four stacked tensors: (trc, meta, target, mask).
    """
    trc = torch.stack([t[0] for t in trial_list])
    meta = torch.stack([t[1] for t in trial_list])
    tgt = torch.stack([t[2] for t in trial_list])
    msk = torch.stack([t[3] for t in trial_list])
    return TensorDataset(trc, meta, tgt, msk)


def apply_znorm_to_trials(
    trial_list: List[Tuple],
    normalizer: Optional[dict],
) -> List[Tuple]:
    """Apply z-score normalisation to the target tensor in each trial.

    The normaliser dict has the format produced by
    ``src.utils.target_normalization.fit_target_normalizer``:
    ``{"indices": [...], "mean": [...], "std": [...]}``.

    Normalisation is only applied to the valid (non-padded) portion of each
    sequence, identified via the mask tensor.

    Args:
        trial_list: List of ``(trc, meta, tgt, mask)`` tuples.
        normalizer: Normaliser dict, or ``None`` to pass through unchanged.

    Returns:
        New list of tuples with normalised target tensors.
    """
    if normalizer is None:
        return trial_list

    indices = torch.tensor(normalizer["indices"], dtype=torch.long)
    means = torch.tensor(normalizer["mean"], dtype=torch.float32)
    stds = torch.tensor(normalizer["std"], dtype=torch.float32)

    normalized = []
    for trc, meta, tgt, mask in trial_list:
        tgt_n = tgt.clone()
        valid_len = int(mask.sum().item())
        if valid_len > 0:
            tgt_n[:valid_len, indices] = (tgt_n[:valid_len, indices] - means) / stds
        normalized.append((trc, meta, tgt_n, mask))
    return normalized
