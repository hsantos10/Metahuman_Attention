# -*- coding: utf-8 -*-
"""Per-fold input (feature) normalization.

The per-trial input z-scoring in the data pipeline discards each trial's
absolute scale (speed, range of motion, and crucially acceleration magnitude,
which AP force depends on). When ``INPUT_NORM_MODE=per_fold`` the pipeline
disables the per-trial norm and instead fits a single normalizer on the
training-fold frames only (leakage-safe) and applies it to train/val/test and
the cross-lab evaluation trials.

Fit/apply operate on lists of per-trial ``(T, F)`` feature tensors plus their
``(T,)`` validity masks; ``apply_input_normalizer_to_batch`` handles the
stacked ``(N, T, F)`` cross-lab case.
"""
from __future__ import annotations

import numpy as np
import torch


def fit_input_normalizer(feature_tensors, mask_tensors):
    """Fit per-feature mean/std over the valid frames of the given trials.

    Returns a dict ``{"mean": (F,), "std": (F,)}`` (float32), or ``None`` if
    there are no valid frames. Zero-variance features get std=1.0.
    """
    chunks = []
    for f, m in zip(feature_tensors, mask_tensors):
        valid_len = int(m.sum().item())
        if valid_len > 0:
            chunks.append(f[:valid_len].detach().cpu().numpy())
    if not chunks:
        return None
    stacked = np.concatenate(chunks, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def apply_input_normalizer(feature_tensors, mask_tensors, normalizer):
    """Apply a fitted normalizer to a list of ``(T, F)`` trials (valid frames only)."""
    if normalizer is None:
        return list(feature_tensors)
    mean_t = torch.as_tensor(normalizer["mean"], dtype=torch.float32)
    std_t = torch.as_tensor(normalizer["std"], dtype=torch.float32)
    out = []
    for f, m in zip(feature_tensors, mask_tensors):
        valid_len = int(m.sum().item())
        g = f.clone()
        if valid_len > 0:
            g[:valid_len] = (g[:valid_len] - mean_t) / std_t
        out.append(g)
    return out


def apply_input_normalizer_to_batch(feature_stack, mask_stack, normalizer):
    """Apply a fitted normalizer to a stacked ``(N, T, F)`` batch (valid frames only)."""
    if normalizer is None:
        return feature_stack
    mean_t = torch.as_tensor(normalizer["mean"], dtype=torch.float32)
    std_t = torch.as_tensor(normalizer["std"], dtype=torch.float32)
    out = feature_stack.clone()
    for n in range(out.shape[0]):
        valid_len = int(mask_stack[n].sum().item())
        if valid_len > 0:
            out[n, :valid_len] = (out[n, :valid_len] - mean_t) / std_t
    return out
