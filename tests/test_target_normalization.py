# -*- coding: utf-8 -*-
"""Unit tests for src/utils/target_normalization.py.

Guards the two properties the LOSO protocol depends on:
  1. statistics are fit on the provided (training) frames only;
  2. normalize -> inverse round-trips back to the original values.
"""
from __future__ import annotations

import numpy as np
import torch

from src.utils.target_normalization import (
    fit_target_normalizer,
    inverse_target_normalization,
)

COLS = ["a", "b"]


def _trial(values, valid_len=None):
    """Build a (T, F) target tensor + a (T,) mask with `valid_len` valid frames."""
    t = torch.tensor(values, dtype=torch.float32)
    T = t.shape[0]
    valid_len = T if valid_len is None else valid_len
    mask = torch.zeros(T, dtype=torch.float32)
    mask[:valid_len] = 1.0
    return t, mask


def test_fit_uses_training_frames_only():
    # Two "train" trials; normalizer stats must match their pooled valid frames,
    # and must be independent of any held-out trial we never pass in.
    t1, m1 = _trial([[0.0, 10.0], [2.0, 12.0]])
    t2, m2 = _trial([[4.0, 14.0], [6.0, 16.0]])
    norm = fit_target_normalizer([t1, t2], [m1, m2], COLS)

    pooled = np.array([[0, 10], [2, 12], [4, 14], [6, 16]], dtype=np.float64)
    np.testing.assert_allclose(norm["mean"], pooled.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(norm["std"], pooled.std(axis=0), rtol=1e-6)


def test_fit_ignores_padded_frames():
    # Frames beyond the mask must not enter the statistics.
    t, m = _trial([[0.0, 10.0], [2.0, 12.0], [999.0, 999.0]], valid_len=2)
    norm = fit_target_normalizer([t], [m], COLS)
    np.testing.assert_allclose(norm["mean"], [1.0, 11.0], rtol=1e-6)


def test_inverse_roundtrip_recovers_original():
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((5, len(COLS))).astype(np.float32) * 3.0 + 5.0
    t, m = _trial(raw)
    norm = fit_target_normalizer([t], [m], COLS)

    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    normalized = ((raw - mean) / std)[None, :, :]           # (N=1, T, F)
    masks = np.ones((1, raw.shape[0]), dtype=np.float32)

    restored = inverse_target_normalization(normalized, masks, norm)
    np.testing.assert_allclose(restored[0], raw, rtol=1e-5, atol=1e-5)


def test_inverse_none_normalizer_is_noop():
    arr = np.ones((1, 3, 2), dtype=np.float32)
    out = inverse_target_normalization(arr, None, None)
    np.testing.assert_array_equal(out, arr)
