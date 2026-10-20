# -*- coding: utf-8 -*-
"""Unit tests for src/utils/input_normalization.py (per-fold input norm, M6)."""
from __future__ import annotations

import numpy as np
import torch

from src.utils.input_normalization import (
    apply_input_normalizer,
    apply_input_normalizer_to_batch,
    fit_input_normalizer,
)


def _trial(values, valid_len=None):
    t = torch.tensor(values, dtype=torch.float32)
    T = t.shape[0]
    valid_len = T if valid_len is None else valid_len
    mask = torch.zeros(T, dtype=torch.float32)
    mask[:valid_len] = 1.0
    return t, mask


def test_fit_pools_valid_frames_only():
    t1, m1 = _trial([[0.0, 100.0], [2.0, 102.0]])
    t2, m2 = _trial([[4.0, 104.0], [6.0, 106.0], [999.0, 999.0]], valid_len=2)
    norm = fit_input_normalizer([t1, t2], [m1, m2])
    pooled = np.array([[0, 100], [2, 102], [4, 104], [6, 106]], dtype=np.float64)
    np.testing.assert_allclose(norm["mean"], pooled.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(norm["std"], pooled.std(axis=0), rtol=1e-6)


def test_apply_standardizes_fit_set():
    t, m = _trial([[0.0], [2.0], [4.0], [6.0]])
    norm = fit_input_normalizer([t], [m])
    out = apply_input_normalizer([t], [m], norm)[0]
    # After standardization the valid frames have ~zero mean, unit std.
    np.testing.assert_allclose(out.numpy().mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(out.numpy().std(), 1.0, atol=1e-6)


def test_apply_leaves_padding_untouched():
    t, m = _trial([[1.0], [3.0], [777.0]], valid_len=2)
    norm = fit_input_normalizer([t], [m])
    out = apply_input_normalizer([t], [m], norm)[0]
    assert out[2, 0].item() == 777.0  # padded frame unchanged


def test_zero_variance_feature_is_safe():
    t, m = _trial([[5.0, 1.0], [5.0, 3.0]])   # col 0 constant
    norm = fit_input_normalizer([t], [m])
    assert norm["std"][0] == 1.0
    out = apply_input_normalizer([t], [m], norm)[0]
    assert np.isfinite(out.numpy()).all()


def test_batch_apply_matches_list_apply():
    t1, m1 = _trial([[0.0], [2.0], [0.0]], valid_len=2)
    t2, m2 = _trial([[4.0], [6.0], [0.0]], valid_len=2)
    norm = fit_input_normalizer([t1, t2], [m1, m2])
    list_out = apply_input_normalizer([t1, t2], [m1, m2], norm)
    stack = torch.stack([t1, t2])
    masks = torch.stack([m1, m2])
    batch_out = apply_input_normalizer_to_batch(stack, masks, norm)
    for i in range(2):
        vl = int(masks[i].sum().item())
        np.testing.assert_allclose(batch_out[i, :vl].numpy(), list_out[i][:vl].numpy(), rtol=1e-6)


def test_none_normalizer_is_noop():
    t, m = _trial([[1.0], [2.0]])
    out = apply_input_normalizer([t], [m], None)
    np.testing.assert_array_equal(out[0].numpy(), t.numpy())
