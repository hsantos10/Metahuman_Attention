# -*- coding: utf-8 -*-
"""Unit tests for src/utils/conversions.py (NBW <-> Newton scaling, force indices)."""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.conversions import (
    convert_nbw_targets_to_newtons,
    force_feature_indices,
    metadata_weight_to_bw_newtons,
)


def test_weight_kg_to_newtons():
    # < 200 is treated as kg and multiplied by g.
    out = metadata_weight_to_bw_newtons([70.0])
    assert out[0] == pytest.approx(70.0 * 9.81, rel=1e-6)


def test_weight_already_newtons_passthrough():
    # > 200 is assumed to already be Newtons.
    out = metadata_weight_to_bw_newtons([800.0])
    assert out[0] == pytest.approx(800.0, rel=1e-6)


def test_weight_clipped_positive():
    out = metadata_weight_to_bw_newtons([0.0])
    assert out[0] > 0.0  # clipped to >= 1e-6, never zero/negative


def test_force_feature_indices_selects_force_not_cop():
    cols = [
        "ground_force_1_ml", "ground_force_1_vert", "ground_force_1_ap",
        "ground_force_1_px", "ground_force_1_pz",
    ]
    # px / pz are CoP positions and must be excluded.
    assert force_feature_indices(cols) == [0, 1, 2]


def test_nbw_to_newton_roundtrip_scales_only_force_columns():
    rng = np.random.default_rng(0)
    cols = ["ground_force_1_ml", "ground_force_1_ap", "ground_force_1_px"]
    force_idx = force_feature_indices(cols)          # [0, 1]
    nbw = rng.standard_normal((3, 5, len(cols))).astype(np.float32)
    mass = np.array([[70.0], [80.0], [60.0]], dtype=np.float32)

    newtons = convert_nbw_targets_to_newtons(nbw, mass, force_idx)

    bw = metadata_weight_to_bw_newtons(mass[:, 0]).reshape(-1, 1, 1)
    # Force columns are scaled by body weight...
    np.testing.assert_allclose(newtons[:, :, force_idx], nbw[:, :, force_idx] * bw, rtol=1e-5)
    # ...and the CoP column is left untouched.
    np.testing.assert_allclose(newtons[:, :, 2], nbw[:, :, 2], rtol=1e-6)
    # Original array is not mutated.
    assert not np.shares_memory(newtons, nbw)
