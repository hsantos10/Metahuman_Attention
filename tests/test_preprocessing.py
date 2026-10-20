# -*- coding: utf-8 -*-
"""Unit tests for the physics-critical preprocessing steps.

Covers pelvis-root subtraction (translation invariance) and the
``rotate_force_horizontal`` co-rotation fix (Issue #1 in the pipeline audit):
horizontal-force magnitude and vertical force must be exactly preserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MARKER_NAME_MAP
from src.data.preprocessing import subtract_pelvis_root
from src.data.data_pipeline import rotate_force_horizontal


# ── Pelvis-root subtraction ──────────────────────────────────────────────────

def _col(short, ax):
    """Resolve a short marker name to its full DataFrame column, e.g. LKNE_X."""
    return f"{MARKER_NAME_MAP.get(short, short)}_{ax}"


_COORDS = {
    "LASI": ([1.0, 2.0], [0.0, 0.0], [0.0, 0.0]),
    "RASI": ([3.0, 4.0], [0.0, 0.0], [0.0, 0.0]),
    "LPSI": ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
    "RPSI": ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
    "LKNE": ([10.0, 10.0], [5.0, 5.0], [2.0, 2.0]),
}


def _marker_df(x_offset=0.0):
    data = {}
    for short, (xs, ys, zs) in _COORDS.items():
        data[_col(short, "X")] = [v + x_offset for v in xs]
        data[_col(short, "Y")] = list(ys)
        data[_col(short, "Z")] = list(zs)
    return pd.DataFrame(data)


def test_pelvis_origin_is_per_frame_mean():
    _, origin = subtract_pelvis_root(_marker_df())
    # pelvis_X = mean(LASI,RASI,LPSI,RPSI) per frame = mean(1,3,0,0)=1 ; mean(2,4,0,0)=1.5
    np.testing.assert_allclose(origin["pelvis_X"].to_numpy(), [1.0, 1.5], rtol=1e-6)


def test_pelvis_subtraction_makes_markers_relative():
    out, _ = subtract_pelvis_root(_marker_df())
    np.testing.assert_allclose(out[_col("LKNE", "X")].to_numpy(), [9.0, 8.5], rtol=1e-6)
    np.testing.assert_allclose(out[_col("LKNE", "Y")].to_numpy(), [5.0, 5.0], rtol=1e-6)


def test_pelvis_subtraction_is_translation_invariant():
    out0, _ = subtract_pelvis_root(_marker_df(x_offset=0.0))
    out100, _ = subtract_pelvis_root(_marker_df(x_offset=100.0))
    np.testing.assert_allclose(
        out0[_col("LKNE", "X")].to_numpy(), out100[_col("LKNE", "X")].to_numpy(), rtol=1e-6,
    )


# ── Force co-rotation (audit Issue #1) ───────────────────────────────────────

FORCE_COLS = ["ground_force_1_ml", "ground_force_1_ap", "ground_force_1_vert"]


def _force_seq(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((8, len(FORCE_COLS))).astype(np.float64)


@pytest.mark.parametrize("is_ph", [False, True])
def test_rotation_preserves_horizontal_magnitude(is_ph):
    seq = _force_seq()
    out = rotate_force_horizontal(seq, FORCE_COLS, heading_theta=0.7, is_patient_handling=is_ph)
    mag_before = seq[:, 0] ** 2 + seq[:, 1] ** 2   # ml^2 + ap^2
    mag_after = out[:, 0] ** 2 + out[:, 1] ** 2
    np.testing.assert_allclose(mag_after, mag_before, rtol=1e-10)


def test_rotation_leaves_vertical_unchanged():
    seq = _force_seq()
    out = rotate_force_horizontal(seq, FORCE_COLS, heading_theta=0.7, is_patient_handling=False)
    np.testing.assert_allclose(out[:, 2], seq[:, 2], rtol=1e-12)


def test_rotation_actually_changes_horizontal_components():
    seq = _force_seq()
    out = rotate_force_horizontal(seq, FORCE_COLS, heading_theta=0.7, is_patient_handling=False)
    assert not np.allclose(out[:, :2], seq[:, :2])   # guard against a silent no-op


def test_rotation_none_theta_is_noop():
    seq = _force_seq()
    out = rotate_force_horizontal(seq, FORCE_COLS, heading_theta=None, is_patient_handling=False)
    np.testing.assert_array_equal(out, seq)
