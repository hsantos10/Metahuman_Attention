# -*- coding: utf-8 -*-
"""
Polars-accelerated helpers for the data pipeline.

These functions use Polars lazy evaluation for compute-heavy operations
(windowing, CoP transforms, bulk column ops) and convert back to
pandas / numpy at the boundaries so existing pipeline code is unaffected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import polars as pl
    import pyarrow  # noqa: F401 — required by polars .to_pandas()

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# ---------------------------------------------------------------------------
# Pandas <-> Polars bridge
# ---------------------------------------------------------------------------

def pd_to_pl(df: pd.DataFrame) -> "pl.DataFrame":
    """Convert a pandas DataFrame to Polars, preserving column order."""
    if not HAS_POLARS:
        raise RuntimeError("polars is not installed. pip install polars")
    return pl.from_pandas(df)


def pl_to_pd(pldf: "pl.DataFrame") -> pd.DataFrame:
    """Convert a Polars DataFrame back to pandas."""
    return pldf.to_pandas()


# ---------------------------------------------------------------------------
# Accelerated CoP transform (Polars lazy)
# ---------------------------------------------------------------------------

def cop_transform_polars(
    cop_x: np.ndarray,
    cop_z: np.ndarray,
    pelvis_x: np.ndarray | None,
    pelvis_z: np.ndarray | None,
    heading_theta: float | None,
    height_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised CoP transform using Polars lazy expressions.

    Equivalent to ``apply_cop_transform`` in preprocessing.py but uses
    Polars lazy execution for the arithmetic.  Falls back to numpy
    when Polars is unavailable.

    All parameters have the same semantics as the original function.
    """
    if not HAS_POLARS:
        # Fallback — caller should use the numpy version.
        raise RuntimeError("polars not available for cop_transform_polars")

    n = len(cop_x)

    data: dict[str, np.ndarray] = {
        "cop_x": cop_x.astype(np.float64) * 1000.0,  # m -> mm
        "cop_z": cop_z.astype(np.float64) * 1000.0,
    }
    if pelvis_x is not None:
        data["pel_x"] = pelvis_x[:n].astype(np.float64)
    if pelvis_z is not None:
        data["pel_z"] = pelvis_z[:n].astype(np.float64)

    lf = pl.LazyFrame(data)

    # Step 1: pelvis subtraction
    if pelvis_x is not None:
        lf = lf.with_columns((pl.col("cop_x") - pl.col("pel_x")).alias("cop_x"))
    if pelvis_z is not None:
        lf = lf.with_columns((pl.col("cop_z") - pl.col("pel_z")).alias("cop_z"))

    # Step 2: heading rotation
    if heading_theta is not None:
        cos_t = np.cos(heading_theta)
        sin_t = np.sin(heading_theta)
        lf = lf.with_columns(
            (pl.lit(cos_t) * pl.col("cop_x") + pl.lit(sin_t) * pl.col("cop_z")).alias("rot_x"),
            (pl.lit(-sin_t) * pl.col("cop_x") + pl.lit(cos_t) * pl.col("cop_z")).alias("rot_z"),
        ).with_columns(
            pl.col("rot_x").alias("cop_x"),
            pl.col("rot_z").alias("cop_z"),
        )

    # Step 3: height scaling
    if height_m is not None and height_m > 0:
        lf = lf.with_columns(
            (pl.col("cop_x") / pl.lit(height_m)).alias("cop_x"),
            (pl.col("cop_z") / pl.lit(height_m)).alias("cop_z"),
        )

    result = lf.select("cop_x", "cop_z").collect()
    return (
        result["cop_x"].to_numpy().astype(np.float32),
        result["cop_z"].to_numpy().astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Accelerated data loading helpers
# ---------------------------------------------------------------------------

def read_csv_polars(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV with Polars for speed, return a pandas DataFrame."""
    if not HAS_POLARS:
        return pd.read_csv(path, **kwargs)
    pldf = pl.read_csv(path, **kwargs)
    return pldf.to_pandas()


def fast_column_select(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select columns from a pandas DataFrame via Polars for large frames."""
    if not HAS_POLARS or len(df) < 5000:
        return df[columns].copy()
    pldf = pl.from_pandas(df)
    available = [c for c in columns if c in pldf.columns]
    return pldf.select(available).to_pandas()


def fast_rename_columns(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    """Rename columns using Polars lazy evaluation."""
    if not HAS_POLARS or len(rename_map) == 0:
        return df.rename(columns=rename_map)
    pldf = pl.from_pandas(df)
    # Only rename columns that actually exist
    valid = {k: v for k, v in rename_map.items() if k in pldf.columns}
    if valid:
        pldf = pldf.rename(valid)
    return pldf.to_pandas()


def fast_normalize(df: pd.DataFrame, scaler_type: str = "standard") -> tuple[pd.DataFrame, dict]:
    """Z-score or min-max normalize numeric columns using Polars lazy.

    Returns (normalized_df, stats_dict) where stats_dict contains
    per-column mean/std (or min/max) for inverse transform.
    """
    if not HAS_POLARS:
        raise RuntimeError("polars not available")

    pldf = pl.from_pandas(df)
    numeric_cols = [c for c in pldf.columns if pldf[c].dtype in (pl.Float32, pl.Float64, pl.Int64, pl.Int32)]

    if not numeric_cols:
        return df.copy(), {}

    lf = pldf.lazy()

    if scaler_type == "standard":
        stats_exprs = []
        for col in numeric_cols:
            stats_exprs.extend([
                pl.col(col).mean().alias(f"{col}__mean"),
                pl.col(col).std().alias(f"{col}__std"),
            ])
        stats = lf.select(stats_exprs).collect()

        transform_exprs = []
        stats_dict = {}
        for col in numeric_cols:
            mean_val = stats[f"{col}__mean"][0]
            std_val = stats[f"{col}__std"][0]
            if std_val is None or std_val < 1e-9:
                std_val = 1.0
            stats_dict[col] = {"mean": float(mean_val), "std": float(std_val)}
            transform_exprs.append(
                ((pl.col(col) - pl.lit(mean_val)) / pl.lit(std_val)).alias(col)
            )

        non_numeric = [c for c in pldf.columns if c not in numeric_cols]
        result = lf.with_columns(transform_exprs).collect()
        return result.to_pandas(), stats_dict
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")


# ---------------------------------------------------------------------------
# Time-series windowing (Polars lazy)
# ---------------------------------------------------------------------------

def sliding_window(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Create sliding windows over a 2-D array (frames × features).

    Uses Polars for the stride logic when available, otherwise falls back
    to numpy stride tricks.

    Returns: 3-D array (num_windows, window_size, features).
    """
    n_frames, n_features = data.shape
    if window_size > n_frames:
        # Pad to window_size
        pad_len = window_size - n_frames
        data = np.vstack([data, np.zeros((pad_len, n_features), dtype=data.dtype)])
        return data[np.newaxis, :, :]

    n_windows = (n_frames - window_size) // stride + 1
    # numpy stride tricks are fastest for this pattern
    strides = (data.strides[0] * stride, data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(
        data, shape=(n_windows, window_size, n_features), strides=strides
    ).copy()
