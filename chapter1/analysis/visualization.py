# -*- coding: utf-8 -*-
"""
Plotly-based visualization functions for the GRF prediction pipeline.

Replaces matplotlib/seaborn plots with interactive Plotly figures.
All functions save HTML files (interactive) and optionally static PNG.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_fig(fig: "go.Figure", plot_dir: str, filename: str, write_png: bool = True) -> str:
    """Save a plotly figure as HTML (always) and PNG (when kaleido available)."""
    plot_dir = _ensure_dir(plot_dir)
    base = Path(filename).stem
    html_path = os.path.join(plot_dir, f"{base}.html")
    fig.write_html(html_path)
    png_path = os.path.join(plot_dir, f"{base}.png")
    if write_png:
        try:
            fig.write_image(png_path, scale=2)
        except Exception:
            pass  # kaleido not installed — skip static export
    return html_path


# ---------------------------------------------------------------------------
# Training curve plots (replaces plot_results in train.py)
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses, val_losses,
    train_maes, val_maes,
    train_r2s, val_r2s,
    save_dir: str | None = None,
    show: bool = False,
) -> "go.Figure":
    """Plot loss, MAE, and R² training curves using Plotly subplots."""
    epochs = list(range(1, len(train_losses) + 1))
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Loss", "MAE", "R²"),
        vertical_spacing=0.08,
    )

    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, name="Train Loss", line=dict(color="#1f77b4")), row=1, col=1)
    if val_losses and not all(np.isnan(v) for v in val_losses):
        fig.add_trace(go.Scatter(x=epochs, y=val_losses, name="Val Loss", line=dict(color="#ff7f0e")), row=1, col=1)

    # MAE
    fig.add_trace(go.Scatter(x=epochs, y=train_maes, name="Train MAE", line=dict(color="#1f77b4")), row=2, col=1)
    if val_maes and not all(np.isnan(v) for v in val_maes):
        fig.add_trace(go.Scatter(x=epochs, y=val_maes, name="Val MAE", line=dict(color="#ff7f0e")), row=2, col=1)

    # R²
    fig.add_trace(go.Scatter(x=epochs, y=train_r2s, name="Train R²", line=dict(color="#1f77b4")), row=3, col=1)
    if val_r2s and not all(np.isnan(v) for v in val_r2s):
        fig.add_trace(go.Scatter(x=epochs, y=val_r2s, name="Val R²", line=dict(color="#ff7f0e")), row=3, col=1)

    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    fig.update_layout(height=900, template="plotly_white", showlegend=True, legend=dict(x=1.02, y=1))

    if save_dir:
        _save_fig(fig, save_dir, "training_curves")
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Optuna HPO plots (replaces matplotlib in main.py)
# ---------------------------------------------------------------------------

def plot_optuna_history(
    trials_df: pd.DataFrame,
    best_value: float,
    model_name: str,
    save_dir: str | None = None,
) -> "go.Figure":
    """Plot Optuna optimisation history."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trials_df.index, y=trials_df["value"],
        mode="lines+markers", name="Trial Loss",
        marker=dict(size=6),
    ))
    fig.add_hline(y=best_value, line_dash="dash", line_color="red",
                  annotation_text=f"Best: {best_value:.4f}")
    fig.update_layout(
        title=f"{model_name} — Optimization History",
        xaxis_title="Trial",
        yaxis_title="Validation Loss",
        template="plotly_white",
    )
    if save_dir:
        _save_fig(fig, save_dir, f"{model_name}_optimization_history")
    return fig


def plot_optuna_trial_scatter(
    trials_df: pd.DataFrame,
    best_value: float,
    model_name: str,
    save_dir: str | None = None,
) -> "go.Figure":
    """Scatter of trial values, best highlighted in green."""
    colours = ["green" if v == best_value else "steelblue" for v in trials_df["value"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trials_df.index, y=trials_df["value"],
        mode="markers", marker=dict(color=colours, size=8, opacity=0.7),
        name="Trials",
    ))
    fig.add_hline(y=best_value, line_dash="dash", line_color="red",
                  annotation_text=f"Best: {best_value:.4f}")
    fig.update_layout(
        title=f"{model_name} — Trial Values",
        xaxis_title="Trial", yaxis_title="Validation Loss",
        template="plotly_white",
    )
    if save_dir:
        _save_fig(fig, save_dir, f"{model_name}_trial_values")
    return fig


def plot_optuna_param_scatter(
    trials_df: pd.DataFrame,
    model_name: str,
    save_dir: str | None = None,
) -> "go.Figure | None":
    """Scatter of each numeric parameter vs loss."""
    param_cols = [c for c in trials_df.columns if c.startswith("params_")]
    numeric_pcs = [
        c for c in param_cols
        if pd.to_numeric(trials_df[c], errors="coerce").notna().sum() > 0
    ]
    if not numeric_pcs:
        return None

    n = len(numeric_pcs)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    subplot_titles = [c.replace("params_", "") for c in numeric_pcs]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    for idx, col in enumerate(numeric_pcs):
        r = idx // cols + 1
        c = idx % cols + 1
        x = pd.to_numeric(trials_df[col], errors="coerce")
        y = trials_df["value"]
        valid = x.notna() & y.notna()
        fig.add_trace(
            go.Scatter(x=x[valid], y=y[valid], mode="markers",
                       marker=dict(size=5, opacity=0.6), showlegend=False),
            row=r, col=c,
        )
        fig.update_xaxes(title_text=col.replace("params_", ""), row=r, col=c)
        fig.update_yaxes(title_text="Loss", row=r, col=c)

    fig.update_layout(
        height=350 * rows,
        title=f"{model_name} — Parameter vs Loss",
        template="plotly_white",
    )
    if save_dir:
        _save_fig(fig, save_dir, f"{model_name}_param_vs_loss_scatter")
    return fig


# ---------------------------------------------------------------------------
# Model comparison bar chart (replaces seaborn in main.py & post_analysis.py)
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results_df: pd.DataFrame,
    save_dir: str | None = None,
) -> "go.Figure":
    """Bar chart comparing test MAE and R² across models."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Test MAE", "Test R²", "Training Time (s)"))

    fig.add_trace(go.Bar(x=results_df["model"], y=results_df["test_mae"], marker_color="#1f77b4", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=results_df["model"], y=results_df["test_r2"], marker_color="#2ca02c", showlegend=False), row=1, col=2)
    if "training_time_sec" in results_df.columns:
        fig.add_trace(go.Bar(x=results_df["model"], y=results_df["training_time_sec"], marker_color="#d62728", showlegend=False), row=1, col=3)

    fig.add_hline(y=0.75, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_layout(height=450, template="plotly_white", title="Model Comparison")
    if save_dir:
        _save_fig(fig, save_dir, "model_comparison_plots")
    return fig


# ---------------------------------------------------------------------------
# Prediction quality (replaces matplotlib in post_analysis.py)
# ---------------------------------------------------------------------------

def plot_prediction_quality(
    predictions: np.ndarray,
    actuals: np.ndarray,
    target_feature_names: list[str],
    model_name: str,
    sample_idx: int = 0,
    masks: np.ndarray | None = None,
    save_dir: str | None = None,
) -> "go.Figure":
    """Line plots of predictions vs actuals for each target feature."""
    n_feat = len(target_feature_names)
    valid_len = int(np.sum(masks[sample_idx])) if masks is not None else actuals.shape[1]

    fig = make_subplots(rows=n_feat, cols=1, shared_xaxes=True,
                        subplot_titles=[f"{model_name} — {name}" for name in target_feature_names])

    for i in range(n_feat):
        t = np.arange(valid_len)
        fig.add_trace(go.Scatter(x=t, y=actuals[sample_idx, :valid_len, i],
                                 name=f"Actual ({target_feature_names[i]})", line=dict(color="#1f77b4"),
                                 showlegend=(i == 0)), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=t, y=predictions[sample_idx, :valid_len, i],
                                 name=f"Predicted ({target_feature_names[i]})",
                                 line=dict(color="#d62728", dash="dash"),
                                 showlegend=(i == 0)), row=i + 1, col=1)
        fig.update_yaxes(title_text=target_feature_names[i], row=i + 1, col=1)

    fig.update_xaxes(title_text="Timestep", row=n_feat, col=1)
    fig.update_layout(
        height=250 * n_feat,
        template="plotly_white",
        title=f"Prediction vs Actual — Sample {sample_idx}",
    )
    if save_dir:
        _save_fig(fig, save_dir, f"prediction_quality_{model_name}_sample_{sample_idx}")
    return fig


# ---------------------------------------------------------------------------
# Performance comparison bar (post_analysis.py)
# ---------------------------------------------------------------------------

def plot_performance_comparison(
    results_df: pd.DataFrame,
    save_dir: str | None = None,
) -> "go.Figure":
    """Side-by-side bar of test MAE and R²."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Test MAE", "Test R²"))
    fig.add_trace(go.Bar(x=results_df["model"], y=results_df["test_mae"],
                         marker_color="#1f77b4", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=results_df["model"], y=results_df["test_r2"],
                         marker_color="#2ca02c", showlegend=False), row=1, col=2)
    fig.add_hline(y=0.75, line_dash="dash", line_color="red", row=1, col=2,
                  annotation_text="Target R²=0.75")
    fig.update_layout(height=450, template="plotly_white")
    if save_dir:
        _save_fig(fig, save_dir, "performance_comparison")
    return fig


# ---------------------------------------------------------------------------
# Attention heatmap (replaces sns.heatmap in post_analysis.py)
# ---------------------------------------------------------------------------

def plot_attention_heatmap_plotly(
    attention_weights: np.ndarray,
    num_layers: int,
    num_heads: int,
    model_name: str = "",
    sample_idx: int = 0,
    save_dir: str | None = None,
    max_heatmap_points: int = 400,
    downsample_fn=None,
) -> "go.Figure":
    """Grid of attention heatmaps (layers × heads) using Plotly."""
    fig = make_subplots(
        rows=num_layers, cols=num_heads,
        subplot_titles=[f"L{i+1} H{j+1}" for i in range(num_layers) for j in range(num_heads)],
    )
    for i in range(num_layers):
        for j in range(num_heads):
            attn_2d = attention_weights[i, j]
            if downsample_fn:
                attn_2d = downsample_fn(attn_2d, max_points=max_heatmap_points)
            fig.add_trace(go.Heatmap(
                z=attn_2d, colorscale="Viridis", showscale=(i == 0 and j == num_heads - 1),
                zmin=0, zmax=float(np.quantile(attn_2d, 0.995)) if attn_2d.size else None,
            ), row=i + 1, col=j + 1)

    fig.update_layout(
        height=350 * num_layers,
        width=350 * num_heads,
        title=f"Attention Heatmaps — {model_name} Sample {sample_idx}",
        template="plotly_white",
    )
    if save_dir:
        _save_fig(fig, save_dir, f"attention_heatmap_{model_name}_sample_{sample_idx}")
    return fig


# ---------------------------------------------------------------------------
# Overlaid heatmap + signal
# ---------------------------------------------------------------------------

def plot_overlaid_heatmap_and_signal(
    attn_map: np.ndarray,
    vy_signal: np.ndarray,
    model_name: str,
    sample_idx: int = 0,
    layer_idx: int = 0,
    head_idx: int = 0,
    save_dir: str | None = None,
) -> "go.Figure":
    """Composite: vertical GRF signal on top, attention heatmap below."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.35, 0.65],
        vertical_spacing=0.08,
        subplot_titles=(
            f"Vertical GRF — Sample {sample_idx}",
            f"Attention (L{layer_idx+1} H{head_idx+1})",
        ),
    )
    fig.add_trace(go.Scatter(
        y=vy_signal, mode="lines", line=dict(color="#1f77b4", width=2), name="vert GRF",
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=attn_map, colorscale="Viridis",
        zmin=0, zmax=float(np.quantile(attn_map, 0.995)) if attn_map.size else None,
        showscale=True, name="Attention",
    ), row=2, col=1)
    fig.update_layout(height=700, template="plotly_white",
                      title=f"Overlaid Attention — {model_name}")
    if save_dir:
        _save_fig(fig, save_dir, f"overlaid_attention_{model_name}_sample_{sample_idx}")
    return fig


# ---------------------------------------------------------------------------
# Multi-head grid & attention evolution
# ---------------------------------------------------------------------------

def plot_multi_head_grid(
    layer_attns: np.ndarray,
    num_heads: int,
    layer_to_show: int,
    model_name: str = "",
    sample_idx: int = 0,
    save_dir: str | None = None,
    max_heatmap_points: int = 350,
    downsample_fn=None,
) -> "go.Figure":
    """Grid of all heads from a single layer."""
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Head {h+1}" for h in range(num_heads)])
    for h in range(num_heads):
        r = h // cols + 1
        c = h % cols + 1
        data = layer_attns[h]
        if downsample_fn:
            data = downsample_fn(data, max_points=max_heatmap_points)
        fig.add_trace(go.Heatmap(
            z=data, colorscale="Viridis", showscale=False,
            zmin=0, zmax=float(np.quantile(data, 0.995)) if data.size else None,
        ), row=r, col=c)

    fig.update_layout(
        height=350 * rows, width=350 * cols,
        title=f"Multi-Head Grid — Layer {layer_to_show+1} ({model_name})",
        template="plotly_white",
    )
    if save_dir:
        _save_fig(fig, save_dir, f"multi_head_grid_{model_name}_L{layer_to_show+1}_sample_{sample_idx}")
    return fig


def plot_attention_evolution(
    attention_weights: np.ndarray,
    num_layers: int,
    head_to_show: int,
    valid_len: int,
    model_name: str = "",
    sample_idx: int = 0,
    save_dir: str | None = None,
    max_heatmap_points: int = 400,
    downsample_fn=None,
) -> "go.Figure":
    """Show how a single head's attention evolves across layers."""
    fig = make_subplots(rows=1, cols=num_layers,
                        subplot_titles=[f"Layer {i+1}" for i in range(num_layers)])
    for i in range(num_layers):
        data = attention_weights[i, head_to_show, :valid_len, :valid_len]
        if downsample_fn:
            data = downsample_fn(data, max_points=max_heatmap_points)
        fig.add_trace(go.Heatmap(
            z=data, colorscale="Viridis", showscale=(i == num_layers - 1),
            zmin=0, zmax=float(np.quantile(data, 0.995)) if data.size else None,
        ), row=1, col=i + 1)

    fig.update_layout(
        height=500, width=450 * num_layers,
        title=f"Attention Evolution — Head {head_to_show+1} ({model_name})",
        template="plotly_white",
    )
    if save_dir:
        _save_fig(fig, save_dir, f"attention_evolution_{model_name}_head_{head_to_show+1}_sample_{sample_idx}")
    return fig


# ============================================================================
# 3D Gait Visualization  (NEW)
# ============================================================================

# Marker connectivity for stick-figure skeleton
_SKELETON_EDGES = [
    # Left side
    ("LSHO", "LELBL"), ("LELBL", "LWRL"),
    ("LASI", "LKNE"), ("LKNE", "LANKL"), ("LANKL", "LTOE"), ("LANKL", "LHEE"),
    # Right side
    ("RSHO", "RELBL"), ("RELBL", "RWRM"),
    ("RASI", "RKNE"), ("RKNE", "RANKL"), ("RANKL", "RTOE"), ("RANKL", "RHEE"),
    # Trunk
    ("LSHO", "RSHO"),
    ("LASI", "RASI"),
    ("LSHO", "LASI"), ("RSHO", "RASI"),
]

# OpenSim-style short-name aliases used in MARKER_NAME_MAP keys
_RANKL_ALIAS = {"RANKL", "RANKM"}  # Either may appear in the dataset


def plot_gait_3d(
    df: pd.DataFrame,
    frame: int = 0,
    cop_x: np.ndarray | None = None,
    cop_z: np.ndarray | None = None,
    pelvis_origin: pd.DataFrame | None = None,
    marker_name_map: dict[str, str] | None = None,
    title: str = "3D Gait Visualization",
    save_dir: str | None = None,
) -> "go.Figure":
    """
    Interactive 3D scatter of markers + skeleton edges + CoP trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        Marker data with columns like ``"Left Shoulder marker_X"`` etc.
        If multi-frame, one row per frame.
    frame : int
        Which frame (row) to render the skeleton for.
    cop_x, cop_z : optional 1-D arrays
        Centre-of-pressure trajectory (pelvis-relative, mm).
        Plotted as a 3D line at Y=0 plane.
    pelvis_origin : optional DataFrame
        With columns pelvis_X, pelvis_Y, pelvis_Z — used to anchor CoP line
        if absolute coordinates are needed.
    marker_name_map : dict
        Short-name → full-name mapping (from ``src.config.MARKER_NAME_MAP``).
        If None, tries to import from src.config.
    title : str
        Plot title.
    save_dir : str | None
        If provided, saves figure to this directory.
    """
    if marker_name_map is None:
        try:
            from src.config import MARKER_NAME_MAP
            marker_name_map = MARKER_NAME_MAP
        except ImportError:
            marker_name_map = {}

    # Reverse map: full-name → short-name
    rev_map = {v: k for k, v in marker_name_map.items()}

    # Extract marker positions at the requested frame
    marker_xyz: dict[str, tuple[float, float, float]] = {}
    for col in df.columns:
        if not col.endswith("_X"):
            continue
        base = col[:-2]  # e.g. "Left Shoulder marker"
        short = rev_map.get(base, base)
        y_col = f"{base}_Y"
        z_col = f"{base}_Z"
        if y_col in df.columns and z_col in df.columns:
            row = df.iloc[frame] if len(df) > frame else df.iloc[0]
            marker_xyz[short] = (float(row[col]), float(row[y_col]), float(row[z_col]))

    if not marker_xyz:
        print("Warning: No marker data found for 3D plot.")
        return go.Figure()

    fig = go.Figure()

    # --- Markers as 3D scatter ---
    names = list(marker_xyz.keys())
    xs = [marker_xyz[n][0] for n in names]
    ys = [marker_xyz[n][1] for n in names]
    zs = [marker_xyz[n][2] for n in names]

    fig.add_trace(go.Scatter3d(
        x=xs, y=zs, z=ys,  # swap Y/Z for Y-up convention in plotly (Z-up default)
        mode="markers+text",
        marker=dict(size=5, color="#1f77b4"),
        text=names,
        textposition="top center",
        textfont=dict(size=8),
        name="Markers",
    ))

    # --- Skeleton edges ---
    for m1, m2 in _SKELETON_EDGES:
        # Handle RANKL/RANKM alias
        key1 = m1
        key2 = m2
        if m1 not in marker_xyz and m1 in _RANKL_ALIAS:
            alt = (_RANKL_ALIAS - {m1}).pop()
            if alt in marker_xyz:
                key1 = alt
        if m2 not in marker_xyz and m2 in _RANKL_ALIAS:
            alt = (_RANKL_ALIAS - {m2}).pop()
            if alt in marker_xyz:
                key2 = alt

        if key1 in marker_xyz and key2 in marker_xyz:
            p1, p2 = marker_xyz[key1], marker_xyz[key2]
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[2], p2[2]], z=[p1[1], p2[1]],
                mode="lines",
                line=dict(color="gray", width=3),
                showlegend=False,
            ))

    # --- CoP trajectory ---
    if cop_x is not None and cop_z is not None:
        n_cop = len(cop_x)
        cop_y = np.zeros(n_cop)  # ground plane
        fig.add_trace(go.Scatter3d(
            x=cop_x.astype(float), y=cop_z.astype(float), z=cop_y,
            mode="lines+markers",
            marker=dict(size=2, color="red", opacity=0.5),
            line=dict(color="red", width=2),
            name="CoP Trajectory",
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Z (mm)",
            zaxis_title="Y (mm)",
            aspectmode="data",
        ),
        template="plotly_white",
        width=900,
        height=700,
    )
    if save_dir:
        _save_fig(fig, save_dir, "gait_3d")
    return fig
