# -*- coding: utf-8 -*-
"""
LOSO post-analysis plotting.

Generates per-fold R²/MAE bar charts, feature heatmaps, domain-gap
visualisations, and scatter plots from a saved ``loso_results.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_loso_plots(results_dir: Path, plots_dir: Path, selected_dataset: str) -> None:
    """Read *results_dir/loso_results.csv* and write plots to *plots_dir*."""

    csv_path = results_dir / "loso_results.csv"
    if not csv_path.exists():
        print("  No loso_results.csv found — skipping plots.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping LOSO plots.")
        return

    loso_df = pd.read_csv(csv_path)

    primary_tag = "ph" if selected_dataset == "PATIENT_HANDLING" else "gl"
    xlab_tag    = "gl" if selected_dataset == "PATIENT_HANDLING" else "ph"
    primary_label = primary_tag.upper()
    xlab_label    = xlab_tag.upper()

    # ── 1. Per-fold R² bar chart ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(loso_df))
    w = 0.35
    if f"{primary_tag}_r2" in loso_df.columns:
        ax.bar(x - w / 2, loso_df[f"{primary_tag}_r2"], w,
               label=f"{primary_label} (held-out)", color="steelblue")
    if f"{xlab_tag}_r2" in loso_df.columns:
        ax.bar(x + w / 2, loso_df[f"{xlab_tag}_r2"], w,
               label=f"{xlab_label} (cross-lab)", color="coral")
    ax.set(xlabel="LOSO Fold", ylabel="Mean R²",
           title="LOSO Cross-Validation: Per-Fold Mean R²")
    ax.set_xticks(x)
    ax.set_xticklabels(loso_df["subject"], rotation=45, ha="right")
    ax.legend(); ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(plots_dir / "loso_per_fold_r2.png", dpi=150); plt.close()
    print(f"  Plot: {plots_dir / 'loso_per_fold_r2.png'}")

    # ── Helper: extract per-feature DataFrames ──────────────────────────
    prim_feat_cols = [c for c in loso_df.columns if c.startswith(f"{primary_tag}_r2_")]
    xlab_feat_cols = [c for c in loso_df.columns if c.startswith(f"{xlab_tag}_r2_")]

    def _make_feat_df(cols, tag):
        names = [c.replace(f"{tag}_r2_", "") for c in cols]
        df = loso_df[cols].copy()
        df.columns = names
        df.index = loso_df["subject"]
        return df, names

    prim_feat_df, prim_feat_names = _make_feat_df(prim_feat_cols, primary_tag) if prim_feat_cols else (None, [])
    xlab_feat_df, xlab_feat_names = _make_feat_df(xlab_feat_cols, xlab_tag) if xlab_feat_cols else (None, [])

    # ── 2. Per-feature R² heatmap (primary) ─────────────────────────────
    if prim_feat_df is not None:
        _heatmap(prim_feat_df, f"{primary_label} Per-Feature R² by LOSO Fold",
                 plots_dir / f"loso_{primary_tag}_feature_heatmap.png")

    # ── 3. Per-feature R² heatmap (cross-lab) ──────────────────────────
    if xlab_feat_df is not None:
        _heatmap(xlab_feat_df, f"{xlab_label} Cross-Lab Per-Feature R² by LOSO Fold",
                 plots_dir / f"loso_{xlab_tag}_feature_heatmap.png")

    # ── 4. Mean per-feature R² comparison bar chart ─────────────────────
    if prim_feat_cols:
        _feature_comparison_bar(
            loso_df, prim_feat_cols, xlab_feat_cols,
            primary_tag, xlab_tag, primary_label, xlab_label,
            plots_dir / "loso_mean_feature_r2.png",
        )

    # ── 5. Per-fold MAE bar chart ───────────────────────────────────────
    if f"{primary_tag}_mae" in loso_df.columns:
        _mae_bar(loso_df, primary_tag, xlab_tag, primary_label, xlab_label,
                 plots_dir / "loso_per_fold_mae.png")

    # ── 6. Domain gap heatmap ───────────────────────────────────────────
    if prim_feat_df is not None and xlab_feat_df is not None:
        _domain_gap_heatmap(prim_feat_df, xlab_feat_df,
                            primary_label, xlab_label,
                            plots_dir / "loso_domain_gap_heatmap.png")

    # ── 7. Per-feature R² with error bars ───────────────────────────────
    if prim_feat_cols and xlab_feat_cols:
        _feature_errorbar(loso_df, prim_feat_cols, xlab_feat_cols,
                          primary_tag, xlab_tag, primary_label, xlab_label,
                          plots_dir / "loso_feature_r2_errorbar.png")

    # ── 8. Scatter: Primary vs Cross-lab R² per-feature ─────────────────
    if prim_feat_cols and xlab_feat_cols:
        _scatter_primary_vs_xlab(loso_df, prim_feat_cols, xlab_feat_cols,
                                 primary_tag, xlab_tag, primary_label, xlab_label,
                                 plots_dir / "loso_scatter_primary_vs_xlab.png")

    print("\n--- LOSO post-analysis complete ---")


# ── Plot primitives ─────────────────────────────────────────────────────────

def _heatmap(feat_df: pd.DataFrame, title: str, save_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(feat_df.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(feat_df.shape[1]))
    ax.set_xticklabels(feat_df.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(feat_df)))
    ax.set_yticklabels(feat_df.index)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="R²")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Plot: {save_path}")


def _feature_comparison_bar(
    df, prim_cols, xlab_cols, prim_tag, xlab_tag, prim_label, xlab_label, save_path,
) -> None:
    import matplotlib.pyplot as plt
    prim_means = df[prim_cols].mean()
    prim_means.index = [c.replace(f"{prim_tag}_r2_", "") for c in prim_cols]
    x = np.arange(len(prim_means))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, prim_means.values, w, label=prim_label, color="steelblue")
    if xlab_cols:
        xlab_means = df[xlab_cols].mean()
        xlab_means.index = [c.replace(f"{xlab_tag}_r2_", "") for c in xlab_cols]
        ax.bar(x + w / 2, xlab_means.values, w, label=xlab_label, color="coral")
    ax.set(xlabel="Target Feature", ylabel="Mean R² (across folds)",
           title=f"LOSO Mean Per-Feature R²: {prim_label} vs {xlab_label}")
    ax.set_xticks(x); ax.set_xticklabels(prim_means.index, rotation=60, ha="right", fontsize=8)
    ax.legend(); ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Plot: {save_path}")


def _mae_bar(df, prim_tag, xlab_tag, prim_label, xlab_label, save_path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df[f"{prim_tag}_mae"], w,
           label=f"{prim_label} (held-out)", color="steelblue")
    if f"{xlab_tag}_mae" in df.columns:
        ax.bar(x + w / 2, df[f"{xlab_tag}_mae"], w,
               label=f"{xlab_label} (cross-lab)", color="coral")
    ax.set(xlabel="LOSO Fold", ylabel="Mean MAE",
           title="LOSO Cross-Validation: Per-Fold Mean MAE")
    ax.set_xticks(x); ax.set_xticklabels(df["subject"], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Plot: {save_path}")


def _domain_gap_heatmap(prim_df, xlab_df, prim_label, xlab_label, save_path) -> None:
    import matplotlib.pyplot as plt
    gap = prim_df.values - xlab_df.values
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(gap, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(prim_df.shape[1]))
    ax.set_xticklabels(prim_df.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(prim_df)))
    ax.set_yticklabels(prim_df.index)
    ax.set_title(f"Domain Gap ({prim_label} R² - {xlab_label} R²)\n"
                 f"(Blue={xlab_label} better, Red={prim_label} better)")
    plt.colorbar(im, ax=ax, label="R² Difference")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Plot: {save_path}")


def _feature_errorbar(df, prim_cols, xlab_cols, prim_tag, xlab_tag,
                       prim_label, xlab_label, save_path) -> None:
    import matplotlib.pyplot as plt
    feat_names = [c.replace(f"{prim_tag}_r2_", "") for c in prim_cols]
    x = np.arange(len(feat_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w / 2, df[prim_cols].mean().values, w, label=prim_label, color="steelblue",
           yerr=df[prim_cols].std().values, capsize=5, error_kw={"linewidth": 1})
    ax.bar(x + w / 2, df[xlab_cols].mean().values, w, label=xlab_label, color="coral",
           yerr=df[xlab_cols].std().values, capsize=5, error_kw={"linewidth": 1})
    ax.set(xlabel="Target Feature", ylabel="Mean R² (± std)",
           title=f"Per-Feature R² with Fold Variability ({prim_label} vs {xlab_label})")
    ax.set_xticks(x); ax.set_xticklabels(feat_names, rotation=60, ha="right", fontsize=8)
    ax.legend(); ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Plot: {save_path}")


def _scatter_primary_vs_xlab(df, prim_cols, xlab_cols, prim_tag, xlab_tag,
                              prim_label, xlab_label, save_path) -> None:
    import matplotlib.pyplot as plt
    feat_names = [c.replace(f"{prim_tag}_r2_", "") for c in prim_cols]
    colors = plt.cm.tab10(np.linspace(0, 1, len(feat_names)))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, feat in enumerate(feat_names):
        pc = f"{prim_tag}_r2_{feat}"
        xc = f"{xlab_tag}_r2_{feat}"
        if pc in df.columns and xc in df.columns:
            ax.scatter(df[pc], df[xc], label=feat, s=100, alpha=0.7, color=colors[i])
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=2, label=f"{prim_label} = {xlab_label}")
    ax.set(xlabel=f"{prim_label} (held-out) R²", ylabel=f"{xlab_label} (cross-lab) R²",
           title=f"Cross-Lab Transfer: {prim_label} vs {xlab_label}\n(Below diagonal = poor transfer)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot: {save_path}")
