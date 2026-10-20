# -*- coding: utf-8 -*-
"""Results figures for the JBHI submission, drawn from run outputs (no AI-generated art).
Writes figures/fig_inflation.pdf, fig_frame.pdf, fig_xlab.pdf, fig_models.pdf and fig_subjects.pdf (when multi-seed exists).
Style: one serif-free family, 8-pt labels, single-column width 3.5 in, colorblind-safe palette."""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__import__("os").getenv("JBHI_CODE_ROOT", str(Path(__file__).resolve().parents[1] / "results" / "paper_v2_stage")))   # v2 run family (code/chapter1/experiments/stage_v2_for_paper.py)
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
C = {"icf": "#0072B2", "cnnlstm": "#D55E00", "cnn": "#009E73", "grey": "#7f7f7f", "sub": "#999999", "cor": "#0072B2"}
CH = [("vert", "Vert."), ("ml", "ML"), ("ap", "AP"), ("px", "CoP$_x$"), ("pz", "CoP$_z$")]

def chan_means(df, prefix):
    return {c: float(np.nanmean(df[[k for k in df.columns if k.startswith(prefix) and k.endswith("_" + c) and "mae" not in k]].values)) for c, _ in CH}
def chan_per_plate(df, prefix):
    out = {}
    for c, _ in CH:
        for p in (1, 2):
            k = f"{prefix}ground_force_{p}_{c}"
            if k in df.columns: out[f"F{p} {c}"] = float(df[k].mean())
    return out

# ---------------------------------------------------------------- Fig 1: inflation
def fig_inflation():
    """Per-subject LOSO R2 (mean of three seeds, as in Table I) with its mean as a bar, and the trial-split score of
    the same configuration (mean of three seeds) as a star; the dashed connector is labeled with the gap."""
    cfgs = [("icf", "ICF-hpo", "ICF"), ("cnnlstm", "CNN-LSTM-hpo", "CNN-LSTM"), ("cnn", "CNN-hpo", "CNN")]
    dash = (0, (2, 1.5))
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.15), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, tag in zip(axes, ["gl", "ph"]):
        ms = pd.read_csv(CODE / f"outputs/e1_multiseed_corrected/{tag}_multiseed.csv")
        ts = pd.read_csv(CODE / f"outputs/trial_split_corrected/{tag}_trial_split.csv").set_index("config")["r2"].to_dict()
        for i, (m, cfg, lab) in enumerate(cfgs):
            v = ms[ms["config"] == cfg].groupby("subject")["r2"].mean().values; mu = v.mean()
            x = i - 0.12 + np.random.default_rng(0).uniform(-0.08, 0.08, len(v))
            ax.scatter(x, v, s=10, color=C[m], alpha=0.55, lw=0)
            ax.hlines(mu, i - 0.28, i + 0.04, color=C[m], lw=1.8)
            t = ts[cfg]
            ax.scatter([i + 0.2], [t], marker="*", s=70, color=C[m], edgecolor="k", lw=0.4, zorder=5)
            ax.hlines(mu, i + 0.04, i + 0.2, color="k", lw=0.6, ls=dash); ax.vlines(i + 0.2, mu, t, color="k", lw=0.6, ls=dash)
            ax.text(i + 0.2, t + 0.035, f"+{t - mu:.2f}", fontsize=6.3, va="bottom", ha="center")
        ax.set_xticks(range(3)); ax.set_xticklabels([c[2] for c in cfgs]); ax.set_xlim(-0.5, 2.5)
        ax.set_title(f"{tag.upper()} ({'6' if tag=='gl' else '10'} subjects)"); ax.set_ylim(0.1, 1.0)
    axes[0].set_ylabel("$R^2$")
    h = [plt.Line2D([], [], marker="o", ls="", color="#555", alpha=0.6, ms=4, label="LOSO, one subject"),
         plt.Line2D([], [], color="#555", lw=1.8, label="LOSO mean"),
         plt.Line2D([], [], marker="*", ls="", mfc="#ccc", mec="k", mew=0.4, ms=8, label="trial split")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1), handletextpad=0.4, columnspacing=1.6)
    fig.savefig(FIG / "fig_inflation.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: frame check and its cost (paired dumbbells)
def fig_frame():
    """Same rows in both panels, plate-frame -> co-rotated targets (ICF fixed config, per-trial standardization):
    (a) mean per-trial Newton's-law correlation from newton_check.csv, computed before training;
    (b) LOSO R2 per force channel (mean of the two plates) from the ablation baseline and corot arms, one sub-row per
    model: ICF at the fixed config (outputs/ablation/), CNN-LSTM and CNN at their tuned configs (outputs/ablation_<model>/)."""
    newton = pd.read_csv(HERE / "newton_check.csv")
    grey, blue, red = "#8a8a8a", C["icf"], "#B03030"
    axes_ = [("ap", "forward", "AP"), ("ml", "lateral", "ML"), ("vert", "vertical", "Vertical")]
    MODELS = [("", "ICF"), ("_cnn-lstm", "CNN-LSTM"), ("_cnn", "CNN")]
    MODELS = [mm for mm in MODELS if (CODE / f"outputs/ablation{mm[0]}").exists()]
    def r2(tag, arm, ch, sfx=""):
        df = pd.read_csv(CODE / f"outputs/ablation{sfx}/{tag}_{arm}.csv")
        return float(df[[f"r2_ground_force_{p}_{ch}" for p in (1, 2)]].mean(axis=1).mean())
    def nc(tag, rot, axis):
        return float(newton[(newton.dataset == tag) & (newton.rotate == rot)][f"r_{axis}"].mean())
    def dumbbell(ax, y, a, b, s=20, lw=1.1):
        col = red if b < a - 0.02 else blue
        if abs(b - a) > 0.02:
            ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=6 if s < 20 else 7, shrinkA=2.2, shrinkB=2.2))
        ax.scatter([a], [y], s=s, color="white", edgecolor=grey, lw=lw, zorder=3)
        ax.scatter([b], [y], s=s * 1.1, color=col, lw=0, zorder=4)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.15), sharey=True, gridspec_kw=dict(wspace=0.12))
    rows, y = [], 0
    for tag in ["gl", "ph"]:
        for ch, axn, name in axes_:
            rows.append((tag, ch, axn, name, y)); y -= 1
        y -= 0.6
    for tag, ch, axn, name, yy in rows:
        dumbbell(a1, yy, nc(tag, 0, axn), nc(tag, 1, axn))
        off = (len(MODELS) - 1) / 2
        for k, (sfx, lab) in enumerate(MODELS):
            y2 = yy + (off - k) * 0.27
            dumbbell(a2, y2, r2(tag, "baseline", ch, sfx), r2(tag, "corot", ch, sfx), s=11, lw=0.8)
            if (tag, ch) == ("gl", "ap"):
                a2.text(1.02, y2, lab, transform=a2.get_yaxis_transform(), fontsize=5.6, va="center", ha="left", color="#444")
    a1.set_yticks([r[4] for r in rows]); a1.set_yticklabels([r[3] for r in rows]); a1.tick_params(axis="y", length=0)
    for tag, y0 in (("GL", rows[0][4]), ("PH", rows[3][4])):
        a1.text(-0.62, y0 + 0.62, tag, transform=a1.get_yaxis_transform(), fontsize=8, fontweight="bold", ha="left", va="center")
    a1.axvline(0, color="k", lw=0.6); a1.set_xlim(-0.2, 0.6); a1.set_xticks([-0.2, 0, 0.2, 0.4]); a1.set_xticklabels(["-0.2", "0", "0.2", "0.4"])
    a1.set_xlabel("corr(force, COM acceleration)"); a1.set_title("(a) Newton's-law check", loc="left", fontsize=8)
    a2.set_xlim(0, 1); a2.set_xticks([0, 0.5, 1]); a2.set_xticklabels(["0", "0.5", "1"])
    a2.set_xlabel("LOSO $R^2$"); a2.set_title("(b) Model accuracy", loc="left", fontsize=8)
    for a in (a1, a2):
        a.grid(axis="x", color="#ececec", lw=0.6); a.set_axisbelow(True)
        a.axhspan(rows[0][4] - 0.45, rows[0][4] + 0.45, color="#fbe9df", zorder=0, lw=0)
    a1.set_ylim(rows[-1][4] - 0.6, rows[0][4] + 1.0)
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=grey, mew=1.1, ms=5, label="plate-frame targets"),
         plt.Line2D([], [], marker="o", ls="", color=blue, ms=5, label="co-rotated targets")]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.1), columnspacing=1.5)
    fig.savefig(FIG / "fig_frame.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: cross-lab per channel
def fig_xlab():
    gl2ph = pd.read_csv(CODE / "runs/m6_icf_gl/results/loso_results.csv"); ph2gl = pd.read_csv(CODE / "runs/m6_phxlab_icf/results/loso_results.csv")
    a = chan_per_plate(gl2ph, "ph_r2_"); b = chan_per_plate(ph2gl, "gl_r2_")
    keys = [f"F{p} {ch}" for ch, _ in CH for p in (1, 2)]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    x = np.arange(len(keys)); w = 0.38
    va = np.clip([a[k] for k in keys], -3, None); vb = np.clip([b[k] for k in keys], -3, None)
    ax.bar(x - w / 2, va, w, color=C["icf"], label="train GL, test PH")
    ax.bar(x + w / 2, vb, w, color=C["cnnlstm"], label="train PH, test GL")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-3.1, 0.3)
    ax.set_xticks(x); ax.set_xticklabels([k[1] for k in keys], fontsize=6.5); ax.set_xlabel("plate", fontsize=7, labelpad=1)
    for i, (ch, name) in enumerate(CH): ax.text(2 * i + 0.5, 0.12, name, ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("cross-lab $R^2$ (clipped at $-3$)")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.24))
    fig.savefig(FIG / "fig_xlab.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: model comparison with seeds (optional)
def fig_models():
    files = sorted(glob.glob(str(CODE / "outputs/e1_multiseed_corrected/*_multiseed.csv")))
    if not files: return
    lab = {"CNN-LSTM-hpo": ("CNN-LSTM", C["cnnlstm"]), "ICF-hpo": ("ICF", C["icf"]), "ICF-grid128": ("ICF (grid)", "#56B4E9"), "CNN-hpo": ("CNN", C["cnn"])}
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, f in zip(axes, files):
        df = pd.read_csv(f); tag = df["dataset"].iloc[0]
        order = [c for c in ["CNN-LSTM-hpo", "ICF-hpo", "ICF-grid128", "CNN-hpo"] if c in set(df["config"])]
        for i, cfg in enumerate(order):
            sub = df[df["config"] == cfg]
            per_seed = sub.groupby("seed")["r2"].mean()
            per_subj = sub.groupby("subject")["r2"].mean()
            x = np.full(len(per_subj), i) + np.random.default_rng(1).uniform(-0.12, 0.12, len(per_subj))
            ax.scatter(x, per_subj.values, s=10, color=lab[cfg][1], alpha=0.7, lw=0)
            ax.errorbar(i, per_seed.mean(), yerr=per_seed.std() if len(per_seed) > 1 else 0, fmt="_", color="k", ms=12, mew=1.6, capsize=3, lw=1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[c][0] for c in order], rotation=25, ha="right")
        ax.set_title(tag.upper()); ax.set_ylim(0.1, 0.95)
    axes[0].set_ylabel("LOSO $R^2$")
    fig.savefig(FIG / "fig_models.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- per-subject spread with the tuning proxy marked
PROXY = {"gl": "gl_5", "ph": "ph_12"}
def fig_subjects():
    files = sorted(glob.glob(str(CODE / "outputs/e1_multiseed_corrected/*_multiseed.csv")))
    if not files: return
    lab = {"CNN-LSTM-hpo": ("CNN-LSTM", C["cnnlstm"], "o"), "ICF-hpo": ("ICF", C["icf"], "s"),
           "ICF-grid128": ("ICF (grid)", "#56B4E9", "D"), "CNN-hpo": ("CNN", C["cnn"], "^")}
    dfs = [pd.read_csv(f) for f in files]
    widths = [d["subject"].nunique() for d in dfs]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True, gridspec_kw=dict(wspace=0.06, width_ratios=widths))
    for ax, df in zip(axes, dfs):
        tag = df["dataset"].iloc[0]
        per = df.groupby(["config", "subject"])["r2"].mean().unstack(0)
        per = per[[c for c in lab if c in per.columns]]
        order = per.mean(axis=1).sort_values().index.tolist()          # hardest -> easiest
        x = np.arange(len(order))
        if PROXY[tag] in order:
            k = order.index(PROXY[tag]); ax.axvspan(k - 0.45, k + 0.45, color="#e6e6e6", zorder=0, lw=0)
            ax.text(k, 0.03, "proxy", ha="center", va="bottom", fontsize=6, color="#555555", rotation=90)
        for cfg in per.columns:
            name, col, mk = lab[cfg]
            ax.plot(x, per.loc[order, cfg].values, ls="none", marker=mk, ms=3.2, color=col, alpha=0.9, label=name, mew=0)
        ax.plot(x, per.loc[order].mean(axis=1).values, color="k", lw=0.8, alpha=0.6, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels([s.split("_")[1] for s in order], fontsize=6.5)
        ax.set_xlim(-0.6, len(order) - 0.4); ax.set_title(tag.upper(), pad=2)
        ax.set_xlabel("held-out subject", labelpad=1)
    axes[0].set_ylabel("LOSO $R^2$"); axes[0].set_ylim(0.0, 0.9)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 1.10), handletextpad=0.2, columnspacing=0.8)
    fig.savefig(FIG / "fig_subjects.pdf", bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig_inflation(); fig_frame(); fig_xlab(); fig_models(); fig_subjects()
    print("figures:", sorted(os.path.basename(p) for p in glob.glob(str(FIG / "*.pdf"))))
