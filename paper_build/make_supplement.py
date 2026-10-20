# -*- coding: utf-8 -*-
"""Generate the supplementary-material tables (LaTeX) from run outputs.
Writes supp_tables.tex, included by supplement.tex:
  S1  per-subject LOSO R2, corrected pipeline, re-tuned configs (seed 42), all four configs
  S2  per-subject LOSO R2 across seeds (when e1_multiseed_corrected exists)
  S3  per-fold pooled dual-holdout R2 (corrected), three models
  S4  per-subject cross-laboratory R2 (ICF, both directions, per-fold normalization)
  S5  preprocessing / training-choice ablation (when outputs/ablation_prep exists)
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd

CODE = Path(__import__("os").getenv("JBHI_CODE_ROOT", str(Path(__file__).resolve().parents[1] / "results" / "paper_v2_stage")))   # v2 run family (code/chapter1/experiments/stage_v2_for_paper.py)
HERE = Path(__file__).resolve().parent
L = []
def esc(s): return str(s).replace("_", "\\_")
def table(caption, label, header, rows, colspec, size="\\scriptsize"):
    L.append("\\begin{table}[H]\\centering\\caption{%s}\\label{%s}%s" % (caption, label, size))
    L.append("\\begin{tabular}{%s}\\toprule" % colspec)
    L.append(" & ".join("\\textbf{%s}" % h for h in header) + " \\\\ \\midrule")
    for r in rows: L.append(" & ".join(r) + " \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}\n")
f3 = lambda x: "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"

# S1 per-subject, re-tuned
for tag in ["gl", "ph"]:
    df = pd.read_csv(CODE / f"outputs/hpo_confirm/{tag}_confirm.csv")
    piv = df.pivot(index="subject", columns="config", values="r2")
    cfgs = [c for c in ["CNN-LSTM-hpo", "ICF-hpo", "ICF-grid128", "CNN-hpo"] if c in piv.columns]
    rows = [[esc(s)] + [f3(piv.loc[s, c]) for c in cfgs] for s in piv.index]
    rows.append(["\\textbf{mean}"] + [f3(piv[c].mean()) for c in cfgs])
    table(f"Per-subject LOSO \\rsq{{}} on {tag.upper()}, corrected pipeline, re-tuned configurations, seed 42.",
          f"tab:s1_{tag}", ["Subject"] + [c.replace("-hpo", "").replace("grid128", "grid") for c in cfgs], rows, "l" + "c" * len(cfgs))

# S2 multi-seed
for f in sorted(glob.glob(str(CODE / "outputs/e1_multiseed_corrected/*_multiseed.csv"))):
    df = pd.read_csv(f); tag = df["dataset"].iloc[0]
    g = df.groupby(["config", "seed"])["r2"].mean().unstack("seed")
    rows = [[esc(c)] + [f3(g.loc[c, s]) for s in g.columns] + [f3(g.loc[c].mean()), f3(g.loc[c].std())] for c in g.index]
    table(f"Fold-mean LOSO \\rsq{{}} per seed on {tag.upper()}, corrected pipeline, all folds, three seeds.",
          f"tab:s2_{tag}", ["Config"] + [f"seed {s}" for s in g.columns] + ["mean", "SD"], rows, "l" + "c" * (len(g.columns) + 2))

# S3 pooled per fold
rows = []
for m, lab in [("icf", "ICF"), ("cnn-lstm", "CNN-LSTM"), ("cnn", "CNN")]:
    df = pd.read_csv(CODE / f"outputs/m6_combined_{m}/results/combined_dual_holdout_results.csv")
    for _, r in df.iterrows():
        rows.append([lab, str(int(r["fold"])), esc(r["gl_held_out"]), esc(r["ph_held_out"]), f3(r["gl_r2"]), f3(r["ph_r2"])])
table("Pooled dual-holdout CV per fold, corrected pipeline.", "tab:s3", ["Model", "Fold", "GL held out", "PH held out", "GL \\rsq", "PH \\rsq"], rows, "llllcc")

# S4 cross-lab per subject (ICF)
rows = []
a = pd.read_csv(CODE / "runs/m6_icf_gl/results/loso_results.csv")
for _, r in a.iterrows(): rows.append(["GL $\\to$ PH", esc(r["subject"]), f3(r["gl_r2"]), f3(r["ph_r2"])])
b = pd.read_csv(CODE / "runs/m6_phxlab_icf/results/loso_results.csv")
for _, r in b.iterrows(): rows.append(["PH $\\to$ GL", esc(r["subject"]), f3(r["ph_r2"]), f3(r["gl_r2"])])
table("Cross-laboratory transfer per source-laboratory LOSO model (ICF, corrected pipeline, per-fold input standardization). Own = held-out subject of the training laboratory; Other = all trials of the other laboratory.",
      "tab:s4", ["Direction", "Held-out subject", "Own \\rsq", "Other-lab \\rsq"], rows, "llcc")

# S5 preprocessing ablation
fs = sorted(glob.glob(str(CODE / "outputs/ablation_prep/*_ablation_prep.csv")))
if fs:
    rows = []
    order = ["full", "no_pelvis", "no_heading", "no_height", "no_biomech", "no_derived", "no_rel", "no_flip", "no_peak"]
    names = {"full": "full pipeline", "no_pelvis": "no pelvis-relative markers", "no_heading": "no heading alignment (targets in plate frame)",
             "no_height": "no height scaling", "no_biomech": "none of the three above", "no_derived": "no velocity/acceleration features",
             "no_rel": "no relative-position features", "no_flip": "no bilateral flip augmentation", "no_peak": "plain MSE (no peak weight)"}
    dfs = {pd.read_csv(f)["dataset"].iloc[0]: pd.read_csv(f) for f in fs}
    nfold = {"gl": 6, "ph": 10}
    for arm in order:
        row = [names[arm]]
        for tag in ["gl", "ph"]:
            d = dfs.get(tag)
            if d is None or arm not in set(d["arm"]): row += ["--", "--", "--", "--"]; continue
            # fold mean per seed, complete seeds only; SD across seeds; paired delta vs full on the same seeds
            fm = d.groupby(["arm", "seed"]).agg(r2=("r2", "mean"), vert=("vert", "mean"), ml=("ml", "mean"), ap=("ap", "mean"), n=("r2", "size")).reset_index()
            fm = fm[fm.n == nfold[tag]]
            a = fm[fm.arm == arm].set_index("seed"); fu = fm[fm.arm == "full"].set_index("seed")
            common = a.index.intersection(fu.index)
            delta = (a.loc[common, "r2"] - fu.loc[common, "r2"]).mean() if len(common) else float("nan")
            sd = a["r2"].std(ddof=0) if len(a) > 1 else 0.0
            row += [f"{a['r2'].mean():.3f} $\\pm$ {sd:.3f}", ("--" if arm == "full" else f"{delta:+.3f}"), f3(a["ml"].mean()), f3(a["ap"].mean())]
        rows.append(row)
    table("Preprocessing and training-choice ablation: ICF at the fixed configuration ($d_{\\mathrm{model}}=64$, $L=2$), corrected pipeline, full LOSO, one factor removed per row, three training seeds (42, 43, 44). \\rsq{}: mean over seeds of the fold-mean LOSO \\rsq{} $\\pm$ SD across seeds. $\\Delta$: paired difference from the full pipeline on the same seeds. ML and AP: mean horizontal-force \\rsq{} over plates and seeds. Mass and height inputs, per-fold input standardization, body-weight target normalization, centre-of-mass kinematics, and force co-rotation (wherever a heading exists) are kept in every arm.",
          "tab:s5", ["Arm", "GL \\rsq", "GL $\\Delta$", "GL ML", "GL AP", "PH \\rsq", "PH $\\Delta$", "PH ML", "PH AP"], rows, "lcccccccc")
else:
    L.append("% S5 pending: outputs/ablation_prep not found\n")

out = "\n".join(L)
# The nine-column ablation table does not fit one column of the two-column supplement: make it span both.
if fs:
    i = out.find("\\begin{table}[H]\\centering\\caption{Preprocessing and training-choice ablation")
    if i >= 0:
        j = out.find("\\end{table}", i) + len("\\end{table}")
        block = out[i:j].replace("\\begin{table}[H]", "\\begin{table*}[t]", 1).replace("\\end{table}", "\\end{table*}")
        out = out[:i] + block + out[j:]
(HERE / "supp_tables.tex").write_text(out, encoding="utf-8")
print("wrote supp_tables.tex;", "ablation" if fs else "no ablation yet")
