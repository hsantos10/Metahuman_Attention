# -*- coding: utf-8 -*-
"""Rebuild every number for the JBHI submission from the corrected-pipeline runs
(July 2026) plus the frozen submitted-pipeline artifacts. Writes
RESULTS_CORRECTED.md (human-readable tables) and results.json (for the draft).

Sources (under JBHI_CODE_ROOT, default ../results/paper_v2_stage):
  outputs/hpo_confirm/{gl,ph}_confirm.csv      re-tuned ranking, corrected pipeline, seed 42
  outputs/e1_multiseed_corrected/*.csv         multi-seed (when finished)
  outputs/ablation/{gl,ph}_{arm}.csv           2x2 co-rotation x per-fold input norm, ICF, paper HP
  runs/m6_{icf,cnn,cnn-lstm}_{gl,ph}/results   paper HP on corrected pipeline (+ cross-lab GL->PH columns)
  runs/m6_phxlab_*/results                     PH-trained, tested on GL (cross-lab PH->GL)
  outputs/m6_combined_*/results                combined dual-holdout, corrected pipeline
  runs/m6_recover_{gl,ph}/results              excluded subjects re-included
  outputs/e3_ridge_loso_{gl,ph}.csv            ridge baseline (submitted pipeline)
  outputs/e2_matched_*.csv                     matched pooling comparison (submitted pipeline)
  outputs/{gl,ph}_standard_*/                  trial-level split (submitted pipeline)
"""
import json, glob, os, re
from pathlib import Path
import numpy as np, pandas as pd

CODE = Path(__import__("os").getenv("JBHI_CODE_ROOT", str(Path(__file__).resolve().parents[1] / "results" / "paper_v2_stage")))   # v2 run family (code/chapter1/experiments/stage_v2_for_paper.py)
OUT = Path(__file__).resolve().parent
R = {}          # numbers for the draft
MD = []         # markdown lines

def h(s): MD.append(f"\n## {s}\n")
def note(s): MD.append(s + "\n")
def table(df, floatfmt=3):
    MD.append(df.to_markdown(floatfmt=f".{floatfmt}f") + "\n")

CH = ["ml", "vert", "ap", "px", "pz"]
def chan_means(df, prefix):
    """mean R2 per channel type (both plates) from wide per-channel columns"""
    out = {}
    for c in CH:
        cols = [k for k in df.columns if k.startswith(prefix) and k.endswith("_" + c) and "mae" not in k]
        out[c] = float(np.nanmean(df[cols].values)) if cols else np.nan
    return out

# ---------------------------------------------------------------- 1. headline, re-tuned, corrected
h("1. Headline subject-held-out R2, corrected pipeline, per-model re-tuned (seed 42)")
note("Source: `outputs/hpo_confirm/*_confirm.csv`. Each model tuned by the same procedure (Optuna TPE + Hyperband on one "
     "proxy fold, corrected pipeline), then full LOSO. ICF-grid128 is the depth/width grid winner, kept as a robustness row.")
rows = []
for tag in ["gl", "ph"]:
    df = pd.read_csv(CODE / f"outputs/hpo_confirm/{tag}_confirm.csv")
    g = df.groupby("config").agg(R2=("r2", "mean"), R2_sd=("r2", "std"), ML=("ml", "mean"), AP=("ap", "mean"), n=("r2", "size")).reset_index()
    g.insert(0, "dataset", tag.upper())
    rows.append(g)
    R[f"headline_{tag}"] = {r.config: dict(r2=round(r.R2, 3), sd=round(r.R2_sd, 3), ml=round(r.ML, 3), ap=round(r.AP, 3), n=int(r.n)) for r in g.itertuples()}
    # paired Wilcoxon between top-2 and each pair
    from scipy.stats import wilcoxon
    piv = df.pivot(index="subject", columns="config", values="r2")
    pairs = {}
    cfgs = list(piv.columns)
    for i in range(len(cfgs)):
        for j in range(i + 1, len(cfgs)):
            a, b = cfgs[i], cfgs[j]
            d = piv[a] - piv[b]
            try:
                p = float(wilcoxon(piv[a], piv[b]).pvalue)
            except ValueError:
                p = float("nan")
            pairs[f"{a} vs {b}"] = dict(delta=round(float(d.mean()), 3), p=round(p, 3))
    R[f"headline_{tag}_pairs"] = pairs
table(pd.concat(rows))
note("Paired Wilcoxon on per-subject R2 (fold count 6 / 10; p-values near the floor of the test):")
for tag in ["gl", "ph"]:
    for k, v in R[f"headline_{tag}_pairs"].items():
        MD.append(f"- {tag.upper()} {k}: mean delta {v['delta']:+.3f}, p = {v['p']:.3f}")
MD.append("")

# ---------------------------------------------------------------- 1b. multi-seed (if present)
ms = sorted(glob.glob(str(CODE / "outputs/e1_multiseed_corrected/*_multiseed.csv")))
if ms:
    h("1b. Multi-seed headline (corrected pipeline), mean of per-seed fold means +/- SD across seeds")
    rows = []
    for f in ms:
        df = pd.read_csv(f)
        per_seed = df.groupby(["config", "seed"])[["r2", "vert", "ml", "ap"]].mean().reset_index()
        g = per_seed.groupby("config").agg(R2=("r2", "mean"), R2_sd_seeds=("r2", "std"), VERT=("vert", "mean"), ML=("ml", "mean"), AP=("ap", "mean"), seeds=("seed", "nunique")).reset_index()
        g.insert(0, "dataset", df["dataset"].iloc[0].upper())
        # folds = the smallest per-seed subject count, so an incomplete seed cannot pass as complete
        folds_done = df.groupby(["config", "seed"])["subject"].nunique().groupby("config").min().to_dict()
        g["folds"] = g["config"].map(folds_done)
        rows.append(g)
        R[f"multiseed_{df['dataset'].iloc[0]}"] = g.set_index("config").round(3).to_dict("index")
    table(pd.concat(rows))
else:
    h("1b. Multi-seed headline")
    note("Not yet available: `outputs/e1_multiseed_corrected/` (run launched 2026-09-21).")

# ---------------------------------------------------------------- 2. correction effect at fixed HP (ablation 2x2)
h("2. Effect of the two pipeline corrections at fixed hyperparameters (ICF, paper config d64/L2, seed 42)")
note("Source: `outputs/ablation/`. Arms: baseline = submitted pipeline; corot = horizontal force targets rotated with the "
     "heading frame; perfold = input z-score fit on training fold instead of per trial; both = corrected pipeline.")
rows = []
for tag in ["gl", "ph"]:
    for arm in ["baseline", "corot", "perfold", "both"]:
        df = pd.read_csv(CODE / f"outputs/ablation/{tag}_{arm}.csv")
        cm = chan_means(df, "r2_")
        rows.append(dict(dataset=tag.upper(), arm=arm, R2=df["r2"].mean(), R2_sd=df["r2"].std(), **{k.upper(): v for k, v in cm.items()}, n=len(df)))
abl = pd.DataFrame(rows)
table(abl)
R["ablation"] = {f"{r.dataset}_{r.arm}": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r._asdict().items() if k not in ("Index",)} for r in abl.itertuples()}
# paired per-subject deltas baseline -> both
for tag in ["gl", "ph"]:
    b = pd.read_csv(CODE / f"outputs/ablation/{tag}_baseline.csv"); c = pd.read_csv(CODE / f"outputs/ablation/{tag}_both.csv")
    key = "subject" if "subject" in b.columns else b.columns[0]
    m = b.merge(c, on=key, suffixes=("_b", "_c"))
    d = m["r2_c"] - m["r2_b"]
    from scipy.stats import wilcoxon
    try: p = float(wilcoxon(m["r2_c"], m["r2_b"]).pvalue)
    except ValueError: p = float("nan")
    ap_b = np.nanmean(m[[k for k in m.columns if k.endswith("_ap_b")]].values); ap_c = np.nanmean(m[[k for k in m.columns if k.endswith("_ap_c")]].values)
    R[f"ablation_paired_{tag}"] = dict(delta_r2=round(float(d.mean()), 3), wins=int((d > 0).sum()), n=int(len(d)), p=round(p, 3), ap_before=round(float(ap_b), 3), ap_after=round(float(ap_c), 3))
    MD.append(f"- {tag.upper()} baseline -> both, paired per subject: delta R2 {d.mean():+.3f}, {int((d>0).sum())}/{len(d)} subjects improve, Wilcoxon p = {p:.3f}; AP R2 {ap_b:.3f} -> {ap_c:.3f}")
MD.append("")
# same 2x2 for the CNN-LSTM and the CNN at their tuned per-dataset configs (outputs/ablation_<model>/, seed 42)
if (CODE / "outputs/ablation_cnn-lstm").exists():
    h("2b. Frame 2x2 repeated for the CNN-LSTM and the CNN (tuned configs, seed 42)")
    rows = []
    for mdl in ["cnn-lstm", "cnn"]:
        for tag in ["gl", "ph"]:
            for arm in ["baseline", "corot", "perfold", "both"]:
                df = pd.read_csv(CODE / f"outputs/ablation_{mdl}/{tag}_{arm}.csv")
                rows.append(dict(model=mdl, dataset=tag.upper(), arm=arm, R2=df["r2"].mean(), **{k.upper(): v for k, v in chan_means(df, "r2_").items()}, n=len(df)))
    ablm = pd.DataFrame(rows); table(ablm)
    R["ablation_models"] = {f"{r.model}_{r.dataset}_{r.arm}": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r._asdict().items() if k != "Index"} for r in ablm.itertuples()}

# ---------------------------------------------------------------- 3. paper HP on corrected pipeline (m6) + channel hierarchy
h("3. Paper hyperparameters on the corrected pipeline (m6, seed 42): overall and per-channel R2")
note("Source: `runs/m6_{icf,cnn,cnn-lstm}_{gl,ph}/results/loso_results.csv`. Same models and hyperparameters as the "
     "submitted paper, corrected pipeline only. Per-channel means pool both plates.")
rows = []
for tag in ["gl", "ph"]:
    for m, lab in [("icf", "ICF"), ("cnn-lstm", "CNN-LSTM"), ("cnn", "CNN")]:
        f = CODE / f"runs/m6_{m}_{tag}/results/loso_results.csv"
        if not f.exists(): continue
        df = pd.read_csv(f)
        cm = chan_means(df, f"{tag}_r2_")
        rows.append(dict(dataset=tag.upper(), model=lab, R2=df[f"{tag}_r2"].mean(), R2_sd=df[f"{tag}_r2"].std(), **{k.upper(): v for k, v in cm.items()}, n=len(df)))
m6 = pd.DataFrame(rows); table(m6)
R["m6_paperHP_corrected"] = {f"{r.dataset}_{r.model}": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r._asdict().items() if k != "Index"} for r in m6.itertuples()}
# submitted-pipeline per-channel hierarchy, from ablation baseline (same as paper pipeline)
h("3b. Channel difficulty ordering, submitted vs corrected (ICF, paper HP)")
rows = []
for tag in ["gl", "ph"]:
    b = chan_means(pd.read_csv(CODE / f"outputs/ablation/{tag}_baseline.csv"), "r2_")
    c = chan_means(pd.read_csv(CODE / f"outputs/ablation/{tag}_both.csv"), "r2_")
    for ch in CH:
        rows.append(dict(dataset=tag.upper(), channel=ch, submitted=b[ch], corrected=c[ch], delta=c[ch] - b[ch]))
hier = pd.DataFrame(rows); table(hier)
R["hierarchy"] = hier.round(3).to_dict("records")

# ---------------------------------------------------------------- 4. cross-lab, corrected
h("4. Cross-laboratory transfer, corrected pipeline (train on one laboratory, test on the other, no target-lab subjects)")
note("Source: GL->PH from the `ph_*` columns of `runs/m6_*_gl`; PH->GL from the `gl_*` columns of `runs/m6_phxlab_*`.")
rows = []
for m, lab in [("icf", "ICF"), ("cnn-lstm", "CNN-LSTM"), ("cnn", "CNN")]:
    f = CODE / f"runs/m6_{m}_gl/results/loso_results.csv"
    if f.exists():
        df = pd.read_csv(f)
        if "ph_r2" in df.columns:
            cm = chan_means(df, "ph_r2_")
            rows.append(dict(direction="GL -> PH", model=lab, R2=df["ph_r2"].mean(), R2_min=df["ph_r2"].min(), R2_max=df["ph_r2"].max(), **{k.upper(): v for k, v in cm.items()}))
    f = CODE / f"runs/m6_phxlab_{m}/results/loso_results.csv"
    if f.exists():
        df = pd.read_csv(f)
        if "gl_r2" in df.columns:
            cm = chan_means(df, "gl_r2_")
            rows.append(dict(direction="PH -> GL", model=lab, R2=df["gl_r2"].mean(), R2_min=df["gl_r2"].min(), R2_max=df["gl_r2"].max(), **{k.upper(): v for k, v in cm.items()}))
xl = pd.DataFrame(rows); table(xl)
R["crosslab_corrected"] = {f"{r.direction}_{r.model}": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r._asdict().items() if k != "Index"} for r in xl.itertuples()}

# ---------------------------------------------------------------- 5. combined dual-holdout, corrected + matched comparison
h("5. Combined training (both laboratories), dual-holdout CV, corrected pipeline")
note("Source: `outputs/m6_combined_*/results/combined_dual_holdout_results.csv`. Each fold holds out one GL and one PH subject.")
rows = []; matched = []
for m, lab in [("icf", "ICF"), ("cnn-lstm", "CNN-LSTM"), ("cnn", "CNN")]:
    f = CODE / f"outputs/m6_combined_{m}/results/combined_dual_holdout_results.csv"
    if not f.exists(): continue
    df = pd.read_csv(f)
    rows.append(dict(model=lab, GL_R2=df["gl_r2"].mean(), GL_sd=df["gl_r2"].std(), PH_R2=df["ph_r2"].mean(), PH_sd=df["ph_r2"].std(), folds=len(df)))
    # matched comparison on the corrected pipeline: same held-out subjects, single-lab LOSO (m6) vs combined
    for tag, col in [("gl", "gl_held_out"), ("ph", "ph_held_out")]:
        single = CODE / f"runs/m6_{m}_{tag}/results/loso_results.csv"
        if not single.exists() or col not in df.columns: continue
        s = pd.read_csv(single).set_index("subject")[f"{tag}_r2"]
        mm = df[[col, f"{tag}_r2"]].rename(columns={col: "subject", f"{tag}_r2": "combined"})
        mm["single"] = mm["subject"].map(s)
        mm = mm.dropna()
        d = mm["combined"] - mm["single"]
        from scipy.stats import wilcoxon
        try: p = float(wilcoxon(mm["combined"], mm["single"]).pvalue)
        except ValueError: p = float("nan")
        matched.append(dict(model=lab, dataset=tag.upper(), n=len(mm), single_matched=mm["single"].mean(), combined=mm["combined"].mean(), delta=d.mean(), wins=int((d > 0).sum()), p=p))
comb = pd.DataFrame(rows); table(comb)
R["combined_corrected"] = comb.round(3).set_index("model").to_dict("index")
h("5b. Matched pooling comparison, corrected pipeline (same held-out subjects; single-laboratory LOSO vs combined)")
mt = pd.DataFrame(matched); table(mt)
R["matched_corrected"] = mt.round(3).to_dict("records")
h("5c. Matched pooling comparison, submitted pipeline (July revision, ICF)")
rows = []
for f, lab in [("e2_matched_gl.csv", "GL"), ("e2_matched_ph.csv", "PH")]:
    df = pd.read_csv(CODE / "outputs" / f); s = [c for c in df.columns if c.endswith("_loso")][0]
    rows.append(dict(dataset=lab, n=len(df), single_matched=df[s].mean(), combined=df["combined"].mean(), delta=df["delta"].mean(), wins=int((df["delta"] > 0).sum())))
table(pd.DataFrame(rows))
R["matched_submitted"] = pd.DataFrame(rows).round(3).to_dict("records")

# ---------------------------------------------------------------- 6. baselines & sensitivity
h("6. Ridge baseline (per-channel ridge on the same features and folds, submitted pipeline)")
rows = []
for tag in ["gl", "ph"]:
    df = pd.read_csv(CODE / f"outputs/e3_ridge_loso_{tag}.csv")
    cm = chan_means(df, f"{tag}_r2_")
    rows.append(dict(dataset=tag.upper(), R2=df[f"{tag}_r2"].mean(), R2_sd=df[f"{tag}_r2"].std(), **{k.upper(): v for k, v in cm.items()}, n=len(df)))
rg = pd.DataFrame(rows); table(rg)
R["ridge"] = rg.round(3).set_index("dataset").to_dict("index")

h("7. Exclusion sensitivity: excluded subjects re-included (ICF, paper HP, corrected pipeline)")
note("Source: `runs/m6_recover_{gl,ph}`. GL adds s001 (gl_1); PH adds the two excluded subjects.")
rows = []
for tag in ["gl", "ph"]:
    df = pd.read_csv(CODE / f"runs/m6_recover_{tag}/results/loso_results.csv")
    base = pd.read_csv(CODE / f"runs/m6_icf_{tag}/results/loso_results.csv")
    extra = sorted(set(df["subject"]) - set(base["subject"]))
    rows.append(dict(dataset=tag.upper(), n_all=len(df), R2_all=df[f"{tag}_r2"].mean(), n_base=len(base), R2_base=base[f"{tag}_r2"].mean(),
                     re_included=", ".join(extra), R2_re_included=", ".join(f"{v:.3f}" for v in df.set_index("subject").loc[extra, f"{tag}_r2"]),
                     R2_all_excluding_reincluded=df[~df["subject"].isin(extra)][f"{tag}_r2"].mean()))
rc = pd.DataFrame(rows); table(rc)
R["recover"] = rc.round(3).set_index("dataset").to_dict("index")

# ---------------------------------------------------------------- 8. trial-level split (submitted pipeline)
h("8. Trial-level random split (submitted pipeline; ICF tuned on that split)")
rows = []
for tag, d in [("gl", "gl_standard_2026-02-17"), ("ph", "ph_standard_2026-02-19")]:
    hits = glob.glob(str(CODE / f"outputs/{d}/**/*.csv"), recursive=True) + glob.glob(str(CODE / f"outputs/{d}/**/*.json"), recursive=True)
    val = None
    for f in hits:
        try:
            if f.endswith(".csv"):
                df = pd.read_csv(f)
                for c in df.columns:
                    if c.lower() in ("test_r2", "r2_test"):
                        val = float(df[c].iloc[-1]); break
            else:
                j = json.load(open(f))
                if isinstance(j, dict):
                    for k in ("test_r2", "r2_test", "R2"):
                        if k in j: val = float(j[k]); break
        except Exception:
            pass
        if val is not None:
            rows.append(dict(dataset=tag.upper(), test_R2=val, source=os.path.relpath(f, CODE))); break
    if val is None:
        rows.append(dict(dataset=tag.upper(), test_R2=np.nan, source="not found; paper values 0.860 (GL) / 0.856 (PH)"))
ts = pd.DataFrame(rows); table(ts)
R["trial_split"] = ts.to_dict("records")

# ---------------------------------------------------------------- 8b. trial-level split on the corrected pipeline (same configs as LOSO)
h("8b. Trial-level split, corrected pipeline, same configuration per model (seed 42)")
tsc = {}
for f in sorted(glob.glob(str(CODE / "outputs/trial_split_corrected/*_trial_split.csv"))):
    df = pd.read_csv(f); tag = df["dataset"].iloc[0]
    tsc[tag] = {r.config: dict(r2=round(float(r.r2), 3), vert=round(float(r.vert), 3), ml=round(float(r.ml), 3), ap=round(float(r.ap), 3)) for r in df.itertuples()}
if tsc:
    rows = [dict(dataset=t.upper(), config=c, **v) for t, d in tsc.items() for c, v in d.items()]
    table(pd.DataFrame(rows))
else:
    note("Not yet available: `outputs/trial_split_corrected/` (runs launched 2026-09-21).")
R["trial_split_corrected"] = tsc

# ---------------------------------------------------------------- 9. masked rescoring (decided 2026-10-10)
h("9. Rescored with the masking rules: frame 0 dropped, CoP only on loaded plates (> 5 %BW)")
note("Source: `outputs/rescore_masked/summary.csv` (code/chapter1/experiments/rescore_masked.py), the same saved predictions as sections 1b and 8b.")
mf = CODE / "outputs/rescore_masked/summary.csv"
if mf.exists():
    ms_ = pd.read_csv(mf); table(ms_)
    R["masked"] = {f"{r.dataset}_{r.protocol}_{r.config}": dict(orig=round(r.r2_orig, 3), masked=round(r.r2_masked, 3), vert_all=round(r.vert_all, 2), vert_loaded=round(r.vert_loaded, 2), low_load_frac=round(r.low_load_frac, 3)) for r in ms_.itertuples()}

# ---------------------------------------------------------------- 10. biomechanical units (per-trial means, LOSO)
h("10. Errors in body weights and centimetres (LOSO, mean over subjects of per-trial means, 3 seeds)")
uf = CODE / "outputs/biomech_units_v2/summary.csv"
if uf.exists():
    u = pd.read_csv(uf)
    cols_ = ["rmse_pbw_vert_pooled", "peak_abserr_pbw_pooled", "cop_rmse_cm_contact_pooled"]
    g_ = u.groupby(["dataset", "protocol", "config"])[cols_].mean().reset_index(); table(g_, 2)
    R["units"] = {f"{r.dataset}_{r.protocol}_{r.config}": dict(vert=round(r.rmse_pbw_vert_pooled, 1), peak=round(r.peak_abserr_pbw_pooled, 1), cop=round(r.cop_rmse_cm_contact_pooled, 1)) for r in g_.itertuples()}

# ---------------------------------------------------------------- write
(OUT / "RESULTS_CORRECTED.md").write_text("# Results for the JBHI submission, rebuilt from run outputs\n\nGenerated by `build_results.py`. "
                                         "All R2 are means over held-out subjects (LOSO folds) of the per-channel R2 averaged over the 10 output channels unless stated.\n"
                                         + "\n".join(MD), encoding="utf-8")
json.dump(R, open(OUT / "results.json", "w"), indent=1, default=float)
print("wrote", OUT / "RESULTS_CORRECTED.md", "and results.json")
