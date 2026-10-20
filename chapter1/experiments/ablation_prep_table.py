# -*- coding: utf-8 -*-
"""Summarize the ablation CSVs across seeds: per arm mean ± SD (across seeds) of the fold-mean R2,
paired delta vs 'full' per seed, per-channel means. Usage: python chapter1/experiments/ablation_prep_table.py
The authoritative copy of this summary is paper/tbme/make_supplement.py:61-87 (supplement table); this is a console preview."""
import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ORDER = ["full", "no_pelvis", "no_heading", "no_height", "no_biomech", "no_derived", "no_rel", "no_flip", "no_peak"]
for tag, nf in (("gl", 6), ("ph", 10)):
    df = pd.read_csv(ROOT / f"outputs/ablation_prep/{tag}_ablation_prep.csv")
    print(f"\n================ {tag.upper()} ================")
    fm = df.groupby(["arm", "seed"]).agg(r2=("r2", "mean"), vert=("vert", "mean"), ml=("ml", "mean"), ap=("ap", "mean"), n=("r2", "size")).reset_index()
    fm = fm[fm.n == nf]  # complete seeds only
    full = fm[fm.arm == "full"].set_index("seed")
    rows = []
    for arm in ORDER:
        a = fm[fm.arm == arm].set_index("seed")
        if a.empty: continue
        common = a.index.intersection(full.index)
        d = (a.loc[common, "r2"] - full.loc[common, "r2"])
        # paired per subject (all seeds pooled): wins
        sub_a = df[(df.arm == arm) & df.seed.isin(common)].set_index(["seed", "subject"])["r2"]
        sub_f = df[(df.arm == "full") & df.seed.isin(common)].set_index(["seed", "subject"])["r2"]
        pd_ = (sub_a - sub_f).dropna()
        rows.append(dict(arm=arm, seeds=len(a), r2=f"{a.r2.mean():.3f} ± {a.r2.std(ddof=0):.3f}",
                         vert=f"{a.vert.mean():.3f}", ml=f"{a.ml.mean():.3f}", ap=f"{a.ap.mean():.3f}",
                         delta_vs_full=f"{d.mean():+.3f}" if len(common) else "", delta_seed_range=f"[{d.min():+.3f}, {d.max():+.3f}]" if len(common) else "",
                         subj_worse=f"{int((pd_ < 0).sum())}/{len(pd_)}" if len(pd_) else ""))
    print(pd.DataFrame(rows).to_string(index=False))
