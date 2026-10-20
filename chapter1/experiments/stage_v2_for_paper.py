# -*- coding: utf-8 -*-
"""Stage the v2 run family (GroundLink retrained after the kN fix of 2026-10-08) in the folder layout that
paper/jbhi/build_results.py, make_figures.py and make_supplement.py read, so the paper rebuilds from one run
family without editing every path. Point the paper scripts here with JBHI_CODE_ROOT=<this folder>.

GroundLink: every file comes from the v2 retrain (runs/hprc/outputs/v2, runs/hprc/runs/v2, Grace jobs
20067543-45, 20072023, 20072035, 20082570, 20111484). Patient Handling: headline and trial split from the FASTER rerun
with saved predictions (runs/hprc/outputs/e1_pred, ts_pred), the rest from the published PH files (PH targets
were never affected). Cross-lab columns come from the eval-only *_xlaball runs (every subject of the other lab)
when they exist, else from the single-subject runs.

  python chapter1/experiments/stage_v2_for_paper.py            # -> outputs/paper_v2_stage
"""
import glob, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "runs" / "hprc"
OUT = ROOT / "outputs" / "paper_v2_stage"


def cat(pattern):
    fs = sorted(glob.glob(str(pattern)))
    if not fs:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def put(df, rel):
    p = OUT / rel; p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False); print(f"  {rel}: {len(df)} rows")


def copy(src, rel):
    p = OUT / rel; p.parent.mkdir(parents=True, exist_ok=True)
    (shutil.copytree(src, p, dirs_exist_ok=True) if Path(src).is_dir() else shutil.copy2(src, p)); print(f"  {rel}: copied")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    e1 = {"gl": cat(H / "outputs/v2/e1/gl_*_multiseed.csv"), "ph": cat(H / "outputs/e1_pred/ph_*_multiseed.csv")}
    ts = {"gl": cat(H / "outputs/v2/ts/gl_*_trial_split.csv"), "ph": cat(H / "outputs/ts_pred/ph_*_trial_split.csv")}
    for tag in ("gl", "ph"):
        put(e1[tag], f"outputs/e1_multiseed_corrected/{tag}_multiseed.csv")
        put(e1[tag][e1[tag].seed == 42][["config", "subject", "r2", "ml", "ap"]], f"outputs/hpo_confirm/{tag}_confirm.csv")
        num = ts[tag].select_dtypes("number").columns.drop("seed")
        t = ts[tag].groupby(["config", "model", "dataset", "protocol"], as_index=False)[list(num)].mean()   # mean of 3 seeds
        t.insert(3, "seed", "42-44"); put(t, f"outputs/trial_split_corrected/{tag}_trial_split.csv")
    for arm in ("baseline", "corot", "perfold", "both"):
        copy(H / f"outputs/v2/ablation/gl_{arm}.csv", f"outputs/ablation/gl_{arm}.csv")
        copy(ROOT / f"outputs/ablation/ph_{arm}.csv", f"outputs/ablation/ph_{arm}.csv")
    put(cat(H / "outputs/v2/ablation_prep/*/gl_ablation_prep.csv"), "outputs/ablation_prep/gl_ablation_prep.csv")
    copy(ROOT / "outputs/ablation_prep/ph_ablation_prep.csv", "outputs/ablation_prep/ph_ablation_prep.csv")
    for m in ("cnn-lstm", "cnn"):                 # frame factorial repeated at each model's tuned config (Grace 20111484)
        for tag in ("gl", "ph"):
            for arm in ("baseline", "corot", "perfold", "both"):
                copy(H / f"outputs/v2/ablation_{m}/{tag}_{arm}.csv", f"outputs/ablation_{m}/{tag}_{arm}.csv")
    for m in ("icf", "cnn-lstm", "cnn"):
        gl = pd.read_csv(H / f"runs/v2/m6_{m}_gl/results/loso_results.csv")
        xa = H / f"runs/v2/m6_{m}_gl_xlaball/results/loso_results.csv"
        if xa.exists():                       # cross-lab on every PH subject replaces the ph_1-only columns
            x = pd.read_csv(xa).set_index("subject"); keep = [c for c in gl.columns if not c.startswith("ph_")]
            gl = gl[keep].join(x[[c for c in x.columns if c.startswith("ph_")]], on="subject")
        put(gl, f"runs/m6_{m}_gl/results/loso_results.csv")
        copy(ROOT / f"runs/m6_{m}_ph/results/loso_results.csv", f"runs/m6_{m}_ph/results/loso_results.csv")
        ph = pd.read_csv(H / f"runs/v2/m6_phxlab_{m}/results/loso_results.csv")
        xa = H / f"runs/v2/m6_phxlab_{m}_xlaball/results/loso_results.csv"
        if xa.exists():
            x = pd.read_csv(xa).set_index("subject"); keep = [c for c in ph.columns if not c.startswith("gl_")]
            ph = ph[keep].join(x[[c for c in x.columns if c.startswith("gl_")]], on="subject")
        put(ph, f"runs/m6_phxlab_{m}/results/loso_results.csv")
        copy(H / f"outputs/v2_m6_combined_{m}/results/combined_dual_holdout_results.csv",
             f"outputs/m6_combined_{m}/results/combined_dual_holdout_results.csv")
    copy(H / "runs/v2/m6_recover_gl/results/loso_results.csv", "runs/m6_recover_gl/results/loso_results.csv")
    copy(ROOT / "runs/m6_recover_ph/results/loso_results.csv", "runs/m6_recover_ph/results/loso_results.csv")
    for tag in ("gl", "ph"):
        v2 = ROOT / f"outputs/v2/e3_ridge_loso_{tag}.csv"
        copy(v2 if v2.exists() else ROOT / f"outputs/e3_ridge_loso_{tag}.csv", f"outputs/e3_ridge_loso_{tag}.csv")
    for f in ("e2_matched_gl.csv", "e2_matched_ph.csv"):            # submitted-pipeline history, unchanged
        copy(ROOT / "outputs" / f, f"outputs/{f}")
    for d in ("gl_standard_2026-02-17", "ph_standard_2026-02-19"):
        copy(ROOT / "outputs" / d, f"outputs/{d}")
    for d in ("rescore_masked", "biomech_units_v2"):
        copy(ROOT / "outputs" / d, f"outputs/{d}")
    print("staged ->", OUT)


if __name__ == "__main__":
    main()
