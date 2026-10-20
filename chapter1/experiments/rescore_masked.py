# -*- coding: utf-8 -*-
"""Re-score saved per-frame predictions with the three masking rules adopted on 2026-10-10
(Verification tab of the Chapter 1 status report):

  1. frame 0 dropped from every trial (GroundLink targets read ~0 BW there in 279/279 trials);
  2. CoP channels (px, pz) scored only on frames where that plate is loaded, measured vertical > 5 %BW
     (the threshold of the contact-only CoP error in biomech_units.py); force channels use every frame;
  3. GroundLink vertical RMSE reported with and without low-load frames (total measured vertical < 0.6 BW),
     which mix flight phases with feet off the plates.

R2 is computed exactly as src/train.py analyze_results does (per channel, pooled over the masked frames of
all test trials of the fold, then the mean of the 10 channels), so the "orig" column reproduces the logged
fold R2 and the "masked" column differs only by the rules above. Inputs: the .npz files written by
e1_multiseed_corrected.py (LOSO) and trial_split_corrected.py (trial split).

  python chapter1/experiments/rescore_masked.py --pred runs/hprc/outputs/v2/e1/pred runs/hprc/outputs/e1_pred/pred \
         runs/hprc/outputs/v2/ts/pred runs/hprc/outputs/ts_pred/pred --out outputs/rescore_masked
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd

CONTACT_BW = 0.05      # plate loaded: measured vertical above 5 %BW
LOW_TOTAL_BW = 0.6     # GroundLink low-load frames: total measured vertical below 0.6 BW


def r2(y, yh):
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - yh) ** 2).sum() / ss if ss > 0 else np.nan


def score_file(f):
    z = np.load(f, allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    p = z["preds"] * z["std"] + z["mean"]; a = z["acts"] * z["std"] + z["mean"]
    m = z["mask"] > 0
    m0 = m.copy(); m0[:, 0] = False                                   # rule 1
    iv = {k: cols.index(f"ground_force_{k}_vert") for k in (1, 2)}
    tot = a[..., iv[1]] + a[..., iv[2]]
    out = dict(file=Path(f).name, dataset=str(z["dataset"]), protocol=str(z["protocol"]), config=str(z["config"]),
               seed=int(z["seed"]), held_out=str(z["held_out"]) if "held_out" in z.files else "")
    orig, masked = [], []
    for j, c in enumerate(cols):
        orig.append(r2(a[..., j][m], p[..., j][m]))
        mm = m0
        if c.endswith(("_px", "_pz")):                                # rule 2
            plate = int(c.split("_")[2])
            mm = m0 & (a[..., iv[plate]] > CONTACT_BW)
        masked.append(r2(a[..., j][mm], p[..., j][mm]))
        out[f"r2_masked_{c}"] = masked[-1]
    out["r2_orig"] = float(np.nanmean(orig)); out["r2_masked"] = float(np.nanmean(masked))
    for name, idx in (("vert", [iv[1], iv[2]]),):
        e = (p[..., idx] - a[..., idx]) ** 2
        out["vert_rmse_pbw_all"] = float(np.sqrt(e[m0].mean()) * 100)
        hi = m0 & (tot >= LOW_TOTAL_BW)                               # rule 3
        out["vert_rmse_pbw_loaded"] = float(np.sqrt(e[hi].mean()) * 100)
        out["low_load_frac"] = float(1 - hi.sum() / m0.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/rescore_masked")
    args = ap.parse_args()
    # The first --pred folder that holds a (dataset, protocol, config, seed, fold) wins: list the retrained
    # GroundLink folders before the older ones that also hold (mis-scaled) GroundLink files.
    rows, seen, files = [], set(), []
    for d in args.pred:
        for f in sorted(glob.glob(str(Path(d) / "*.npz"))):
            r = score_file(f); key = (r["dataset"], r["protocol"], r["config"], r["seed"], r["held_out"])
            if key in seen: continue
            seen.add(key); rows.append(r); files.append(f)
    df = pd.DataFrame(rows)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "folds.csv", index=False)
    per_seed = df.groupby(["dataset", "protocol", "config", "seed"]).mean(numeric_only=True).reset_index()
    summ = per_seed.groupby(["dataset", "protocol", "config"]).agg(
        n_seeds=("seed", "size"), r2_orig=("r2_orig", "mean"), r2_masked=("r2_masked", "mean"),
        r2_masked_sd=("r2_masked", "std"), vert_all=("vert_rmse_pbw_all", "mean"),
        vert_loaded=("vert_rmse_pbw_loaded", "mean"), low_load_frac=("low_load_frac", "mean")).reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    print(f"{len(files)} files used ({len(df)} folds/runs)"); print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
