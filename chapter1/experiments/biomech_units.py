# -*- coding: utf-8 -*-
"""Biomechanical-unit metrics from saved per-frame predictions (the .npz files written by
e1_multiseed_corrected.py with E1_SAVE_PRED and trial_split_corrected.py with TS_SAVE_PRED).

Units. Force targets were divided by body weight before z-scoring (force_target_mode "nbw"), so a
de-normalised value is in body weights; x100 gives %BW. If a file says force_mode "newton" the
value is divided by mass*g from the trial's metadata instead. CoP targets are pelvis-relative,
heading-aligned and divided by subject height, in mm/m; x height (m) gives mm, /10 gives cm.

Per trial and channel: RMSE and MAE (%BW or cm). Per trial and plate: peak vertical force error
(%BW, signed and absolute; peak = max over valid frames), vertical impulse relative error (%; the
frame period cancels), CoP planar RMSE (cm, sqrt(mean(dx^2 + dz^2))).

Writes <out>/trials.csv (one row per trial), <out>/summary.csv (per dataset, protocol, config,
seed: mean over subjects of the per-subject trial means for LOSO; mean over trials for the trial
split), and <out>/r2_merged.csv (the per-task R2 CSVs of the same runs, concatenated).

  python chapter1/experiments/biomech_units.py --pred outputs/e1_pred/pred outputs/ts_pred/pred \
         --r2 "outputs/e1_pred/*_multiseed.csv" "outputs/ts_pred/*_trial_split.csv" --out outputs/biomech_units
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd

G = 9.81
FORCE_SUFFIX = ("_ml", "_ap", "_vert")


def denorm(a, mask, mean, std):
    out = a.astype(np.float64) * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    out[mask <= 0] = np.nan
    return out


def trial_rows(f):
    z = np.load(f, allow_pickle=False)
    cols = [str(c) for c in z["columns"]]
    preds = denorm(z["preds"], z["mask"], z["mean"], z["std"])
    acts = denorm(z["acts"], z["mask"], z["mean"], z["std"])
    meta = z["meta"]                                    # (N, 2) = [mass kg, height m]
    n = preds.shape[0]
    trials = [str(t) for t in z["trials"]] if len(z["trials"]) == n else [f"trial{i}" for i in range(n)]
    subjects = [str(s) for s in z["subjects"]] if len(z["subjects"]) == n else [str(z["held_out"])] * n
    force_mode = str(z["force_mode"])
    base = dict(dataset=str(z["dataset"]), protocol=str(z["protocol"]), config=str(z["config"]),
                model=str(z["model"]), seed=int(z["seed"]), held_out=str(z["held_out"]), file=Path(f).name)
    rows = []
    for i in range(n):
        mass, height = float(meta[i, 0]), float(meta[i, 1])
        row = dict(base, trial=trials[i], subject=subjects[i], mass_kg=mass, height_m=height,
                   n_frames=int((z["mask"][i] > 0).sum()))
        p, a = preds[i], acts[i]
        for j, c in enumerate(cols):
            pj, aj = p[:, j], a[:, j]
            if c.endswith(FORCE_SUFFIX):
                k = 100.0 if force_mode == "nbw" else (100.0 / (mass * G) if mass > 0 else np.nan)
                unit = "pbw"
            elif c.endswith(("_px", "_pz")):
                k = height / 10.0 if height > 0 else np.nan   # mm/m -> mm -> cm
                unit = "cm"
            else:
                continue
            e = (pj - aj) * k
            row[f"rmse_{unit}_{c}"] = float(np.sqrt(np.nanmean(e ** 2)))
            row[f"mae_{unit}_{c}"] = float(np.nanmean(np.abs(e)))
        # per-plate quantities
        for plate in ("ground_force_1", "ground_force_2", "ground_force_s"):
            v = f"{plate}_vert"
            contact = None
            if v in cols:
                j = cols.index(v)
                k = 100.0 if force_mode == "nbw" else (100.0 / (mass * G) if mass > 0 else np.nan)
                pv, av = p[:, j] * k, a[:, j] * k
                valid = np.isfinite(av)                           # padded frames are NaN after denorm
                # contact = frames where the plate carries more than 5 %BW of measured vertical force; CoP is
                # undefined and impulse meaningless when a plate is unloaded (those frames still count in R2)
                contact = valid & (av > 5.0)
                row[f"contact_frac_{plate}"] = float(contact.sum() / valid.sum()) if valid.sum() else np.nan
                if valid.any():
                    row[f"peak_err_pbw_{plate}"] = float(np.nanmax(pv) - np.nanmax(av))
                    row[f"peak_abserr_pbw_{plate}"] = abs(row[f"peak_err_pbw_{plate}"])
                    row[f"peak_act_pbw_{plate}"] = float(np.nanmax(av))
                    ia, ip = np.nansum(av), np.nansum(pv)
                    row[f"impulse_relerr_pct_{plate}"] = float(100.0 * (ip - ia) / ia) if (contact.sum() >= 10 and ia > 0) else np.nan
            px, pz = f"{plate}_px", f"{plate}_pz"
            if px in cols and pz in cols and height > 0:
                jx, jz = cols.index(px), cols.index(pz)
                dx = (p[:, jx] - a[:, jx]) * height / 10.0
                dz = (p[:, jz] - a[:, jz]) * height / 10.0
                d2 = dx ** 2 + dz ** 2
                row[f"cop_rmse_cm_{plate}"] = float(np.sqrt(np.nanmean(d2)))
                if contact is not None:
                    row[f"cop_rmse_cm_contact_{plate}"] = float(np.sqrt(np.nanmean(d2[contact]))) if contact.sum() >= 10 else np.nan
        rows.append(row)
    return rows


def summarise(trials: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in trials.columns if c.split("_")[0] in ("rmse", "mae", "peak", "impulse", "cop", "contact")]
    out = []
    for (ds, prot, cfg, seed), g in trials.groupby(["dataset", "protocol", "config", "seed"]):
        if prot == "loso":
            per_subj = g.groupby("subject")[metric_cols].mean()
            m, sd, n = per_subj.mean(), per_subj.std(), len(per_subj)
        else:
            m, sd, n = g[metric_cols].mean(), g[metric_cols].std(), len(g)
        med = g[metric_cols].median()                      # per-trial medians: the distributions have heavy tails
        expected = {"gl": 6, "ph": 10}.get(ds)
        row = dict(dataset=ds, protocol=prot, config=cfg, model=g["model"].iloc[0], seed=seed, n_units=n,
                   n_trials=len(g), complete=(prot != "loso") or (expected is None) or (n >= expected))
        row.update({c: float(m[c]) for c in metric_cols})
        row.update({f"{c}_sd": float(sd[c]) for c in metric_cols})
        row.update({f"{c}_med": float(med[c]) for c in metric_cols})
        # convenience: force channels pooled over plates
        for ax in ("vert", "ap", "ml"):
            cs = [c for c in metric_cols if c.startswith("rmse_pbw_") and c.endswith(f"_{ax}")]
            if cs: row[f"rmse_pbw_{ax}_pooled"] = float(np.nanmean([m[c] for c in cs]))
        cs = [c for c in metric_cols if c.startswith("cop_rmse_cm_") and "contact" not in c]
        if cs: row["cop_rmse_cm_pooled"] = float(np.nanmean([m[c] for c in cs]))
        cs = [c for c in metric_cols if c.startswith("cop_rmse_cm_contact_")]
        if cs: row["cop_rmse_cm_contact_pooled"] = float(np.nanmean([m[c] for c in cs]))
        cs = [c for c in metric_cols if c.startswith("peak_abserr_pbw_")]
        if cs: row["peak_abserr_pbw_pooled"] = float(np.nanmean([m[c] for c in cs]))
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="folders of .npz prediction files")
    ap.add_argument("--r2", nargs="*", default=[], help="globs of the per-task R2 CSVs to merge")
    ap.add_argument("--out", default="outputs/biomech_units")
    a = ap.parse_args()
    files = sorted(f for d in a.pred for f in glob.glob(str(Path(d) / "*.npz")))
    if not files:
        raise SystemExit("no .npz prediction files found")
    rows = []
    for f in files:
        rows.extend(trial_rows(f))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    trials = pd.DataFrame(rows); trials.to_csv(out / "trials.csv", index=False)
    summ = summarise(trials); summ.to_csv(out / "summary.csv", index=False)
    if a.r2:
        parts = [pd.read_csv(f) for pat in a.r2 for f in sorted(glob.glob(pat))]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out / "r2_merged.csv", index=False)
    show = [c for c in ["dataset", "protocol", "config", "seed", "n_units", "rmse_pbw_vert_pooled", "rmse_pbw_ap_pooled",
                        "rmse_pbw_ml_pooled", "peak_abserr_pbw_pooled", "cop_rmse_cm_pooled", "cop_rmse_cm_contact_pooled"] if c in summ.columns]
    print(f"{len(files)} files, {len(trials)} trials -> {out}")
    print(summ[show].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
