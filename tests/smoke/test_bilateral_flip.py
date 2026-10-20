# -*- coding: utf-8 -*-
"""Check that bilateral_flip swaps every left/right marker pair of a marker set (no marker left on the wrong side).

  python tests/smoke/test_bilateral_flip.py            (checks common20 and full40)

Builds the input column names the pipeline uses ('<full name>_<axis>[_vel|_acc]'), fills each column with a
unique value, flips, and reports for every L*/R* marker whether its columns now hold its partner's values.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (MARKER_NAME_MAP, ESSENTIAL_MARKER_SHORT_NAMES_COMBINED as W20,
                        ESSENTIAL_MARKER_SHORT_NAMES_WIND_TURBINE as W40)
from src.data.augmentation import bilateral_flip

ok_all = True
for name, mk in [("common20", W20), ("full40", W40)]:
    cols = [f"{MARKER_NAME_MAP.get(m, m)}_{ax}{s}" for m in mk for ax in "XYZ" for s in ("", "_vel", "_acc")]
    x = torch.arange(1, len(cols) + 1, dtype=torch.float32).repeat(4, 1)
    tgt = torch.zeros(4, 5); mask = torch.ones(4)
    fx, _, _ = bilateral_flip(x, tgt, mask, cols, ["ground_force_s_ml", "ground_force_s_vert", "ground_force_s_ap",
                                                    "ground_force_s_px", "ground_force_s_pz"])
    src = {round(abs(float(v))): c for c, v in zip(cols, x[0])}
    bad = []
    for m in mk:
        if m[0] not in "LR" or m == "MChest":
            continue
        partner = ("R" if m[0] == "L" else "L") + m[1:]
        col = f"{MARKER_NAME_MAP.get(m, m)}_Y"
        got = src[round(abs(float(fx[0, cols.index(col)])))]
        want = f"{MARKER_NAME_MAP.get(partner, partner)}_Y" if partner in mk else col
        status = "swapped" if got != col else ("unpaired (partner not in set)" if partner not in mk else "NOT SWAPPED")
        if status == "NOT SWAPPED" or got != want:
            bad.append((m, status))
        if partner not in mk:
            print(f"  {name}: {m} has no partner in the set -> stays in place (mirrored only)")
    print(f"{name}: {len(mk)} markers, problems: {bad or 'none'}")
    ok_all &= not bad

# Full pipeline column list (markers + rel pairs + CoM, as data_pipeline.input_feature_names), 20-marker set:
# CoM X columns must be negated, CoM Y/Z untouched; a rel pair whose partner marker is missing (RWRL) must not be
# swapped with its always-zero mirror column; a complete rel pair (LSHO-LELBL / RSHO-RELBL) must be swapped.
PAIRS = [("LSHO", "LELBL"), ("LELBL", "LWRL"), ("RSHO", "RELBL"), ("RELBL", "RWRL"), ("LKNE", "LANKL"),
         ("LANKL", "LTOE"), ("RKNE", "RANKM"), ("RANKM", "RTOE"), ("LSHO", "RSHO"), ("LASI", "LSHO")]
COM = ["COM_vel_X", "COM_acc_X", "COM_vel_Y", "COM_acc_Y", "COM_vel_Z", "COM_acc_Z"]
base = sorted(f"{MARKER_NAME_MAP.get(m, m)}_{ax}" for m in W20 for ax in "XYZ")
cols = [c for b in base for c in (b, b + "_vel", b + "_acc")]
cols += [f"rel_{a}_{b}_{ax}" for a, b in PAIRS for ax in "XYZ"] + COM
x = torch.arange(1, len(cols) + 1, dtype=torch.float32).repeat(4, 1)
fx, _, _ = bilateral_flip(x, torch.zeros(4, 5), torch.ones(4), cols,
                          ["ground_force_s_ml", "ground_force_s_vert", "ground_force_s_ap", "ground_force_s_px", "ground_force_s_pz"])
v = lambda c: float(fx[0, cols.index(c)]); o = lambda c: float(x[0, cols.index(c)])
checks = {
    "COM_vel_X negated": v("COM_vel_X") == -o("COM_vel_X"),
    "COM_acc_X negated": v("COM_acc_X") == -o("COM_acc_X"),
    "COM_vel_Y / COM_acc_Z unchanged": v("COM_vel_Y") == o("COM_vel_Y") and v("COM_acc_Z") == o("COM_acc_Z"),
    "rel_LELBL_LWRL_Y stays (RWRL missing)": v("rel_LELBL_LWRL_Y") == o("rel_LELBL_LWRL_Y"),
    "rel_RELBL_RWRL_Y stays": v("rel_RELBL_RWRL_Y") == o("rel_RELBL_RWRL_Y"),
    "rel_LSHO_LELBL_Y <- rel_RSHO_RELBL_Y": v("rel_LSHO_LELBL_Y") == o("rel_RSHO_RELBL_Y"),
}
for k, ok in checks.items():
    print(f"  {'OK  ' if ok else 'FAIL'} {k}")
ok_all &= all(checks.values())
print("ALL OK" if ok_all else "FAILURES")
sys.exit(0 if ok_all else 1)
