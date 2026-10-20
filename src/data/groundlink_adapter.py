
import numpy as np
import pandas as pd
import ezc3d
import torch
import re
import os
from src.config import MARKER_NAME_MAP


# Mapping from GroundLink (QTM/Motion/C3D) marker names to project standard names.
# Project Standard Names (keys in MARKER_NAME_MAP in utils.py):
# 'LSHO', 'RSHO', 'LELBL', 'LELBM', 'LWRL', 'LWRM', 'RELBL', 'RELBM', 'RWRL', 'RWRM',
# 'LKNE', 'RKNE', 'LTOE', 'LHEE', 'LANKL', 'RTOE', 'RHEE', 'RANKM', 'LASI', 'LPSI',
# 'RPSI', 'RASI', 'MChest', 'SENL', 'SENR', 'LANKM', 'RANKL'

GROUNDLINK_MARKER_MAP = {
    'QTMMod_l_sho_ant': 'LSHO', # Approximation
    'QTMMod_r_sho_ant': 'RSHO',
    'QTMMod_l_elb_lat': 'LELBL',
    'QTMMod_l_elb_med': 'LELBM',
    'QTMMod_l_wri_lat': 'LWRL',
    'QTMMod_l_wri_med': 'LWRM',
    'QTMMod_r_elb_lat': 'RELBL',
    'QTMMod_r_elb_med': 'RELBM',
    'QTMMod_r_wri_lat': 'RWRL',
    'QTMMod_r_wri_med': 'RWRM',
    'QTMMod_l_knee_lat': 'LKNE',
    'QTMMod_r_knee_lat': 'RKNE',
    'QTMMod_l_foot_toe': 'LTOE',
    'QTMMod_l_foot_heel': 'LHEE',
    'QTMMod_l_ank_lat': 'LANKL',
    'QTMMod_l_ank_med': 'LANKM',
    'QTMMod_r_foot_toe': 'RTOE',
    'QTMMod_r_foot_heel': 'RHEE',
    'QTMMod_r_ank_lat': 'RANKL',
    'QTMMod_r_ank_med': 'RANKM',
    'QTMMod_pelvis_l_asis': 'LASI',
    'QTMMod_pelvis_l_psis': 'LPSI',
    'QTMMod_pelvis_r_psis': 'RPSI',
    'QTMMod_pelvis_r_asis': 'RASI',
    # 'QTMMod_trunk_manu': 'MChest', # Approximation
}

_GROUNDLINK_MASS_MAP = {
    '1': 69.86,
    '2': 66.68,
    '3': 53.07,
    '4': 71.67,
    '5': 90.7,
    '6': 48.99,
    '7': 63.96,
}

_PRINTED_MARKER_UNIT_MSG = False
_PRINTED_GRF_UNIT_MSG = False


def _normalize_subject_id(raw: str) -> str:
    match = re.search(r"s(\d{3})", raw.lower())
    if match:
        return str(int(match.group(1)))
    digits = re.search(r"(\d+)", raw)
    return str(int(digits.group(1))) if digits else raw


def set_groundlink_mass_map(mass_map):
    """Update the module-level mass lookup (expects a mapping subject_id->kg)."""
    global _GROUNDLINK_MASS_MAP
    normalized = {}
    for key, value in mass_map.items():
        try:
            normalized_key = _normalize_subject_id(str(key))
            normalized[normalized_key] = float(value)
        except Exception:
            continue

    if normalized:
        _GROUNDLINK_MASS_MAP = normalized

def read_groundlink_c3d(file_path):
    """
    Reads a GroundLink C3D file and returns a DataFrame in the TRC format expected by the pipeline.

    The pipeline expects a DataFrame where:
    - Columns are multi-index (Marker, Axis) or flattened 'Marker_Axis'.
    - Time is implicit or first column.

    This function returns a DataFrame with columns renamed to standard marker names.
    It handles 250Hz -> 50Hz downsampling if needed (simple decimation).
    """
    try:
        c = ezc3d.c3d(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read C3D file {file_path}: {e}")

    labels = c['parameters']['POINT']['LABELS']['value']
    points = c['data']['points'] # Shape: (4, n_markers, n_frames)
    frame_rate = c['header']['points']['frame_rate']

    # Downsample factor
    target_rate = 50
    step = int(frame_rate / target_rate) if frame_rate > target_rate else 1

    # Marker unit handling
    # Keep marker coordinates in millimeters by default to match TRC/OpenSim files.
    marker_units_mode = os.getenv("GROUNDLINK_MARKER_UNITS", "mm").strip().lower()
    if marker_units_mode not in {"auto", "m", "mm"}:
        marker_units_mode = "mm"

    all_points_xyz = points[0:3, :, :]
    finite_points = all_points_xyz[np.isfinite(all_points_xyz)]
    abs_p95 = float(np.percentile(np.abs(finite_points), 95)) if finite_points.size else 0.0

    if marker_units_mode == "mm":
        marker_scale = 1.0
        resolved_marker_units = "mm"
    elif marker_units_mode == "m":
        marker_scale = 0.001
        resolved_marker_units = "m->mm"
    else:
        # Auto mode keeps canonical output in millimeters.
        # If values look meter-scale, convert m->mm.
        if abs_p95 > 10.0:
            marker_scale = 1.0
            resolved_marker_units = "auto(mm)"
        else:
            marker_scale = 1000.0
            resolved_marker_units = "auto(m->mm)"

    global _PRINTED_MARKER_UNIT_MSG
    if not _PRINTED_MARKER_UNIT_MSG:
        print(
            f"GroundLink marker units: mode={marker_units_mode}, resolved={resolved_marker_units}, "
            f"abs_p95={abs_p95:.3f}, scale={marker_scale}"
        )
        _PRINTED_MARKER_UNIT_MSG = True

    # Extract data
    data_dict = {}

    # Iterate over our map to find indices
    for gl_name, std_name in GROUNDLINK_MARKER_MAP.items():
        if gl_name in labels:
            idx = labels.index(gl_name)
            # points[0:3, idx, :] are X, Y, Z (GroundLink coordinates)
            x = points[0, idx, ::step] * marker_scale
            y = points[1, idx, ::step] * marker_scale
            z = points[2, idx, ::step] * marker_scale

            # Coordinate System Transformation:
            # GroundLink (QTM): Z is VERTICAL (UP), X and Y are horizontal
            # Patient Handling Model: Y is VERTICAL (UP), X and Z are horizontal
            # 
            # Transformation applied:
            # GroundLink X (horizontal) â†’ Model X (horizontal)
            # GroundLink Z (vertical) â†’ Model Y (vertical)  [CRITICAL SWAP]
            # GroundLink Y (horizontal) â†’ Model Z (horizontal)

            canonical_name = MARKER_NAME_MAP.get(std_name, std_name)
            data_dict[f"{canonical_name}_X"] = x       # X â†’ X (horizontal)
            data_dict[f"{canonical_name}_Y"] = z       # Z (vertical in GL) â†’ Y (vertical in model)
            data_dict[f"{canonical_name}_Z"] = y       # Y (horizontal in GL) â†’ Z (horizontal in model)
        else:
            # Missing marker handling: will be filled with 0s or NaNs by reindex later
            pass

    df = pd.DataFrame(data_dict)

    # Add Time column if needed, but the pipeline seems to handle Time separately or implicitly
    # The pipeline uses Frame# and Time in read_trc_file but mostly ignores them after
    # We return just the marker data.

    return df

def read_groundlink_force(file_path):
    """
    Reads a GroundLink Force NPY file and returns a DataFrame.
    The NPY file contains a dictionary with 'GRF' and 'CoP'.
    GRF shape: (Time, 2, 3) where 2 is Likely Left/Right.
    Scales the kN forces of the released files to Newtons (see GROUNDLINK_GRF_UNITS below).
    """
    # 1. Extract Subject ID
    # Pattern: Look for s followed by 3 digits (s001) in the file path
    match = re.search(r'(s\d{3})', file_path)
    subject_id = _normalize_subject_id(match.group(1)) if match else None
    if subject_id is None:
        basename_digits = re.search(r"(\d+)", os.path.basename(file_path))
        if basename_digits:
            subject_id = str(int(basename_digits.group(1)))

    # 2. Lookup Mass (fail-fast to avoid silent label-scale distortion)
    if not subject_id:
        raise ValueError(
            f"Could not extract subject ID from GroundLink force path: {file_path}. "
            "Expected pattern like 's001'."
        )

    if subject_id not in _GROUNDLINK_MASS_MAP:
        raise ValueError(
            f"Subject {subject_id} missing from GroundLink mass map. "
            "Refusing to scale GRF with fallback mass."
        )

    mass = _GROUNDLINK_MASS_MAP[subject_id]

    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        raise ValueError(f"Failed to read NPY file {file_path}: {e}")

    grf = data['GRF'] # (Time, 2, 3)
    cop = data['CoP'] # (Time, 2, 3)

    if isinstance(grf, torch.Tensor):
        grf = grf.numpy()
    if isinstance(cop, torch.Tensor):
        cop = cop.numpy()

    # 3. Scale GRF into Newtons.
    # The released GroundLink .npy forces are in KILONEWTONS (checked 2026-10-08: the median summed vertical
    # force in every subject's idling trial equals mass*g/1000, e.g. s003 0.483 kN for 53.07 kg, s006 0.482 kN
    # for 48.99 kg; a body-weight-normalized file would read 1.0 for everyone). The paper's tables are
    # "normalized by weight" because the authors divide at evaluation time, not in the files.
    # Modes via env GROUNDLINK_GRF_UNITS:
    # - "kilonewton" (default): multiply by 1000 -> N
    # - "newton": already in N, no scaling
    # - "normalized": LEGACY, WRONG for the released files. Treats values as body weights and multiplies by
    #   mass*g; combined with the pipeline's F/mg this leaves the raw kN value as the target, i.e. a hidden
    #   per-subject gain of mass*g/1000 (0.48-0.97). Every GroundLink run before 2026-10-08 used this mode.
    # - "auto": kilonewton (the old magnitude heuristic could not tell kN from body weights)
    grf_units_mode = os.getenv("GROUNDLINK_GRF_UNITS", "kilonewton").strip().lower()
    if grf_units_mode not in {"auto", "normalized", "newton", "kilonewton"}:
        grf_units_mode = "auto"

    finite_grf = grf[np.isfinite(grf)]
    grf_abs_max = float(np.max(np.abs(finite_grf))) if finite_grf.size else 0.0

    if grf_units_mode == "normalized":
        resolved_grf_units = "normalized->newton (LEGACY, mis-scaled)"
        grf = grf * mass * 9.81
    elif grf_units_mode == "newton":
        resolved_grf_units = "newton"
    else:
        resolved_grf_units = "kilonewton->newton"
        grf = grf * 1000.0

    global _PRINTED_GRF_UNIT_MSG
    if not _PRINTED_GRF_UNIT_MSG:
        print(
            f"GroundLink GRF units: mode={grf_units_mode}, resolved={resolved_grf_units}, "
            f"abs_max={grf_abs_max:.3f}"
        )
        _PRINTED_GRF_UNIT_MSG = True

    # Downsample if needed (assuming 250Hz -> 50Hz)
    # Ideally should match the C3D downsampling.
    # We assume the caller knows the rate or we hardcode 50Hz target.
    # GroundLink is 250Hz.
    step = 5

    grf = grf[::step]
    cop = cop[::step]

    df = pd.DataFrame()

    # GRF axes per GroundLink paper (SIGGRAPH Asia 2023, Figure 4):
    #   raw axis 0 = X = Anteroposterior (AP)
    #   raw axis 1 = Y = Mediolateral (ML)
    #   raw axis 2 = Z = Vertical
    df['ground_force_1_ap'] = grf[:, 0, 0]
    df['ground_force_1_ml'] = grf[:, 0, 1]
    df['ground_force_1_vert'] = grf[:, 0, 2]

    df['ground_force_2_ap'] = grf[:, 1, 0]
    df['ground_force_2_ml'] = grf[:, 1, 1]
    df['ground_force_2_vert'] = grf[:, 1, 2]

    # CoP is planar (2D) on the plate surface.
    # GL raw coordinate system: CoP axis 0 = AP (QTM X), CoP axis 1 = ML (QTM Y->model Z)
    #
    # IMPORTANT: CoP must stay in the RAW marker coordinate frame here
    # (axis 0 -> _px, axis 1 -> _pz) so that pelvis subtraction in
    # apply_cop_transform works correctly (pelvis_X and pelvis_Z are also
    # in this raw frame).  The heading alignment rotation in the pipeline
    # then naturally rotates both markers and CoP to the canonical frame
    df['ground_force_1_px'] = cop[:, 0, 0]   # raw AP (QTM X dir) -> _px
    df['ground_force_1_pz'] = cop[:, 0, 1]   # raw ML (QTM Y->Z dir) -> _pz

    df['ground_force_2_px'] = cop[:, 1, 0]   # raw AP (QTM X dir) -> _px
    df['ground_force_2_pz'] = cop[:, 1, 1]   # raw ML (QTM Y->Z dir) -> _pz

    return df
