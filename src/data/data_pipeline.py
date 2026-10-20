# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:27:02 2025

@author: David
"""

# data_pipeline.py
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import re
from typing import Optional, List

# Imports from our new/refactored modules
from src.config import ProcessingConfig, MARKER_NAME_MAP
from src.features import FeatureEngineer
from src.data.preprocessing import robust_normalize_data, subtract_pelvis_root, align_heading, scale_by_height, apply_cop_transform
from src.data.file_readers import read_trc_file, read_mot_file, read_sto_file, FileReadResult, TRCFileProcessingError, EssentialMarkerMissingError, TRCFileReadingError
from src.utils.utils import extract_subject_id_from_filepath
from src.data.groundlink_adapter import read_groundlink_c3d, read_groundlink_force

try:
    from src.data.polars_ops import fast_rename_columns, fast_column_select, cop_transform_polars, HAS_POLARS
except ImportError:
    HAS_POLARS = False


def _resolve_mass_kg(weight_value, weight_col_name=None):
    if weight_value is None:
        return None
    try:
        value = float(weight_value)
    except Exception:
        return None
    if not np.isfinite(value) or value <= 0:
        return None

    col_name = (weight_col_name or "").lower()
    if "weight/n" in col_name or col_name.endswith("/n") or "(n" in col_name:
        return value / 9.81

    # Heuristic fallback: values >200 are likely Newtons, not kg
    if value > 200:
        return value / 9.81
    return value


def _convert_force_targets_to_nbw(data_df_selected, data_columns_to_use, mass_kg):
    if mass_kg is None or not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("force_target_mode='nbw' requires a valid subject mass in kg.")

    bw_newtons = mass_kg * 9.81
    cols = list(data_df_selected.columns)

    force_suffixes = ("_ml", "_ap", "_vert", "_vx", "_vy", "_vz")
    if data_columns_to_use and all(isinstance(col, str) for col in data_columns_to_use):
        force_cols = [c for c in cols if c in data_columns_to_use and c.lower().endswith(force_suffixes)]
    else:
        force_cols = [c for c in cols if isinstance(c, str) and c.lower().endswith(force_suffixes)]

    if not force_cols:
        return data_df_selected

    data_df_selected = data_df_selected.copy()
    data_df_selected[force_cols] = data_df_selected[force_cols] / bw_newtons
    return data_df_selected

def read_data_file(file_path):
    """Reads a data file (.mot or .sto) into a Pandas DataFrame using readers from file_readers.py."""
    if file_path.endswith(".mot"):
        try:
            result = read_mot_file(file_path)
            if result.issues:
                print(f"Issues reading MOT file {file_path}: {result.issues}")
            return result.dataframe
        except Exception as e:
            print(f"Error reading MOT file {file_path}: {e}")
            return None
    elif file_path.endswith(".sto"):
        try:
            result = read_sto_file(file_path)
            if result.issues:
                print(f"Issues reading STO file {file_path}: {result.issues}")
            return result.dataframe
        except Exception as e:
            print(f"Error reading STO file {file_path}: {e}")
            return None
    elif file_path.endswith(".npy"): # Adapter for GroundLink Force data
        try:
            df = read_groundlink_force(file_path)
            return df
        except ValueError as e:
            print(f"Error reading GroundLink NPY file {file_path}: {e}")
            return None
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return None

def pad_or_truncate(sequence, max_length):
    seq_length = sequence.shape[0]
    if seq_length < max_length:
        pad_length = max_length - seq_length
        pad_array = np.zeros((pad_length, sequence.shape[1]))
        padded_sequence = np.concatenate([sequence, pad_array], axis=0)
        return padded_sequence, seq_length
    elif seq_length > max_length:
        truncated_sequence = sequence[:max_length]
        return truncated_sequence, max_length
    else:
        return sequence, seq_length


def apply_coordinate_corrections(data_sequence, data_columns_to_use=None):
    """
    Apply coordinate corrections to force/CoP data for Patient Handling (PH)
    trials to align with the motion capture system:

    1. Invert AP-axis forces and CoP (_ap, _pz) by multiplying by -1
    2. Zero shear forces (_ml, _ap) by subtracting mean of first 10 frames

    This addresses the "mirror image" problem where motion capture and force plate systems
    use opposite sign conventions for anteroposterior forces.

    NOTE: This is only applied when config.apply_coordinate_corrections=True (PH data).
    GroundLink data does not need these corrections.

    Args:
        data_sequence (np.ndarray): Force/CoP data, shape (seq_len, num_features)
        data_columns_to_use (list): Column names (strings) or None.
                                     If strings provided, identifies AP-axis and shear columns by name.
    
    Returns:
        np.ndarray: Corrected data sequence with same shape as input
    """
    if data_sequence is None or data_sequence.shape[0] == 0:
        return data_sequence
    
    corrected = data_sequence.copy()
    num_features = corrected.shape[1]
    
    # Identify AP-axis and shear force column indices
    ap_axis_indices = []
    shear_force_indices = []
    
    if data_columns_to_use is not None and all(isinstance(col, str) for col in data_columns_to_use):
        # Find indices by column name (semantic names after legacy rename)
        for idx, col_name in enumerate(data_columns_to_use):
            col_lower = col_name.lower()
            # AP-axis columns: force AP (_ap) and CoP in AP direction (_pz)
            if col_lower.endswith('_ap') or col_lower.endswith('_pz'):
                ap_axis_indices.append(idx)
            # Shear force columns: ML and AP force components (not CoP or vertical)
            if col_lower.endswith('_ml') or col_lower.endswith('_ap'):
                shear_force_indices.append(idx)
    else:
        # Fallback: assume standard 10-feature layout
        # [ground_force_1_ml, ground_force_1_vert, ground_force_1_ap, ground_force_1_px, ground_force_1_pz,
        #  ground_force_2_ml, ground_force_2_vert, ground_force_2_ap, ground_force_2_px, ground_force_2_pz]
        if num_features >= 10:
            ap_axis_indices = [2, 4, 7, 9]   # indices for _ap and _pz columns
            shear_force_indices = [0, 2, 5, 7]  # indices for _ml and _ap columns
    
    # Step 1: Invert AP-axis columns (force AP and CoP AP)
    for idx in ap_axis_indices:
        if idx < num_features:
            corrected[:, idx] = -corrected[:, idx]

    # Step 2: Zero shear forces by subtracting mean of first 10 frames
    num_frames_for_baseline = min(10, corrected.shape[0])
    for idx in shear_force_indices:
        if idx < num_features and num_frames_for_baseline > 0:
            baseline_mean = corrected[:num_frames_for_baseline, idx].mean()
            corrected[:, idx] = corrected[:, idx] - baseline_mean
    
    return corrected


def _causal_preproc() -> bool:
    """True when LOOKAHEAD is set to a number of frames (causal run): input preprocessing then uses frames <= t
    only (heading from a start-of-trial calibration window, backward-difference CoM derivatives).
    CAUSAL_PREPROC=1/0 overrides this, e.g. to run causal preprocessing with a non-causal model."""
    override = os.getenv("CAUSAL_PREPROC", "").strip()
    if override in ("0", "1"):
        return override == "1"
    return os.getenv("LOOKAHEAD", "full").strip().lower() not in ("", "full")


def rotate_force_horizontal(data_sequence, data_columns_to_use, heading_theta, is_patient_handling):
    """Co-rotate the horizontal GRF components (ML, AP) into the same
    heading-aligned canonical frame already applied to markers and CoP.

    Force is a vector and must transform covariantly with the coordinate frame.
    The heading rotation maps the subject's forward direction to +Z, so after
    rotation the canonical axes are X = ML, Z = AP.

    Raw model-frame axis assignment (set by the dataset readers):
      GroundLink       : model-X = AP (_ap), model-Z = ML (_ml)
      Patient Handling : model-X = ML (_ml), model-Z = AP (_ap)

    Rotation (same formula used for markers in align_heading / CoP transform):
      Fx' =  cos(theta) * Fx + sin(theta) * Fz   -> canonical ML
      Fz' = -sin(theta) * Fx + cos(theta) * Fz   -> canonical AP
    """
    if heading_theta is None or data_sequence is None or data_sequence.shape[0] == 0:
        return data_sequence
    if not (data_columns_to_use and all(isinstance(c, str) for c in data_columns_to_use)):
        return data_sequence  # need column names to locate ml/ap channels

    cols = list(data_columns_to_use)
    out = data_sequence.copy()
    cos_t, sin_t = float(np.cos(heading_theta)), float(np.sin(heading_theta))

    for plate in ("ground_force_1", "ground_force_2", "ground_force_s"):
        ml_name, ap_name = f"{plate}_ml", f"{plate}_ap"
        if ml_name in cols and ap_name in cols:
            i_ml, i_ap = cols.index(ml_name), cols.index(ap_name)
            ml = out[:, i_ml].copy()
            ap = out[:, i_ap].copy()
            # Put components onto model (X, Z) axes for this dataset
            if is_patient_handling:
                fx, fz = ml, ap   # model-X = ML, model-Z = AP
            else:
                fx, fz = ap, ml   # model-X = AP, model-Z = ML  (GroundLink)
            fx_p = cos_t * fx + sin_t * fz    # canonical ML
            fz_p = -sin_t * fx + cos_t * fz   # canonical AP
            out[:, i_ml] = fx_p
            out[:, i_ap] = fz_p
    return out


def _recover_excluded_subject(df, source_path):
    """RECOVER_EXCLUDED: repair the documented calibration errors that got the
    excluded subjects dropped, so they can be re-included for an
    exclusion-sensitivity analysis. Operates on the raw (Newton) canonical force
    DataFrame, before NBW/CoP transforms.

      - GL s001: per-channel DC baseline offset (un-zeroed plate). Subtract, per
        plate and channel, the mean over *unloaded* frames (plate vertical near
        zero) — a data-driven tare. Diagnosis: model R² 0.135 -> 0.73 after this.
      - PH subj13/subj14: force plates 1<->2 physically swapped. Swap columns.
        Diagnosis: model R² -1.6 -> 0.6-0.7 after this.
    """
    p = (source_path or "").lower()
    plates = ("ground_force_1", "ground_force_2")
    suffixes = ("ml", "ap", "vert", "px", "pz")
    df = df.copy()

    if "s001" in p:
        for plate in plates:
            vcol = f"{plate}_vert"
            if vcol not in df.columns:
                continue
            v = np.asarray(df[vcol], dtype=float)
            vmax = np.nanmax(np.abs(v)) if v.size else 0.0
            if not np.isfinite(vmax) or vmax <= 0:
                continue
            unloaded = np.abs(v) < 0.05 * vmax
            if unloaded.sum() < 5:  # fallback: 5 least-loaded frames
                unloaded = np.zeros_like(v, dtype=bool)
                unloaded[np.argsort(np.abs(v))[:5]] = True
            for suf in suffixes:
                c = f"{plate}_{suf}"
                if c in df.columns:
                    col = np.asarray(df[c], dtype=float)
                    off = np.nanmean(col[unloaded])
                    if np.isfinite(off):
                        df[c] = col - off
        print(f"  [RECOVER_EXCLUDED] GL s001 DC-offset baseline subtracted: {os.path.basename(p)}")

    elif "subj13" in p or "subj14" in p:
        for suf in suffixes:
            c1, c2 = f"ground_force_1_{suf}", f"ground_force_2_{suf}"
            if c1 in df.columns and c2 in df.columns:
                tmp = df[c1].copy()
                df[c1] = df[c2].to_numpy() if hasattr(df[c2], "to_numpy") else np.asarray(df[c2])
                df[c2] = tmp.to_numpy() if hasattr(tmp, "to_numpy") else np.asarray(tmp)
        print(f"  [RECOVER_EXCLUDED] PH plate 1<->2 swapped: {os.path.basename(p)}")

    return df


def process_trc_data(
    trc_filepath,
    canonical_marker_columns,
    essential_marker_short_names_list,
    nan_threshold=0.10,
    subject_height_m=None,
    enable_biomech_preprocessing=True,
):
    """
    Processes a TRC/C3D file: reads, biomechanical preprocessing, engineers
    features, normalizes.

    Returns
    -------
    normalized_df       : pd.DataFrame or None on failure.
    pelvis_origin       : pd.DataFrame (pelvis_X/Y/Z per frame, marker units) or None.
    heading_theta       : float (radians) or None.
    marker_units_to_mm  : float, converts marker (and pelvis_origin) values to mm.
    problems            : list of warning/error strings.
    """
    problems = []
    pelvis_origin = None
    heading_theta = None
    marker_units_to_mm = 1.0
    try:
        # Part 1: Read
        if trc_filepath.endswith(".c3d"):
            trc_df = read_groundlink_c3d(trc_filepath)   # millimetres (GROUNDLINK_MARKER_UNITS)
            trc_numeric = trc_df.apply(pd.to_numeric, errors='coerce')
            total_points = int(trc_numeric.size)
            nan_count = int(np.isnan(trc_numeric.values).sum())
            nan_percentage = (nan_count / total_points) if total_points > 0 else 0.0
            nan_stats = {
                'nan_count': nan_count,
                'total_points': total_points,
                'nan_percentage': nan_percentage,
            }
        else:
            result = read_trc_file(
                trc_filepath,
                essential_marker_short_names_list=essential_marker_short_names_list
            )
            trc_df = result.dataframe
            nan_stats = result.metadata
            problems.extend(result.issues)
            marker_units_to_mm = float(nan_stats.get('marker_units_to_mm', 1.0))

        if nan_stats.get('nan_percentage', 0.0) > nan_threshold:
            problems.append(f"NaN percentage ({nan_stats.get('nan_percentage', 0.0):.2f}) exceeded threshold.")
            return None, None, None, 1.0, problems

        if trc_df.empty and canonical_marker_columns:
            problems.append("TRC DataFrame is empty.")
            return None, None, None, 1.0, problems

        # Part 1b: Biomechanical Preprocessing (new)
        if enable_biomech_preprocessing:
            # Ablation switches (default on): BIOMECH_PELVIS / BIOMECH_HEADING / BIOMECH_HEIGHT = 0 turns
            # one step off while keeping the other two. With BIOMECH_PELVIS=0 the pelvis origin is still
            # computed (COM kinematics below need it) but markers are not made pelvis-relative. With
            # BIOMECH_HEADING=0 heading_theta stays None, so force co-rotation and COM rotation are skipped.
            # 1. Translational invariance: subtract pelvis origin per frame
            trc_rel, pelvis_origin = subtract_pelvis_root(trc_df)
            if os.getenv("BIOMECH_PELVIS", "1") != "0":
                trc_df = trc_rel
            # 2. Rotational invariance: align heading to +Z
            if os.getenv("BIOMECH_HEADING", "1") != "0":
                # causal runs (LOOKAHEAD set): heading from the first HEADING_CALIB_FRAMES frames (default 50 =
                # 1 s at 50 Hz) instead of the whole-trial mean
                trc_df, heading_theta = align_heading(
                    trc_df, n_frames=int(os.getenv("HEADING_CALIB_FRAMES", "50")) if _causal_preproc() else None)
            # 3. Anthropometric scaling: divide by subject height
            if os.getenv("BIOMECH_HEIGHT", "1") != "0" and subject_height_m is not None and subject_height_m > 0:
                trc_df = scale_by_height(trc_df, subject_height_m)

        # Part 2: Feature Engineering
        engineer = FeatureEngineer(canonical_marker_columns=canonical_marker_columns)
        trc_df_features = engineer.process_features(trc_df)

        # Part 2b: Append CoM kinematics AFTER feature engineering so FeatureEngineer
        #          doesn't re-compute vel/acc of vel/acc cols, and canonical reindex
        #          doesn't strip them (they are appended after reindex).
        if enable_biomech_preprocessing and pelvis_origin is not None:
            sampling_rate = 50.0  # Hz
            td = 1.0 / sampling_rate
            com_cols = {}
            for ax in ('X', 'Y', 'Z'):
                src_col = f'pelvis_{ax}'
                if src_col in pelvis_origin.columns:
                    pos_vals = pelvis_origin[src_col].values
                    if _causal_preproc():
                        # backward differences (frames <= t), as the marker vel/acc in FeatureEngineer
                        vel = np.diff(pos_vals, prepend=pos_vals[:1]) / td
                        acc = np.diff(vel, prepend=vel[:1]) / td
                    else:
                        vel = np.gradient(pos_vals, td)
                        acc = np.gradient(vel, td)
                    com_cols[f'COM_vel_{ax}'] = vel.astype(np.float32)
                    com_cols[f'COM_acc_{ax}'] = acc.astype(np.float32)
            if com_cols:
                # Apply heading rotation to XZ components so COM kinematics are in
                # the same heading-aligned frame as the marker features.
                if heading_theta is not None:
                    cos_t = float(np.cos(heading_theta))
                    sin_t = float(np.sin(heading_theta))
                    for prefix in ('COM_vel', 'COM_acc'):
                        kx, kz = f'{prefix}_X', f'{prefix}_Z'
                        if kx in com_cols and kz in com_cols:
                            ox, oz = com_cols[kx].copy(), com_cols[kz].copy()
                            com_cols[kx] = cos_t * ox + sin_t * oz
                            com_cols[kz] = -sin_t * ox + cos_t * oz
                # Apply height scaling so units are dimensionless (vel/height, acc/height)
                if subject_height_m is not None and subject_height_m > 0:
                    for key in list(com_cols.keys()):
                        com_cols[key] = com_cols[key] / float(subject_height_m)
                n = len(trc_df_features)
                com_df = pd.DataFrame(
                    {k: v[:n] for k, v in com_cols.items()},
                    index=trc_df_features.index,
                )
                trc_df_features = pd.concat([trc_df_features, com_df], axis=1)

        # Part 3: Normalization
        # Default: per-trial z-score. Ablation flag INPUT_NORM_MODE=per_fold skips
        # per-trial normalization here so a fold-level normalizer (fit on training
        # frames only) can be applied downstream, preserving cross-trial magnitude.
        if os.getenv("INPUT_NORM_MODE", "per_trial").lower() == "per_fold":
            trc_data_normalized = trc_df_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            trc_data_normalized = robust_normalize_data(trc_df_features, type='standard')

        return trc_data_normalized, pelvis_origin, heading_theta, marker_units_to_mm, problems

    except Exception as e:
        problems.append(f"Error in process_trc_data: {e}")
        return None, None, None, 1.0, problems


def load_and_process_trial_data(config: ProcessingConfig):

    trial_trc_tensor = None
    trial_metadata_tensor = None
    trial_target_tensor = None
    trial_input_mask_tensor = None
    problem_details_for_trial = []
    subject_mass_kg = None
    subject_height_m = None

    try:
        def _format_array_stats(name, arr):
            arr = np.asarray(arr)
            finite = arr[np.isfinite(arr)]
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            if finite.size == 0:
                return f"{name}: shape={arr.shape}, finite=0, nan={nan_count}, inf={inf_count}"
            return (
                f"{name}: shape={arr.shape}, min={float(np.min(finite)):.4f}, "
                f"max={float(np.max(finite)):.4f}, mean={float(np.mean(finite)):.4f}, "
                f"std={float(np.std(finite)):.4f}, nan={nan_count}, inf={inf_count}"
            )

        # Part 4 (early): Pre-extract metadata so height is available for biomech preprocessing
        if config.use_metadata:
            try:
                if config.metadata_df is None:
                    raise ValueError("Metadata processing requested but metadata_df is None.")

                subject_id_for_lookup = extract_subject_id_from_filepath(config.trc_filepath, config.trc_file_prefix, config.trc_file_suffix)
                meta_rows = config.metadata_df.loc[subject_id_for_lookup]

                if isinstance(meta_rows, pd.DataFrame):
                    problem_details_for_trial.append(f"Warning: Duplicate subject ID '{subject_id_for_lookup}' found, using first entry.")
                    meta_row = meta_rows.iloc[0]
                else:
                    meta_row = meta_rows

                weight = float(meta_row[config.weight_col_in_metadata])
                height = float(meta_row[config.height_col_in_metadata])

                if np.isnan(weight) or np.isnan(height):
                    raise ValueError(f"Weight or height is NaN for subject {subject_id_for_lookup}")

                subject_mass_kg = _resolve_mass_kg(weight, config.weight_col_in_metadata)
                if config.force_target_mode == "nbw" and (subject_mass_kg is None or subject_mass_kg <= 0):
                    raise ValueError(
                        f"Could not resolve subject mass in kg for NBW target conversion: subject={subject_id_for_lookup}, "
                        f"raw_weight={weight}, weight_col={config.weight_col_in_metadata}"
                    )

                subject_height_m = height if height > 0 else None
                trial_metadata_tensor = torch.tensor([weight, height], dtype=torch.float32)
            except KeyError:
                problem_details_for_trial.append(f"Metadata lookup error: Subject ID '{subject_id_for_lookup}' not found.")
                return None, None, None, None, problem_details_for_trial
            except Exception as e:
                problem_details_for_trial.append(f"Unexpected error during Metadata Handling for {config.trc_filepath}: {e}")
                return None, None, None, None, problem_details_for_trial

        # Part 2/3: Read, biomech-preprocess, feature-engineer, normalize
        trc_data_normalized, pelvis_origin, heading_theta, marker_units_to_mm, trc_problems = process_trc_data(
            config.trc_filepath,
            config.canonical_marker_columns,
            config.essential_marker_short_names_list,
            config.nan_threshold,
            subject_height_m=subject_height_m,
            enable_biomech_preprocessing=config.enable_biomech_preprocessing,
        )
        problem_details_for_trial.extend(trc_problems)

        if trc_data_normalized is None:
            return None, None, None, None, problem_details_for_trial

        # Part 5: Target Data Processing
        try:
            data_df = read_data_file(config.data_filepath)
            if data_df is None or data_df.empty:
                raise ValueError(f"Target data file {config.data_filepath} is empty or failed to load.")

            # Rename legacy MOT force columns (vx/vy/vz) to semantic names (ml/vert/ap) whenever they are present
            # (OpenSim MOT files: Patient Handling, wind turbine, AddBiomechanics; GroundLink's reader already emits
            # semantic names). Convention: vx=ML, vy=Vertical, vz=AP. Until 2026-10-06 this rename was tied to
            # apply_coordinate_corrections, so the sign corrections could not be switched off for a MOT dataset (B5).
            if True:
                legacy_rename = {
                    'ground_force_1_vx': 'ground_force_1_ml',
                    'ground_force_1_vy': 'ground_force_1_vert',
                    'ground_force_1_vz': 'ground_force_1_ap',
                    'ground_force_2_vx': 'ground_force_2_ml',
                    'ground_force_2_vy': 'ground_force_2_vert',
                    'ground_force_2_vz': 'ground_force_2_ap',
                }
                cols_to_rename = {k: v for k, v in legacy_rename.items() if k in data_df.columns and v not in data_df.columns}
                if cols_to_rename:
                    if HAS_POLARS:
                        data_df = fast_rename_columns(data_df, cols_to_rename)
                    else:
                        data_df = data_df.rename(columns=cols_to_rename)

            # TARGET_LAYOUT=plate_sum (Chapter 2, A0): total ground reaction force as the sum of the two
            # plates and a combined centre of pressure weighted by each plate's vertical force, named
            # ground_force_s_{ml,vert,ap,px,pz}. Built on the raw (Newton) per-plate columns, so the
            # body-weight conversion, coordinate corrections, CoP transform and co-rotation below apply
            # to the sum exactly as they do to a plate. Affine CoP transforms commute with the weighted
            # average because the weights sum to one.
            if os.getenv("TARGET_LAYOUT", "per_plate").strip().lower() == "plate_sum":
                if hasattr(data_df, "to_pandas"):
                    data_df = data_df.to_pandas()
                need = [f"ground_force_{p}_{s}" for p in ("1", "2") for s in ("ml", "vert", "ap", "px", "pz")]
                if all(c in data_df.columns for c in need):
                    v1 = data_df["ground_force_1_vert"].to_numpy(dtype=float)
                    v2 = data_df["ground_force_2_vert"].to_numpy(dtype=float)
                    w1 = np.clip(v1, 0.0, None); w2 = np.clip(v2, 0.0, None); wsum = w1 + w2
                    with np.errstate(invalid="ignore", divide="ignore"):
                        a1 = np.where(wsum > 1e-6, w1 / np.where(wsum > 1e-6, wsum, 1.0), 0.5)
                    a2 = 1.0 - a1
                    for s in ("ml", "vert", "ap"):
                        data_df[f"ground_force_s_{s}"] = data_df[f"ground_force_1_{s}"].to_numpy(dtype=float) + data_df[f"ground_force_2_{s}"].to_numpy(dtype=float)
                    for s in ("px", "pz"):
                        data_df[f"ground_force_s_{s}"] = a1 * data_df[f"ground_force_1_{s}"].to_numpy(dtype=float) + a2 * data_df[f"ground_force_2_{s}"].to_numpy(dtype=float)
                else:
                    problem_details_for_trial.append("TARGET_LAYOUT=plate_sum: per-plate columns missing, sum not built.")

            data_df_selected = data_df
            if config.data_columns_to_use:
                if all(isinstance(col, int) for col in config.data_columns_to_use):
                    data_df_selected = data_df.iloc[:, config.data_columns_to_use].copy()
                elif all(isinstance(col, str) for col in config.data_columns_to_use):
                    expected_cols = list(config.data_columns_to_use)
                    data_cols = list(data_df.columns)
                    missing_cols = [cn for cn in expected_cols if cn not in data_cols]
                    extra_cols = [cn for cn in data_cols if cn not in expected_cols]

                    if config.strict_target_columns and (missing_cols or extra_cols):
                        raise ValueError(
                            f"Target column contract mismatch for {config.data_filepath}. "
                            f"Missing expected: {missing_cols if missing_cols else 'None'}. "
                            f"Unexpected present: {extra_cols if extra_cols else 'None'}."
                        )

                    actual_cols_to_use = [cn for cn in expected_cols if cn in data_df.columns]
                    if not actual_cols_to_use:
                        raise ValueError(f"None of the specified target columns found in {config.data_filepath}.")

                    if not config.strict_target_columns and missing_cols:
                        problem_details_for_trial.append(
                            f"Warning: Missing target columns in {config.data_filepath}: {missing_cols}"
                        )

                    data_df_selected = data_df[actual_cols_to_use].copy()
                else:
                    raise TypeError("data_columns_to_use must be a list of strings or integers.")

            # RECOVER_EXCLUDED: repair documented calibration errors on the
            # excluded subjects (GL s001 DC offset; PH subj13/14 plate swap) on
            # the raw force targets, before NBW/CoP transforms.
            if os.getenv("RECOVER_EXCLUDED", "0") == "1":
                data_df_selected = _recover_excluded_subject(
                    data_df_selected, config.data_filepath or config.trc_filepath
                )

            if config.force_target_mode == "nbw":
                if subject_mass_kg is None:
                    raise ValueError(
                        "force_target_mode='nbw' requires metadata-based subject mass. "
                        f"Missing for file {config.trc_filepath}."
                    )
                data_df_selected = _convert_force_targets_to_nbw(
                    data_df_selected,
                    config.data_columns_to_use,
                    subject_mass_kg,
                )

            # Part 5b: CoP transform — apply same pelvis shift, heading rotation, and
            # height scaling that was applied to marker inputs so the model always sees
            # body-relative, heading-aligned, anthropometrically-normalised CoP targets.
            if config.enable_biomech_preprocessing and (pelvis_origin is not None or heading_theta is not None):
                # The CoP transform works in mm (CoP m -> mm, minus pelvis). Markers stay in their file
                # units (wind turbine: metres; COM kinematics and the physics prior rely on that), so
                # convert only this copy of the pelvis track.
                if pelvis_origin is not None and marker_units_to_mm != 1.0:
                    pelvis_origin = pelvis_origin * marker_units_to_mm
                cop_col_pairs = [
                    ('ground_force_1_px', 'ground_force_1_pz'),
                    ('ground_force_2_px', 'ground_force_2_pz'),
                    ('ground_force_s_px', 'ground_force_s_pz'),
                ]
                for px_col, pz_col in cop_col_pairs:
                    if px_col in data_df_selected.columns and pz_col in data_df_selected.columns:
                        n_frames = len(data_df_selected)
                        _cop_x_raw = data_df_selected[px_col].values[:n_frames]
                        _cop_z_raw = data_df_selected[pz_col].values[:n_frames]
                        _pel_x = pelvis_origin['pelvis_X'].values[:n_frames] if (pelvis_origin is not None and 'pelvis_X' in pelvis_origin) else None
                        _pel_z = pelvis_origin['pelvis_Z'].values[:n_frames] if (pelvis_origin is not None and 'pelvis_Z' in pelvis_origin) else None
                        try:
                            if HAS_POLARS:
                                cop_x, cop_z = cop_transform_polars(
                                    _cop_x_raw, _cop_z_raw, _pel_x, _pel_z,
                                    heading_theta, subject_height_m,
                                )
                            else:
                                raise RuntimeError("polars fallback")
                        except Exception:
                            cop_x, cop_z = apply_cop_transform(
                                _cop_x_raw, _cop_z_raw,
                                pelvis_origin, heading_theta, subject_height_m,
                            )
                        data_df_selected = data_df_selected.copy()
                        data_df_selected[px_col] = cop_x
                        data_df_selected[pz_col] = cop_z

            data_array = data_df_selected.values
            if data_array.ndim == 1: data_array = data_array.reshape(-1, 1)
        except Exception as e:
            problem_details_for_trial.append(f"Unexpected error during Target Data Processing for {config.data_filepath}: {e}")
            return None, None, None, None, problem_details_for_trial

        # Part 6: Final Alignment, Padding, and Tensor Creation
        try:
            trc_norm_vals = trc_data_normalized.values
            if trc_norm_vals.ndim == 1: trc_norm_vals = trc_norm_vals.reshape(-1, 1)

            if trc_norm_vals.shape[1] == 0 or data_array.shape[1] == 0:
                raise ValueError("Zero features in TRC or target data after processing.")

            min_len = min(len(trc_norm_vals), len(data_array))
            if min_len == 0:
                raise ValueError("Zero length after aligning TRC and target data.")

            trc_sequence_aligned = trc_norm_vals[:min_len]
            data_sequence_aligned = data_array[:min_len]

            if config.enable_data_diagnostics:
                print(f"DATA_DIAG [{config.split_name}] Trial: {os.path.basename(config.trc_filepath)}")
                print("  " + _format_array_stats("inputs_aligned", trc_sequence_aligned))
                print("  " + _format_array_stats("targets_aligned", data_sequence_aligned))
            
            # Apply coordinate corrections: invert Z-axis and zero shear forces
            if config.apply_coordinate_corrections:
                data_sequence_aligned = apply_coordinate_corrections(
                    data_sequence_aligned,
                    data_columns_to_use=config.data_columns_to_use
                )

            # Ablation flag ROTATE_FORCE_VECTORS=1: co-rotate horizontal GRF (ML/AP)
            # into the heading-aligned frame so force targets share the kinematic frame.
            if os.getenv("ROTATE_FORCE_VECTORS", "0") == "1" and config.enable_biomech_preprocessing:
                data_sequence_aligned = rotate_force_horizontal(
                    data_sequence_aligned,
                    config.data_columns_to_use,
                    heading_theta,
                    # model-X = ML label for Patient Handling / wind turbine, and for datasets whose markers and forces
                    # already share one world frame (AddBiomechanics: FORCE_MODEL_X_IS_ML=1); GroundLink's reader
                    # swaps axes so its model-X is AP.
                    is_patient_handling=config.apply_coordinate_corrections or os.getenv("FORCE_MODEL_X_IS_ML", "0") == "1",
                )

            trc_sequence_padded, trc_original_len = pad_or_truncate(trc_sequence_aligned, config.max_length)
            data_sequence_padded, _ = pad_or_truncate(data_sequence_aligned, config.max_length)

            input_mask_np = np.zeros(config.max_length, dtype=np.float32)
            input_mask_np[:trc_original_len] = 1.0

            trial_trc_tensor = torch.tensor(trc_sequence_padded, dtype=torch.float32)
            trial_target_tensor = torch.tensor(data_sequence_padded, dtype=torch.float32)
            trial_input_mask_tensor = torch.tensor(input_mask_np, dtype=torch.float32)

            if config.enable_data_diagnostics:
                masked_out_ratio = float(1.0 - (trc_original_len / max(config.max_length, 1)))
                print(
                    f"  mask_stats: valid_frames={trc_original_len}, "
                    f"max_len={config.max_length}, masked_out_ratio={masked_out_ratio:.4f}"
                )
        except Exception as e:
            problem_details_for_trial.append(f"Unexpected error during final tensor creation for {config.trc_filepath}: {e}")
            return None, None, None, None, problem_details_for_trial

        return trial_trc_tensor, trial_metadata_tensor, trial_target_tensor, trial_input_mask_tensor, problem_details_for_trial

    except Exception as e:
        problem_details_for_trial.append(f"An overarching unexpected error occurred in load_and_process_trial_data for {config.trc_filepath}: {e}")
        return None, None, None, None, problem_details_for_trial


# Paths of the trials that survived loading, per split, in tensor order, from the most recent
# prepare_dataset_for_modeling call. One entry per source trial: the Train list is not doubled
# by the bilateral flip. Read by experiments that save per-trial predictions (chapter1/experiments).
LAST_KEPT_PATHS: dict = {}


def prepare_dataset_for_modeling(
    trc_file_paths,
    data_file_paths,
    data_columns_to_use,
    essential_marker_names,
    use_metadata: bool,
    selected_dataset: str,
    metadata_df=None,
    trc_file_prefix="",
    trc_file_suffix=".trc",
    subject_id_col_in_metadata='Subject',
    weight_col_in_metadata='Weight',
    height_col_in_metadata='Height',
    nan_threshold=0.10,
    test_size=0.1,
    val_size=0.1,
    random_state=42,
    apply_force_corrections: bool = True,
    global_max_len_override: int = None,
    strict_target_columns: bool = False,
    enable_data_diagnostics: bool = False,
    force_target_mode: str = "newton",
    # --- Generalisation flags ---
    held_out_subject: Optional[str] = None,   # LOSO: hold out this subject ID for test
    enable_biomech_preprocessing: bool = True, # pelvis root, heading align, height scale
    use_bilateral_flip: bool = True,           # doubles training data via L/R mirroring
    ):

    if use_metadata and metadata_df is None:
        raise ValueError("Metadata processing requested (use_metadata=True), but metadata_df is None.")
    if len(trc_file_paths) != len(data_file_paths):
        raise ValueError("TRC and DATA file lists length mismatch.")
    if not essential_marker_names:
        raise ValueError("`essential_marker_names` must be provided.")

    # Define canonical_marker_columns based on essential_marker_names
    temp_canonical_cols = []
    for short_name in essential_marker_names:
        full_name = MARKER_NAME_MAP.get(short_name, short_name) # MARKER_NAME_MAP from config
        temp_canonical_cols.extend([f"{full_name}_X", f"{full_name}_Y", f"{full_name}_Z"])
    if not temp_canonical_cols:
        raise ValueError("No canonical marker columns could be derived from `essential_marker_names`.")

    base_canonical_marker_columns = sorted(temp_canonical_cols)

    # Add velocity and acceleration features
    vel_acc_canonical_cols = []
    for col in base_canonical_marker_columns:
        vel_acc_canonical_cols.append(col)
        vel_acc_canonical_cols.append(f"{col}_vel")
        vel_acc_canonical_cols.append(f"{col}_acc")

    # Add relative position features
    rel_pos_canonical_cols = []
    marker_pairs = [
        ('LSHO', 'LELBL'), ('LELBL', 'LWRL'),
        ('RSHO', 'RELBL'), ('RELBL', 'RWRL'),
        ('LKNE', 'LANKL'), ('LANKL', 'LTOE'),
        ('RKNE', 'RANKM'), ('RANKM', 'RTOE'),
        ('LSHO', 'RSHO'), ('LASI', 'LSHO')
    ]
    for marker1_short, marker2_short in marker_pairs:
        for axis in ['X', 'Y', 'Z']:
            rel_pos_canonical_cols.append(f"rel_{marker1_short}_{marker2_short}_{axis}")

    canonical_marker_columns = vel_acc_canonical_cols + rel_pos_canonical_cols

    # When biomechanical preprocessing is active, 6 CoM kinematics columns are appended
    # after feature engineering (outside the canonical reindex) so we must track them here
    # for input_feature_names accuracy.
    com_kinematic_cols = (
        ['COM_vel_X', 'COM_acc_X', 'COM_vel_Y', 'COM_acc_Y', 'COM_vel_Z', 'COM_acc_Z']  # tensor order (com_cols dict: vel then acc per axis)
        if enable_biomech_preprocessing else []
    )
    input_feature_names = canonical_marker_columns + com_kinematic_cols

    # --- Subject ID extraction (used for LOSO splitting) ---
    all_subject_ids = [
        extract_subject_id_from_filepath(p, trc_file_prefix, trc_file_suffix)
        for p in trc_file_paths
    ]

    # --- File Path Splitting ---
    indices = list(range(len(trc_file_paths)))
    train_indices, val_indices, test_indices = [], [], []

    if held_out_subject is not None:
        # LOSO split: held_out_subject goes entirely into test; rest split into train/val
        test_indices   = [i for i, sid in enumerate(all_subject_ids) if sid == held_out_subject]
        remain_indices = [i for i, sid in enumerate(all_subject_ids) if sid != held_out_subject]
        if not test_indices:
            print(f"Warning: held_out_subject={held_out_subject!r} not found in file list. Falling back to random split.")
        else:
            print(f"LOSO: held-out subject={held_out_subject!r} → {len(test_indices)} test trials, {len(remain_indices)} train+val trials.")

        if len(remain_indices) < 2:
            train_indices = remain_indices
            val_indices = []
        else:
            actual_val_size = val_size / max(1.0 - test_size, 1e-9)
            if len(remain_indices) * actual_val_size < 1:
                train_indices = remain_indices
                val_indices = []
            else:
                # Group-aware val split so no subject leaks between train and val
                remain_sids = [all_subject_ids[i] for i in remain_indices]
                unique_remain_sids = list(dict.fromkeys(remain_sids))  # preserve order
                n_val_subj = max(1, round(len(unique_remain_sids) * actual_val_size))
                rng = np.random.default_rng(random_state)
                val_subj_set = set(rng.choice(unique_remain_sids, size=n_val_subj, replace=False).tolist())
                train_indices = [i for i in remain_indices if all_subject_ids[i] not in val_subj_set]
                val_indices   = [i for i in remain_indices if all_subject_ids[i] in val_subj_set]
    else:
        # Default: random trial-level split (backward compatible)
        min_samples_for_split = 3
        if len(indices) < min_samples_for_split:
            print(f"Warning: Only {len(indices)} samples available. Not enough to perform train/val/test split. Using all for training.")
            train_indices = indices
        else:
            train_val_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)
            if not train_val_indices:
                train_indices = []
                val_indices = []
            else:
                actual_val_size_for_split = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0
                if len(train_val_indices) * actual_val_size_for_split < 1:
                    train_indices = train_val_indices
                    val_indices = []
                else:
                    train_indices, val_indices = train_test_split(train_val_indices, test_size=actual_val_size_for_split, random_state=random_state, shuffle=True)

    path_splits = {
        "Train": ([trc_file_paths[i] for i in train_indices], [data_file_paths[i] for i in train_indices]),
        "Validation": ([trc_file_paths[i] for i in val_indices], [data_file_paths[i] for i in val_indices]),
        "Test": ([trc_file_paths[i] for i in test_indices], [data_file_paths[i] for i in test_indices])
    }

    # --- Determine Global Max Length for Padding ---
    if global_max_len_override is not None:
        global_max_len = int(global_max_len_override)
        print(f"INFO: Using provided global_max_len override: {global_max_len}")
    else:
        global_max_len = 0
        print("Calculating global_max_len for padding...")
        for i in range(len(trc_file_paths)):
            try:
                if trc_file_paths[i].endswith(".c3d"):
                    df_trc_temp = read_groundlink_c3d(trc_file_paths[i])
                else:
                    result = read_trc_file(trc_file_paths[i], essential_marker_short_names_list=essential_marker_names)
                    df_trc_temp = result.dataframe

                if df_trc_temp is None or df_trc_temp.empty:
                    continue

                df_data_temp = read_data_file(data_file_paths[i])
                if df_data_temp is None or df_data_temp.empty:
                    continue

                current_min_len = min(len(df_trc_temp), len(df_data_temp))
                global_max_len = max(global_max_len, current_min_len)
            except (TRCFileProcessingError, EssentialMarkerMissingError, TRCFileReadingError) as e:
                continue
            except Exception as e:
                continue


        if global_max_len == 0:
            print("Warning: global_max_len is 0 after checking all files. Setting to fallback 651.")
            global_max_len = 651
        print(f"INFO: Determined global_max_len for padding: {global_max_len}")

    # --- Process each data split ---
    all_split_data = {}
    all_discarded_file_reports = []

    for split_name, (current_trc_paths, current_data_paths) in path_splits.items():
        split_trc_seqs, split_meta_seqs, split_data_seqs, split_input_masks = [], [], [], []
        kept_paths = []
        LAST_KEPT_PATHS[split_name] = kept_paths

        if not current_trc_paths:
            all_split_data[split_name] = ([], [], [], []) if use_metadata else ([], [], [])
            continue

        for trc_fp, data_fp in zip(current_trc_paths, current_data_paths):

            # Create ProcessingConfig for this trial
            config = ProcessingConfig(
                trc_filepath=trc_fp,
                data_filepath=data_fp,
                use_metadata=use_metadata,
                metadata_df=metadata_df,
                max_length=global_max_len,
                data_columns_to_use=data_columns_to_use,
                split_name=split_name,
                trc_file_prefix=trc_file_prefix,
                trc_file_suffix=trc_file_suffix,
                subject_id_col_in_metadata=subject_id_col_in_metadata,
                weight_col_in_metadata=weight_col_in_metadata,
                height_col_in_metadata=height_col_in_metadata,
                canonical_marker_columns=canonical_marker_columns,
                essential_marker_short_names_list=essential_marker_names,
                nan_threshold=nan_threshold,
                apply_coordinate_corrections=apply_force_corrections,
                strict_target_columns=strict_target_columns,
                enable_data_diagnostics=enable_data_diagnostics,
                force_target_mode=force_target_mode,
                enable_biomech_preprocessing=enable_biomech_preprocessing,
            )

            trc_tensor, meta_tensor, target_tensor, mask_tensor, problems = load_and_process_trial_data(config)

            if trc_tensor is not None:
                kept_paths.append(trc_fp)
                split_trc_seqs.append(trc_tensor)
                if use_metadata:
                    split_meta_seqs.append(meta_tensor)
                split_data_seqs.append(target_tensor)
                split_input_masks.append(mask_tensor)

                # Bilateral flip augmentation: only on training split, doubles dataset
                if use_bilateral_flip and split_name == "Train":
                    from src.data.augmentation import bilateral_flip
                    # input_feature_names = all tensor columns incl. the 6 CoM columns (canonical_marker_columns
                    # lacks them, which left COM_vel_X / COM_acc_X un-mirrored before 2026-10-02)
                    flip_result = bilateral_flip(
                        trc_tensor, target_tensor, mask_tensor,
                        input_feature_names, data_columns_to_use
                    )
                    if flip_result is not None:
                        flip_trc, flip_target, flip_mask = flip_result
                        split_trc_seqs.append(flip_trc)
                        if use_metadata:
                            split_meta_seqs.append(meta_tensor)  # same subject metadata
                        split_data_seqs.append(flip_target)
                        split_input_masks.append(flip_mask)

            if problems:
                all_discarded_file_reports.append({'filepath': trc_fp, 'reasons': problems, 'split': split_name})

        if use_metadata:
            all_split_data[split_name] = (split_trc_seqs, split_meta_seqs, split_data_seqs, split_input_masks)
        else:
            all_split_data[split_name] = (split_trc_seqs, split_data_seqs, split_input_masks)

    # --- Print Summary of Discarded Files ---
    if all_discarded_file_reports:
        print("\n--- Summary of Discarded/Problematic Files During Data Loading ---")
        for report in all_discarded_file_reports:
            print(f"  - File: {report['filepath']} (Split: {report['split']})")
            for reason in report['reasons']:
                print(f"    - Reason: {reason}")
    else:
        print("\nNo files were discarded during data loading based on NaN threshold or critical errors.")

    # --- Prepare final return tuples ---
    train_tuple = all_split_data["Train"]
    val_tuple = all_split_data["Validation"]
    test_tuple = all_split_data["Test"]

    first_trc_seqs_list = train_tuple[0] if train_tuple[0] else (val_tuple[0] if val_tuple[0] else (test_tuple[0] if test_tuple[0] else []))

    target_idx_in_tuple = 2 if use_metadata else 1
    first_data_seqs_list = train_tuple[target_idx_in_tuple] if train_tuple[target_idx_in_tuple] else \
                           (val_tuple[target_idx_in_tuple] if val_tuple[target_idx_in_tuple] else \
                           (test_tuple[target_idx_in_tuple] if test_tuple[target_idx_in_tuple] else []))

    input_size = first_trc_seqs_list[0].shape[1] if first_trc_seqs_list and len(first_trc_seqs_list[0].shape) > 1 else 0
    output_size = first_data_seqs_list[0].shape[1] if first_data_seqs_list and len(first_data_seqs_list[0].shape) > 1 else 0

    if not first_trc_seqs_list:
         print("CRITICAL WARNING: All data splits (train, validation, test) resulted in empty TRC sequence lists. input_size will be 0.")
    if not first_data_seqs_list:
         print("CRITICAL WARNING: All data splits (train, validation, test) resulted in empty target data sequence lists. output_size will be 0.")


    return train_tuple, val_tuple, test_tuple, global_max_len, input_size, output_size, input_feature_names
