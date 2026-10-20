import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional, Tuple

from src.config import MARKER_NAME_MAP


# ---------------------------------------------------------------------------
# Biomechanical Coordinate-System Transforms
# ---------------------------------------------------------------------------

def _get_marker_cols(df: pd.DataFrame, short_names, axes=('X', 'Y', 'Z')):
    """Return {short_name: {axis: col_name}} for markers that exist in df."""
    available = {}
    for short in short_names:
        full = MARKER_NAME_MAP.get(short, short)
        ax_map = {}
        for ax in axes:
            col = f"{full}_{ax}"
            if col in df.columns:
                ax_map[ax] = col
        if ax_map:
            available[short] = ax_map
    return available


def subtract_pelvis_root(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Translate all marker positions to be relative to the pelvis centre each frame.

    The pelvis origin is the per-frame mean of available pelvis markers
    (LASI, RASI, LPSI, RPSI). Falls back to LASI only when others aren't
    present (e.g. Patient Handling data).

    Returns
    -------
    df_out : DataFrame with pelvis-relative marker coordinates.
    pelvis_origin : DataFrame with columns pelvis_X, pelvis_Y, pelvis_Z
                    (one row per frame), or None if no pelvis markers found.
    """
    preferred = ['LASI', 'RASI', 'LPSI', 'RPSI']
    fallback  = ['LASI']

    for candidates in [preferred, fallback]:
        found = _get_marker_cols(df, candidates, axes=('X', 'Y', 'Z'))
        if found:
            break
    else:
        return df, None

    pelvis_xyz = {}
    for ax in ('X', 'Y', 'Z'):
        ax_cols = [m[ax] for m in found.values() if ax in m]
        if ax_cols:
            pelvis_xyz[f'pelvis_{ax}'] = df[ax_cols].mean(axis=1)

    if not pelvis_xyz:
        return df, None

    pelvis_origin = pd.DataFrame(pelvis_xyz, index=df.index)

    # Subtract from ALL raw marker position columns (skip vel/acc/rel columns)
    df_out = df.copy()
    for ax in ('X', 'Y', 'Z'):
        if f'pelvis_{ax}' not in pelvis_xyz:
            continue
        origin_series = pelvis_xyz[f'pelvis_{ax}']
        raw_pos_cols = [
            c for c in df_out.columns
            if c.endswith(f'_{ax}')
            and not c.endswith(('_vel', '_acc'))
            and not c.startswith('rel_')
        ]
        for col in raw_pos_cols:
            df_out[col] = df_out[col] - origin_series

    return df_out, pelvis_origin


def align_heading(
    df: pd.DataFrame,
    n_frames: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Rotate all marker XZ coordinates so the subject's mean forward direction
    aligns with the +Z axis (Y-up convention).

    n_frames: estimate the heading from the first n_frames only (causal / online use: a short calibration
    window at the start instead of the whole-trial mean, which uses future frames). None = whole trial.

    Forward direction priority:
      1. LPSI/RPSI → LASI/RASI midpoint direction (full pelvis frame)
      2. LASI → RASI perpendicular (only LASI+RASI)
      3. LSHO → RSHO perpendicular (shoulder line)
      4. No rotation (returns original df, theta=None)

    Returns
    -------
    df_out   : DataFrame with rotated XZ coordinates.
    theta    : Rotation angle (radians) applied, or None if skipped.
               Positive == counter-clockwise when viewed from above (+Y).
    """
    # Attempt to compute a forward direction vector (in XZ plane) for the sequence.
    # We average across the whole sequence for a stable per-trial rotation.
    fwd_x, fwd_z = None, None
    df_full = df
    if n_frames is not None:
        df = df.iloc[:max(int(n_frames), 1)]

    def _midpoint_mean(df, short_names, ax):
        cols_info = _get_marker_cols(df, short_names, axes=(ax,))
        ax_cols = [m[ax] for m in cols_info.values() if ax in m]
        if not ax_cols:
            return None
        return df[ax_cols].mean(axis=1).mean()  # scalar: sequence mean

    all_4 = _get_marker_cols(df, ['LASI', 'RASI', 'LPSI', 'RPSI'], axes=('X', 'Z'))
    lasi_rasi = _get_marker_cols(df, ['LASI', 'RASI'], axes=('X', 'Z'))
    lsho_rsho = _get_marker_cols(df, ['LSHO', 'RSHO'], axes=('X', 'Z'))

    if len(all_4) == 4:
        # Forward = ant midpoint − post midpoint
        ant_x = _midpoint_mean(df, ['LASI', 'RASI'], 'X')
        ant_z = _midpoint_mean(df, ['LASI', 'RASI'], 'Z')
        pos_x = _midpoint_mean(df, ['LPSI', 'RPSI'], 'X')
        pos_z = _midpoint_mean(df, ['LPSI', 'RPSI'], 'Z')
        if None not in (ant_x, ant_z, pos_x, pos_z):
            fwd_x = ant_x - pos_x
            fwd_z = ant_z - pos_z
    if fwd_x is None and len(lasi_rasi) == 2:
        # Use LASI→RASI vector; forward is perpendicular to it (−Z of left→right)
        lx = _midpoint_mean(df, ['LASI'], 'X')
        lz = _midpoint_mean(df, ['LASI'], 'Z')
        rx = _midpoint_mean(df, ['RASI'], 'X')
        rz = _midpoint_mean(df, ['RASI'], 'Z')
        if None not in (lx, lz, rx, rz):
            lr_x, lr_z = rx - lx, rz - lz
            # Forward is 90° CCW from L→R in XZ: (−Z component, +X component)
            fwd_x = -lr_z
            fwd_z = lr_x
    if fwd_x is None and len(lsho_rsho) == 2:
        lx = _midpoint_mean(df, ['LSHO'], 'X')
        lz = _midpoint_mean(df, ['LSHO'], 'Z')
        rx = _midpoint_mean(df, ['RSHO'], 'X')
        rz = _midpoint_mean(df, ['RSHO'], 'Z')
        if None not in (lx, lz, rx, rz):
            lr_x, lr_z = rx - lx, rz - lz
            fwd_x = -lr_z
            fwd_z = lr_x

    df = df_full   # heading estimated (possibly on the first n_frames); rotate the whole trial
    if fwd_x is None:
        return df, None

    # Normalize forward vector
    magnitude = np.sqrt(fwd_x ** 2 + fwd_z ** 2)
    if magnitude < 1e-9:
        return df, None

    fwd_x /= magnitude
    fwd_z /= magnitude

    # Angle to rotate: we want fwd → +Z (0, 1)
    # Current heading angle (measured from +Z): atan2(fwd_x, fwd_z)
    # Rotation needed: -heading_angle (rotate to bring fwd to +Z)
    heading_angle = float(np.arctan2(fwd_x, fwd_z))
    theta = -heading_angle
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Rotation matrix (Y-up, rotating in XZ plane):
    # X' =  cos(theta)*X + sin(theta)*Z
    # Z' = -sin(theta)*X + cos(theta)*Z
    df_out = df.copy()

    # Collect all raw X and Z position column pairs
    all_x_cols = [c for c in df.columns if c.endswith('_X') and not c.startswith('rel_')]
    all_z_cols = [c for c in df.columns if c.endswith('_Z') and not c.startswith('rel_')]

    # Match them by stripping the _X/_Z suffix
    x_base = {c[:-2]: c for c in all_x_cols}
    z_base = {c[:-2]: c for c in all_z_cols}
    common_bases = set(x_base.keys()) & set(z_base.keys())

    for base in common_bases:
        x_col = x_base[base]
        z_col = z_base[base]
        orig_x = df[x_col].values.copy()
        orig_z = df[z_col].values.copy()
        df_out[x_col] = cos_t * orig_x + sin_t * orig_z
        df_out[z_col] = -sin_t * orig_x + cos_t * orig_z

    return df_out, theta


def scale_by_height(df: pd.DataFrame, height_m: float) -> pd.DataFrame:
    """
    Divide all raw marker position coordinates by subject height, making
    spatial features dimensionless and comparable across subjects of different
    stature.  Velocity/acceleration columns (magnitude scales with 1/height)
    and relative-position columns are also scaled.
    """
    if height_m is None or height_m <= 0:
        return df
    df_out = df.copy()
    spatial_cols = [
        c for c in df_out.columns
        if not c.startswith('rel_')
    ]
    # Divide all non-rel columns by height
    df_out[spatial_cols] = df_out[spatial_cols] / height_m
    return df_out


def apply_cop_transform(
    cop_x: np.ndarray,
    cop_z: np.ndarray,
    pelvis_origin: Optional[pd.DataFrame],
    heading_theta: Optional[float],
    height_m: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the same pelvis-root subtraction, heading rotation, and height
    scaling to Centre of Pressure (px, pz) targets that was applied to the
    input marker data.

    Coordinate convention
    ---------------------
    cop_x and cop_z arrive in the **raw marker coordinate frame** — the same
    spatial frame that pelvis_origin lives in.  For GL this means X ~ AP,
    Z ~ ML (QTM convention after Y<->Z swap); for PH this is X ~ ML, Z ~ AP
    (OpenSim convention).  The heading rotation (step 2) aligns both to the
    canonical model frame where X = ML and Z = AP.

    Unit handling
    -------------
    Raw CoP from force plates is in **meters** (typical values ~0.3-0.8).
    pelvis_origin must be in **millimeters** (typical values ~300-800); the
    caller converts it when the marker file is in metres (wind turbine, via
    read_trc_file's marker_units_to_mm).  Step 0 converts CoP m -> mm before
    subtracting the pelvis so that everything is in a consistent mm frame.

    Parameters
    ----------
    cop_x, cop_z : 1-D arrays, length == num_frames.
                   Expected in **meters** (raw force-plate units).
    pelvis_origin : DataFrame with pelvis_X and pelvis_Z columns,
                    in **millimeters** (marker coordinate frame).
    heading_theta : Rotation angle (radians) from align_heading.
    height_m      : Subject height in metres.

    Returns
    -------
    Transformed cop_x, cop_z arrays (in mm, pelvis-relative,
    heading-aligned, height-scaled).
    """
    cop_x = cop_x.copy().astype(np.float64)
    cop_z = cop_z.copy().astype(np.float64)
    n = len(cop_x)

    # 0. Convert CoP from meters to millimeters so units match the marker
    #    (and pelvis_origin) coordinate frame.
    cop_x *= 1000.0
    cop_z *= 1000.0

    # 1. Pelvis subtraction (both now in mm)
    if pelvis_origin is not None:
        px_vals = pelvis_origin['pelvis_X'].values[:n] if 'pelvis_X' in pelvis_origin else np.zeros(n)
        pz_vals = pelvis_origin['pelvis_Z'].values[:n] if 'pelvis_Z' in pelvis_origin else np.zeros(n)
        cop_x -= px_vals
        cop_z -= pz_vals

    # 2. Heading rotation (same formula as for marker data in XZ plane)
    if heading_theta is not None:
        cos_t, sin_t = np.cos(heading_theta), np.sin(heading_theta)
        orig_x, orig_z = cop_x.copy(), cop_z.copy()
        cop_x = cos_t * orig_x + sin_t * orig_z
        cop_z = -sin_t * orig_x + cos_t * orig_z

    # 3. Height scaling
    if height_m is not None and height_m > 0:
        cop_x /= height_m
        cop_z /= height_m

    return cop_x.astype(np.float32), cop_z.astype(np.float32)


def normalize_data(df, type='minmax'):
    """Normalizes data using MinMaxScaler or StandardScaler.
    Ensures all column names are strings before scaling and numeric columns are float32.
    """
    # Convert all column names to strings to handle potential MultiIndex or mixed types
    df.columns = df.columns.astype(str)

    if type == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization type. Choose 'minmax' or 'standard'.")

    numeric_cols = df.select_dtypes(include=np.number).columns

    if not numeric_cols.empty:
        # Ensure the data being scaled is explicitly float to prevent issues with scaler
        try:
            df_numeric_float32 = df[numeric_cols].astype(np.float32)
            scaled_values = scaler.fit_transform(df_numeric_float32)
            # Ensure the output of the scaler is also float32
            df[numeric_cols] = scaled_values.astype(np.float32)
        except Exception as e:
            print(f"Error during scaling or type conversion in normalize_data: {e}")
            raise e
    else:
        print("Warning: No numeric columns found to normalize in normalize_data.")

    return df

def robust_normalize_data(data_df, type='standard'):
    """
    Normalizes data while handling potential issues like near-zero variance,
    duplicate columns, and NaNs.
    """
    # Create a copy to avoid modifying the original DataFrame passed to this function
    df_to_normalize = data_df.copy()

    if not isinstance(df_to_normalize, pd.DataFrame):
        print(f"Warning (robust_normalize_data): Input data is not a pandas DataFrame. Type: {type(df_to_normalize)}. Attempting to convert.")
        try:
            df_to_normalize = pd.DataFrame(df_to_normalize)
        except Exception as e:
            print(f"Error (robust_normalize_data): Could not convert input to DataFrame. Error: {e}. Returning original data.")
            return df_to_normalize

    if df_to_normalize.empty:
        return df_to_normalize

    # Fix for duplicate columns
    df_to_normalize = df_to_normalize.loc[:,~df_to_normalize.columns.duplicated()]

    if type == 'standard':
        mean = df_to_normalize.mean()
        std = df_to_normalize.std()

        normalized_df = df_to_normalize # Start with the structure of the copy

        for col in df_to_normalize.columns:
            if std[col] < 1e-9:  # Check if standard deviation is near-zero
                normalized_df[col] = 0.0  # Assign 0.0 to columns with no variance
            else:
                # Apply normalization only if std is not near-zero
                normalized_df[col] = (df_to_normalize[col] - mean[col]) / std[col]

        # Replace any NaNs or infs
        normalized_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        normalized_df.fillna(0.0, inplace=True)

        return normalized_df

    elif type == 'minmax':
        min_val = df_to_normalize.min()
        max_val = df_to_normalize.max()
        range_val = max_val - min_val

        normalized_df = df_to_normalize

        for col in df_to_normalize.columns:
            if range_val[col] < 1e-9:
                normalized_df[col] = 0.0
            else:
                normalized_df[col] = (df_to_normalize[col] - min_val[col]) / range_val[col]

        normalized_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        normalized_df.fillna(0.0, inplace=True)

        return normalized_df
    else:
        print(f"Warning (robust_normalize_data): Unsupported normalization type '{type}'. Returning original data (copy).")
        return df_to_normalize
