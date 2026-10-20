# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:26:17 2025

@author: David
"""

# utils.py
import pandas as pd
import numpy as np
import os
import re

# Note: MARKER_NAME_MAP and ESSENTIAL_MARKER_SHORT_NAMES_* moved to config.py
# Note: normalize_data moved to preprocessing.py

def cross_correlation_lag0(series1, series2):
    """Calculates cross-correlation at lag 0 between two pandas Series."""

    series1_numeric = pd.to_numeric(series1, errors='coerce')
    series2_numeric = pd.to_numeric(series2, errors='coerce')

    if not (series1_numeric.notna().sum() > 2 and series2_numeric.notna().sum() > 2 and series1_numeric.var() > 1e-6 and series2_numeric.var() > 1e-6):
        return np.nan
    else:
        corr_val = series1_numeric.corr(series2_numeric)
        return corr_val

def load_metadata(metadata_filepath, subject_id_col, weight_col, height_col, sheet_name='Experiment Design'):
    try:
        if metadata_filepath.endswith('.csv'):
            metadata_df = pd.read_csv(metadata_filepath)
        else:
            # Assume excel (Patient Handling: 'Experiment Design'; wind-turbine: 'Design')
            metadata_df = pd.read_excel(metadata_filepath, sheet_name=sheet_name)

        actual_columns = metadata_df.columns.tolist()

        found_subject_col, found_weight_col, found_height_col = None, None, None
        for col_name in actual_columns:
            if col_name.strip().lower().startswith(subject_id_col.lower()): found_subject_col = col_name
            if col_name.strip().lower().startswith(weight_col.lower()): found_weight_col = col_name
            if col_name.strip().lower().startswith(height_col.lower()): found_height_col = col_name

        final_subject_col = found_subject_col or subject_id_col
        final_weight_col = found_weight_col or weight_col
        final_height_col = found_height_col or height_col

        if not all([found_subject_col, found_weight_col, found_height_col]):
            missing = [col for exp, found, col in zip([subject_id_col, weight_col, height_col],
                                                    [found_subject_col, found_weight_col, found_height_col],
                                                    [subject_id_col, weight_col, height_col]) if not found]
            print(f"Error: Missing metadata columns: {missing}. Available: {actual_columns}")
            return None

        # Remove 's' or 'subj' prefixes before converting to numeric to handle "s001" or "subj01"
        metadata_df[final_subject_col] = metadata_df[final_subject_col].astype(str).str.lower().str.replace('subj', '').str.replace('s', '')

        metadata_df[final_subject_col] = pd.to_numeric(metadata_df[final_subject_col], errors='coerce')
        metadata_df.dropna(subset=[final_subject_col], inplace=True)
        metadata_df[final_subject_col] = metadata_df[final_subject_col].astype(int).astype(str)

        metadata_df = metadata_df.set_index(final_subject_col)
        metadata_df = metadata_df[~metadata_df.index.duplicated(keep='first')]
        return metadata_df[[final_weight_col, final_height_col]]

    except FileNotFoundError:
        print(f"Error: Metadata file '{metadata_filepath}' not found.")
        return None
    except KeyError as e:
        print(f"Error: Column not found in metadata after attempting to match. Original: {e}. Check names.")
        return None
    except Exception as e:
        print(f"Error reading/processing metadata: {e}")
        return None

def extract_subject_id_from_filepath(filepath, prefix=None, suffix=None):
    # Handle optional prefix/suffix
    import os # ensure os is imported
    import re # ensure re is imported

    # Detect lab namespace from path.
    # When using the CombinedAdapter, the metadata index uses 'gl_' / 'ph_' prefixes
    # so that the same numeric subject ID in two different labs doesn't collide.
    filepath_lower = filepath.replace('\\', '/').lower()
    lab_prefix = ""
    if "groundlink" in filepath_lower:
        lab_prefix = "gl_"
    elif "windturbine" in filepath_lower:
        lab_prefix = "wt_"
    elif "patienthandling" in filepath_lower or ("tamu" in filepath_lower and "opensim" in filepath_lower):
        lab_prefix = "ph_"

    # Try directory matching first (e.g. subj01/...)
    path_parts = filepath.split(os.sep)
    for part in path_parts:
        match = re.match(r'^subj(\d+)$', part, re.IGNORECASE)
        if match:
            return lab_prefix + str(int(match.group(1)))
        # Handle s001 style
        match_s = re.match(r'^s(\d{3})$', part, re.IGNORECASE)
        if match_s:
            return lab_prefix + str(int(match_s.group(1)))

    filename = os.path.basename(filepath)
    if prefix and filename.lower().startswith(prefix.lower()): filename = filename[len(prefix):]
    if suffix and filename.lower().endswith(suffix.lower()): filename = filename[:-len(suffix)]

    if filename.isdigit(): return lab_prefix + str(int(filename))

    match_subj = re.match(r'^subj(\d+)$', filename, re.IGNORECASE)
    if match_subj: return lab_prefix + str(int(match_subj.group(1)))

    # Handle s001 in filename
    match_s_file = re.match(r'^s(\d{3})', filename, re.IGNORECASE)
    if match_s_file: return lab_prefix + str(int(match_s_file.group(1)))

    return lab_prefix + filename
