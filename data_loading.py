# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:27:02 2025

@author: David
"""

# data_loading.py
import pandas as pd
import glob
from utils import MARKER_NAME_MAP, read_sto_file, normalize_data  # Import normalize_data
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import matplotlib.pyplot as plt  # Import matplotlib
import os

def read_trc_file(file_path, markers_to_remove=None):
    """Reads a TRC file and returns a pandas DataFrame, optionally removing specified markers."""
    try:
        with open(file_path, 'r') as f:
            header_lines = [f.readline().strip() for _ in range(5)]
            marker_labels_line = header_lines[3].split('\t')
            axis_labels_line = header_lines[4].split('\t')

        marker_names_abbr = []  # Keep abbreviations for MultiIndex creation
        marker_names_full = []  # List to store full marker names for display

        for label in marker_labels_line[2:]:
            if label != '':
                marker_names_abbr.extend([label] * 3)
                full_name = MARKER_NAME_MAP.get(label, label)  # Use full name if available
                marker_names_full.extend([full_name] * 3)

        axis_final = []
        num_markers = len(marker_names_abbr) // 3  # Use abbreviations for num_markers calculation
        axis_labels_base = ['X', 'Y', 'Z']
        for _ in range(num_markers):
            axis_final.extend(axis_labels_base)

        df = pd.read_csv(file_path, sep='\t', skiprows=5, header=None)  # Read all columns first
        

        
        # --- NEW: NaN Imputation ---
        df = df.iloc[:, :-1]  # Drop the last column (It is noramlly full with nans. this is a temporary fix)
        # print("\n--- Imputing NaNs ---")
        # nan_count_before = df_trimmed.isnull().sum().sum()
        # print(f"NaNs before imputation: {nan_count_before}")
        
        # df_trimmed.interpolate(inplace=True, limit_direction='both')  # Linear interpolation
        # df_trimmed.fillna(0, inplace=True)  # Fill any remaining NaNs (at the start/end) with 0
        
        # nan_count_after = df_trimmed.isnull().sum().sum()
        # print(f"NaNs after imputation: {nan_count_after}")
        # # --- END NaN Imputation ---


        expected_num_cols_data = len(marker_names_abbr) + 2  # Use abbreviations for column count

        if df.shape[1] > expected_num_cols_data:
            df_trimmed = df.iloc[:, :expected_num_cols_data]
        else:
            df_trimmed = df

        column_index = pd.MultiIndex.from_arrays(
            [marker_names_full, axis_final], names=['Marker', 'Axis'])  # Use FULL names for MultiIndex

        temp_cols = ['Frame#', 'Time'] + column_index.tolist()
        
        

        # Modified column assignment:
        if len(df_trimmed.columns) == len(temp_cols):
            df_trimmed.columns = temp_cols
        else:
            print("Warning: Column count mismatch. MultiIndex not applied.")
            temp_cols_simple = [f'Column_{i}' for i in range(len(df_trimmed.columns))]  # Create simple column names
            df_trimmed.columns = temp_cols_simple

        df_trimmed = df_trimmed.drop(columns=['Frame#', 'Time'], errors='ignore')

        if markers_to_remove:
            cols_to_remove = []
            for marker in markers_to_remove:
                # Remove both full name and abbreviation
                for i, name in enumerate(marker_names_full):
                    if marker in name:
                        cols_to_remove.append((name, axis_final[i % 3]))  # Use axis_final index
            if cols_to_remove:  # Only drop if there are columns to remove
                df_trimmed = df_trimmed.drop(columns=cols_to_remove, errors='ignore')
                
        # # --- ADDED DEBUGGING ---
        # print(f"\n--- Processing TRC file: {file_path} ---")
        # print(f"Shape of DataFrame after reading: {df.shape}")
        # nan_count = df.isnull().sum().sum()
        # print(f"Total NaNs in DataFrame after reading: {nan_count}")
        # if nan_count > 0:
        #     print("Location of NaNs:")
        #     for col in df.columns:
        #         nan_in_col = df[col].isnull().sum()
        #         if nan_in_col > 0:
        #             print(f"  Column '{col}': {nan_in_col} NaNs ({(nan_in_col / len(df)) * 100:.2f}%)")
        #             # Optionally, print first few rows with NaNs in this column
        #             print(df[df[col].isnull()].head())
        #     # --- END ADDED DEBUGGING ---
        return df_trimmed

    except (FileNotFoundError, ValueError, IndexError, pd.errors.ParserError) as e:
        print(f"Error reading or processing TRC file: {e}")
        return None


def process_trc_files(trc_files, markers_to_remove=None):
    """
    Reads TRC files, removes specified markers, and ensures all DataFrames have the same columns.

    Args:
        trc_files (list): List of paths to TRC files.
        markers_to_remove (list, optional): List of marker names to remove. Defaults to None.

    Returns:
        list of pandas.DataFrame: A list of processed DataFrames, all with the same columns, or an empty list on error.
    """
    processed_data = []
    all_marker_names = set()

    # Define a list of markers to remove if markers_to_remove is None
    default_markers_to_remove = ['Box1', 'Box2', 'Box3', 'Box4', 'DL1', 'DL2', 'DL3', 'DH1', 'DH2', 'DH3', 'T1', 'T2']
    if markers_to_remove is None:
        markers_to_remove = default_markers_to_remove

    # 1. Read and optionally remove markers, collect all marker names:
    for file_path in trc_files:
        df = read_trc_file(file_path, markers_to_remove)
        if df is not None:
            processed_data.append(df)
            all_marker_names.update(df.columns)
        else:
            print(f"Skipping file: {file_path}")  # Keep the user informed
            return []  # Return empty list if any file fails to process

    if not processed_data:
        print("No TRC files were successfully processed.")
        return []

    # 2. Convert set to list for consistent ordering
    all_marker_names_list = list(all_marker_names)

    # 3. Add missing columns and ensure order:
    for i, df in enumerate(processed_data):
        missing_cols = [col for col in all_marker_names_list if col not in df.columns]
        for col in missing_cols:
            df[col] = 0  # Or np.nan, depending on how you want to handle missing data
        processed_data[i] = df[all_marker_names_list]  # Reorder columns

    return processed_data


def read_data_file(file_path):
    """Reads a data file (.mot or .sto) into a Pandas DataFrame."""
    if file_path.endswith(".mot"):
        df = read_mot_file(file_path)  # Assuming read_mot_file is correct
    elif file_path.endswith(".sto"):
        df = read_sto_file(file_path)  # Now using the improved version
        if df is not None:
            df = pd.DataFrame(df)  # Convert NumPy array to DataFrame
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return df


def read_mot_file(file_path):
    """
    Reads and parses a .mot (Motion) file containing motion data
    Args:
        file_path: Path to the .mot file
    Returns:
        DataFrame containing the motion data
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    header_end_line = 0
    for i, line in enumerate(lines):
        if "endheader" in line.lower():
            header_end_line = i
            break

    header = lines[header_end_line + 1].strip().split('\t')
    data = pd.read_csv(file_path, delimiter='\t', skiprows=header_end_line + 2, names=header)

    # Drop the first column (time) and columns with all zeros
    data = data.drop(data.columns[0], axis=1, errors='ignore')
    cols_to_drop = data.columns[data.eq(0).all()]
    data = data.drop(columns=cols_to_drop, axis=1, errors='ignore')

    return data

def process_data_files(data_files):
    """
    Reads and processes data files (.mot or .sto), ensuring consistent columns.

    Args:
        data_files (list): List of paths to data files.

    Returns:
        list of pandas.DataFrame: A list of processed DataFrames, all with the same columns,
                                 or an empty list on error.
    """
    processed_data = []
    all_columns = set()

    for file_path in data_files:
        try:
            df = read_data_file(file_path)
            if df is not None:
                # Drop the first column (time) and columns with all zeros
                if isinstance(df, pd.DataFrame):  # Check if it's a DataFrame
                    #df = df.drop(df.columns[0], axis=1, errors='ignore') # I think this line was removing the wrong column
                    cols_to_drop = df.columns[df.eq(0).all()]
                    df = df.drop(columns=cols_to_drop, axis=1, errors='ignore')

                processed_data.append(df)
                if isinstance(df, pd.DataFrame):
                    all_columns.update(df.columns)
                else:
                    all_columns.update(df.columns.tolist())  # Handle NumPy array columns
            else:
                print(f"Skipping file: {file_path} due to reading errors.")
                return []  # Exit if any file fails
        except ValueError as e:
            print(f"Error processing file {file_path}: {e}")
            return []  # Exit if any file type is unsupported

    if not processed_data:
        print("No data files were successfully processed.")
        return []

    all_columns_list = list(all_columns)

    for i, df in enumerate(processed_data):
        missing_cols = [col for col in all_columns_list if col not in df.columns]
        for col in missing_cols:
            df[col] = 0  # Or np.nan, depending on how you want to handle missing data
        processed_data[i] = df[all_columns_list]  # Reorder columns
        
    #print(processed_data)

    return processed_data

def pad_or_truncate(sequence, max_length):
    seq_length = sequence.shape[0]
    if seq_length < max_length:
        pad_length = max_length - seq_length
        pad_array = np.zeros((pad_length, sequence.shape[1]))
        padded_sequence = np.concatenate([sequence, pad_array], axis=0)
        return padded_sequence, seq_length # Return original length
    elif seq_length > max_length:
        truncated_sequence = sequence[:max_length]
        return truncated_sequence, max_length # Original length is max_length (or seq_length before truncation)
    else:
        return sequence, seq_length
    
import matplotlib.pyplot as plt  # Import matplotlib


# You might need a similar update for `prepare_data_for_modeling` if you use it directly.
# For `load_and_process_split` (the one without metadata):
def load_and_process_split(trc_dfs, data_dfs, max_length, data_columns_to_use, split_name=""):
    trc_sequences = []
    data_sequences = []
    input_masks = [] # Add masks list

    for i, (trc_df, data_df) in enumerate(zip(trc_dfs, data_dfs)):
        trc_data = trc_df.copy()
        data = data_df.copy()
        
        trc_data_normalized = normalize_data(trc_data.copy(), type='standard')

        if data_columns_to_use:
            # Ensure columns exist, robustly select
            available_cols = [col_idx for col_idx in data_columns_to_use if col_idx < data.shape[1]]
            if len(available_cols) != len(data_columns_to_use):
                print(f"Warning in {split_name}: Not all data_columns_to_use indices are valid for a data_df of shape {data.shape}. Using: {available_cols}")
            if not available_cols:
                 print(f"Error in {split_name}: No valid data_columns_to_use. Skipping file {i+1}.")
                 continue
            data = data.iloc[:, available_cols]


        data_array = data.values

        # Pad and get original lengths
        trc_sequence_padded, trc_original_len = pad_or_truncate(trc_data_normalized.values, max_length)
        data_sequence_padded, data_original_len = pad_or_truncate(data_array, max_length) # Assuming targets also padded

        # Create input mask
        input_mask_np = np.zeros(max_length, dtype=np.float32)
        input_mask_np[:trc_original_len] = 1.0
        
        trc_sequences.append(torch.tensor(trc_sequence_padded, dtype=torch.float32))
        data_sequences.append(torch.tensor(data_sequence_padded, dtype=torch.float32))
        input_masks.append(torch.tensor(input_mask_np, dtype=torch.float32))

    if not trc_sequences: # If list is empty (e.g. due to errors or no data)
        print(f"Warning: No sequences processed for split: {split_name}")
        return [], [], []

    return trc_sequences, data_sequences, input_masks


def prepare_data_for_modeling(processed_trc_data, processed_data, data_columns_to_use, test_size=0.1, val_size=0.1, random_state=42):
    """
    Prepares processed TRC and data DataFrames for modeling by splitting, padding, and normalizing.
    TRC data is normalized, while .mot or .sto data is kept as is.

    Args:
        processed_trc_data (list of pd.DataFrame): List of processed TRC DataFrames.
        processed_data (list of pd.DataFrame): List of processed data DataFrames.
        data_columns_to_use (list): List of columns to use from data files.
        test_size (float): Proportion of data to use for testing.
        val_size (float): Proportion of training data to use for validation.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: A tuple containing (train_data, val_data, test_data, max_length, input_size, output_size)
               where each data split is a tuple of (trc_sequences, data_sequences).
    Prepares processed TRC and data DataFrames for modeling by splitting, padding, normalizing, and creating masks.
    Returns tuples of (sequences, targets, masks).
    """
    trc_train_val, trc_test, data_train_val, data_test = train_test_split(
        processed_trc_data, processed_data, test_size=test_size, random_state=random_state)
    
    # Adjust val_size to be a proportion of the new training set (trc_train_val)
    actual_val_size = val_size / (1 - test_size) if (1-test_size) > 0 else 0
    trc_train, trc_val, data_train, data_val = train_test_split(
        trc_train_val, data_train_val, test_size=actual_val_size, random_state=random_state)

    max_length = 0
    # Determine max_length from original lengths of training TRC data before padding
    for trc_df, data_df_from_list in zip(trc_train, data_train): # Iterate through original dataframes
        # Align lengths before determining max_length for padding
        # data_df needs to be handled carefully if it's a list of DFs or a single DF
        # Assuming processed_data is a list of DFs corresponding to processed_trc_data
        min_len_current_pair = min(trc_df.shape[0], data_df_from_list.shape[0])
        max_length = max(max_length, min_len_current_pair)
        
    if max_length == 0: # Fallback if no training data or all files were empty
        max_length = 651 
    print(f"Determined max_length for padding (from training data, after alignment): {max_length}")

    # load_and_process_split now returns three lists
    train_trc_seqs, train_data_seqs, train_input_masks = load_and_process_split(
        trc_train, data_train, max_length, data_columns_to_use, split_name="Train")
    val_trc_seqs, val_data_seqs, val_input_masks = load_and_process_split(
        trc_val, data_val, max_length, data_columns_to_use, split_name="Validation")
    test_trc_seqs, test_data_seqs, test_input_masks = load_and_process_split(
        trc_test, data_test, max_length, data_columns_to_use, split_name="Test")

    if not train_trc_seqs: # Check if training data processing failed
        print("Error: Training data could not be processed into sequences. Exiting.")
        # Depending on your setup, you might raise an error or handle this to prevent crashes
        # For now, let's return empty structures that might cause downstream errors,
        # or you can explicitly exit/raise.
        return ([], [], []), ([], [], []), ([], [], []), 0, 0, 0


    input_size = train_trc_seqs[0].shape[1] if train_trc_seqs and len(train_trc_seqs[0].shape) > 1 else 0
    output_size = train_data_seqs[0].shape[1] if train_data_seqs and len(train_data_seqs[0].shape) > 1 else 0


    train_data = (train_trc_seqs, train_data_seqs, train_input_masks)
    val_data = (val_trc_seqs, val_data_seqs, val_input_masks)
    test_data = (test_trc_seqs, test_data_seqs, test_input_masks)
    
    return train_data, val_data, test_data, max_length, input_size, output_size

# MODIFIED/NEW FUNCTIONS TO INCORPORATE META DATA START HERE

def load_metadata(metadata_filepath, subject_id_col, weight_col, height_col):
    """Loads metadata and prepares it for easy lookup."""
    try:
        # Change this line: Use pd.read_excel instead of pd.read_csv
        # Specify the sheet_name, which you identified as 'Experiment Design'
        metadata_df = pd.read_excel(metadata_filepath, sheet_name='Experiment Design')
        
        # Set subject ID as index for quick lookup
        metadata_df[subject_id_col] = metadata_df[subject_id_col].astype(str).str.lower().str.replace(r'[^0-9a-z]', '', regex=True)
        metadata_df = metadata_df.set_index(subject_id_col)
        return metadata_df[[weight_col, height_col]]
    except FileNotFoundError:
        print(f"Error: Metadata file '{metadata_filepath}' not found.")
        return None
    except KeyError as e:
        print(f"Error: Column {e} not found in metadata file. Needed: {subject_id_col}, {weight_col}, {height_col}")
        return None
    except Exception as e: # Catch other potential errors from reading Excel
        print(f"Error reading Excel metadata file or sheet: {e}")
        return None

def extract_subject_id_from_filepath(filepath, prefix, suffix):
    """
    Extracts a subject identifier from a filepath.
    Example: 'G:\\...\\MarkerData\\subj00.trc' -> 'subj00' or '00'
    Adjust this function based on your exact file naming and desired ID format.
    """
    filename = os.path.basename(filepath)
    # Remove prefix if it exists
    if prefix: # Handle cases where prefix might not always be there or is complex
        base_name_no_prefix = filename
        # Simplistic prefix removal, adjust if prefix varies
        # For "Experiment Detail.xlsx - subj" from TRC_FILE_PATH
        # Or "ForcesData" or "IK_Results" etc. from DATA_FILE_PATH
        # This part needs to be robust for your TRC_FILE_PATH and DATA_FILE_PATH
        # Example: if prefix is like "Experiment Detail.xlsx - subj"
        specific_prefix_trc = "Experiment Detail.xlsx - subj" # From your previous upload names
        specific_prefix_data = "" # What's the prefix for DATA_FILE_PATH files? Assume none for now
                                 # Or it could be "subject_trial_forces_" etc.

        # This logic needs to be robust based on how `glob.glob` in main.py resolves filenames
        # And what unique part identifies the subject
        if specific_prefix_trc in filename:
             base_name_no_prefix = filename.replace(specific_prefix_trc, "")
        # Add similar logic for DATA_FILE_PATH prefix if needed

        # Remove suffix (.trc, .mot, .sto)
        subject_id_str = os.path.splitext(base_name_no_prefix)[0]

    else: # If no prefix is given, just remove suffix
        subject_id_str = os.path.splitext(filename)[0]

    # Further clean to match metadata index (e.g., '00', '01')
    return subject_id_str.lower().str.replace(r'[^0-9a-z]', '', regex=True)


def load_and_process_split_with_metadata(
    file_list_trc, file_list_data, metadata_df,
    max_length, data_columns_to_use, split_name="",
    trc_file_prefix=None, trc_file_suffix=".trc",
    data_file_prefix=None, data_file_suffix=None, # e.g. ".mot" or ".sto"
    subject_id_col_in_metadata='Subject', # The column name in your metadata CSV for subject IDs
    weight_col_in_metadata='Weight',
    height_col_in_metadata='Height'
    ):
    """
    Loads TRC and other data, adds metadata, performs normalization and padding.
    """
    trc_sequences = []
    data_sequences = [] # This will be target sequences (e.g., ground forces)
    input_masks = [] # To store masks for input (TRC) sequences
    # target_masks = [] # If targets also need masking (e.g. for loss)

    # Determine suffixes if not provided
    if not trc_file_suffix and file_list_trc:
        _, trc_file_suffix = os.path.splitext(file_list_trc[0])
    if not data_file_suffix and file_list_data:
         _, data_file_suffix = os.path.splitext(file_list_data[0])


    for trc_filepath, data_filepath in zip(file_list_trc, file_list_data):
        # --- 1. Extract Subject ID from TRC filepath to lookup metadata ---
        # This assumes TRC and DATA files are paired and correspond to the same subject for this iteration
        subject_id_for_lookup = extract_subject_id_from_filepath(trc_filepath, trc_file_prefix, trc_file_suffix)
        # print(f"Split: {split_name}, TRC: {trc_filepath}, Extracted Subject ID: {subject_id_for_lookup}")


        try:
            meta_row = metadata_df.loc[subject_id_for_lookup]
            weight = meta_row[weight_col_in_metadata]
            height = meta_row[height_col_in_metadata]
        except KeyError:
            print(f"Warning: Subject ID '{subject_id_for_lookup}' from file '{trc_filepath}' not found in metadata. Skipping this file pair.")
            continue
        except Exception as e:
            print(f"Warning: Error accessing metadata for subject '{subject_id_for_lookup}': {e}. Skipping this file pair.")
            continue

        # --- 2. Load TRC data (Input Features X1) ---
        trc_df = read_trc_file(trc_filepath) # Assuming this returns a DataFrame
        if trc_df is None or trc_df.empty:
            print(f"Warning: Could not load TRC data from {trc_filepath}. Skipping.")
            continue

        # Add weight and height to TRC data (input features)
        # These will be repeated for every row (time step) in trc_df
        trc_df['weight'] = weight
        trc_df['height'] = height

        # Normalize TRC data (including new weight/height columns)
        trc_data_normalized = normalize_data(trc_df.copy(), type='standard') # Or your preferred normalization

        # --- 3. Load other data (Target Features Y) ---
        # This is 'processed_data' in your original code, e.g., ground forces from .mot files
        data_df = read_data_file(data_filepath) # Assuming this returns a DataFrame
        if data_df is None or data_df.empty:
            print(f"Warning: Could not load target data from {data_filepath}. Skipping.")
            continue

        if data_columns_to_use: # If you want to select specific columns from target data
            # Ensure columns exist before trying to select them
            actual_cols_to_use = [col for col in data_columns_to_use if col in data_df.columns]
            if len(actual_cols_to_use) != len(data_columns_to_use):
                print(f"Warning: Not all data_columns_to_use found in {data_filepath}. Using available: {actual_cols_to_use}")
            if not actual_cols_to_use:
                print(f"Error: None of the specified data_columns_to_use found in {data_filepath}. Skipping.")
                continue
            data_df_selected = data_df[actual_cols_to_use].copy()
        else:
            data_df_selected = data_df.copy()

        # Ensure data is NumPy array for padding/truncating
        data_array = data_df_selected.values

        # --- 4. Pad/Truncate ---
        # Pad/truncate both normalized TRC data and the target data to max_length
        # Align lengths: a common strategy is to truncate the longer one or pad the shorter one
        # to match the length of the *other* file in the pair if they are supposed to be time-aligned.
        # Or, pad/truncate both to a global max_length.
        # Your current `pad_or_truncate` works on individual sequences.
        # We need to ensure TRC sequence and DATA sequence have the same length *before* passing to DataLoader
        # if they are directly corresponding time-series.
        

        min_len = min(len(trc_data_normalized), len(data_array))
        print(f"Calculated min_len for alignment: {min_len}")
        trc_sequence_aligned = trc_data_normalized.iloc[:min_len].values
        data_sequence_aligned = data_array[:min_len]
        
        
        # Pad and get original length
        trc_sequence_padded, trc_original_len = pad_or_truncate(trc_sequence_aligned, max_length)
        data_sequence_padded, data_original_len = pad_or_truncate(data_sequence_aligned, max_length) # Assuming targets also padded similarly

        # Create input mask
        input_mask = np.zeros(max_length, dtype=np.float32)
        input_mask[:trc_original_len] = 1.0
        
        # # Create target mask (if needed for loss calculation on padded targets)
        # target_mask = np.zeros(max_length, dtype=np.float32)
        # target_mask[:data_original_len] = 1.0
        

        trc_sequences.append(torch.tensor(trc_sequence_padded, dtype=torch.float32))
        data_sequences.append(torch.tensor(data_sequence_padded, dtype=torch.float32))
        input_masks.append(torch.tensor(input_mask, dtype=torch.float32))
        # target_masks.append(torch.tensor(target_mask, dtype=torch.float32))

    if not trc_sequences or not data_sequences: # If lists are empty
        print(f"Warning: No data successfully processed for split: {split_name}")
        # Return empty tensors of appropriate shape if possible, or handle upstream
        # This depends on how your main.py expects empty data
        # For now, let's return empty lists and let the caller handle it.
        return [], [], [] # Or [], [], [], [] if target_masks also returned

    return trc_sequences, data_sequences, input_masks


def prepare_data_for_modeling_with_metadata(
    trc_file_paths, # List of full paths to TRC files
    data_file_paths, # List of full paths to .mot/.sto files (targets)
    metadata_df, # DataFrame from load_metadata()
    data_columns_to_use, # Columns from .mot/.sto to be used as targets
    trc_file_prefix, # Prefix for extract_subject_id_from_filepath
    trc_file_suffix,
    data_file_prefix, # Prefix for data files if needed for ID extraction (can be None)
    data_file_suffix, # Suffix for data files
    subject_id_col_in_metadata,
    weight_col_in_metadata,
    height_col_in_metadata,
    test_size=0.1,
    val_size=0.1, # This is proportion of (1-test_size) data
    random_state=42
    ):
    """
    Prepares TRC (input) and other data (target) files for modeling.
    Includes loading metadata, adding it to TRC data, splitting, padding, and normalization.
    """
    if metadata_df is None:
        raise ValueError("Metadata could not be loaded. Cannot proceed.")

    # Ensure the lists of TRC and DATA files are of the same length and correspond
    if len(trc_file_paths) != len(data_file_paths):
        raise ValueError("Mismatch in the number of TRC and DATA files. They must be paired.")

    # Create indices for splitting to keep file pairs together
    indices = list(range(len(trc_file_paths)))
    train_val_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    # Adjust val_size if it's meant to be a fraction of the original dataset
    # current val_size is a fraction of the (1-test_size) data
    # If you want val_size to be e.g. 0.1 of total, and test_size 0.1 of total:
    # train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size / (1-test_size), random_state=random_state)

    # More direct split if val_size is a fraction of the total dataset:
    # Let's say total data is 1.0. test_size=0.1. Remaining for train+val = 0.9.
    # If val_size (of total) is 0.1, then val_size relative to train_val_indices is 0.1 / 0.9.
    actual_val_size_for_split = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0
    train_indices, val_indices = train_test_split(train_val_indices, test_size=actual_val_size_for_split, random_state=random_state)


    trc_train = [trc_file_paths[i] for i in train_indices]
    data_train = [data_file_paths[i] for i in train_indices]
    trc_val = [trc_file_paths[i] for i in val_indices]
    data_val = [data_file_paths[i] for i in val_indices]
    trc_test = [trc_file_paths[i] for i in test_indices]
    data_test = [data_file_paths[i] for i in test_indices]

    # Determine max_length (from TRC data in the training set *after* loading)
    # This is tricky because max_length depends on file content.
    # A common approach: find max length across ALL files once, or use a fixed one.
    # Your original code finds it from trc_train DataFrames AFTER processing.
    # Let's keep it dynamic based on training files for now, but load them first.
    global_max_len = 0
    for fp_trc, fp_data in zip(trc_file_paths, data_file_paths): # Check all files for robust max_len
        df_trc = read_trc_file(fp_trc)
        df_data = read_data_file(fp_data)
        if df_trc is not None and df_data is not None:
            # Align first, then check length for padding
            current_min_len = min(len(df_trc), len(df_data))
            global_max_len = max(global_max_len, current_min_len)
    
    if global_max_len == 0:
        global_max_len = 651 # Fallback
    print(f"Determined global_max_len for padding (based on min_len of pairs before padding): {global_max_len}")

    # The load_and_process_split_with_metadata now returns three lists: trc_seqs, data_seqs, input_masks
    train_trc_seqs, train_data_seqs, train_input_masks = load_and_process_split_with_metadata(
        trc_train, data_train, metadata_df, global_max_len, data_columns_to_use, "Train",
        trc_file_prefix, trc_file_suffix, data_file_prefix, data_file_suffix,
        subject_id_col_in_metadata, weight_col_in_metadata, height_col_in_metadata
    )
    val_trc_seqs, val_data_seqs, val_input_masks = load_and_process_split_with_metadata(
        trc_val, data_val, metadata_df, global_max_len, data_columns_to_use, "Validation",
        trc_file_prefix, trc_file_suffix, data_file_prefix, data_file_suffix,
        subject_id_col_in_metadata, weight_col_in_metadata, height_col_in_metadata
    )
    test_trc_seqs, test_data_seqs, test_input_masks = load_and_process_split_with_metadata(
        trc_test, data_test, metadata_df, global_max_len, data_columns_to_use, "Test",
        trc_file_prefix, trc_file_suffix, data_file_prefix, data_file_suffix,
        subject_id_col_in_metadata, weight_col_in_metadata, height_col_in_metadata
    )

    if not train_trc_seqs:
        print("Error: Training data could not be processed. Exiting.")
        exit() # Or raise an error

    input_size = train_trc_seqs[0].shape[1] if train_trc_seqs else 0
    output_size = train_data_seqs[0].shape[1] if train_data_seqs else 0

    # Package into tuples: (trc_sequences, data_sequences, input_masks)
    # If you created target_masks, include them as the fourth element.
    train_data_tuple = (train_trc_seqs, train_data_seqs, train_input_masks)
    val_data_tuple = (val_trc_seqs, val_data_seqs, val_input_masks)
    test_data_tuple = (test_trc_seqs, test_data_seqs, test_input_masks)

    return train_data_tuple, val_data_tuple, test_data_tuple, global_max_len, input_size, output_size
