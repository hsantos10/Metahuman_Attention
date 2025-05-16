# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:26:17 2025

@author: David
"""

# utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

MARKER_NAME_MAP = {
    'LSHO': 'Left Shoulder marker',
    'RSHO': 'Right Shoulder marker',
    'LELBL': 'Left Elbow Lateral Epicondyle',
    'LELBM': 'Left Elbow Medial Epicondyle',
    'LWRL': 'Left Wrist Radial Styloid',
    'LWRM': 'Left Wrist Ulnar Styloid',
    'RELBL': 'Right Elbow Lateral Epicondyle',
    'RELBM': 'Right Elbow Medial Epicondyle',
    'RWRL': 'Right Wrist Radial Styloid',
    'RWRM': 'Right Wrist Ulnar Styloid',
    'LKNE': 'Left Knee Lateral Epicondyle',
    'RKNE': 'Right Knee Lateral Epicondyle',
    'LTOE': 'Left Toe marker',
    'LHEE': 'Left Heel marker',
    'LANKL': 'Left Ankle Lateral Malleolus',
    'RTOE': 'Right Toe marker',
    'RHEE': 'Right Heel marker',
    'RANKM': 'Right Ankle Medial Malleolus',
    'LASI': 'Left Anterior Superior Iliac Spine',
    'LPSI': 'Left Posterior Superior Iliac Spine',
    'RPSI': 'Right Posterior Superior Iliac Spine',
    'RASI': 'Right Anterior Superior Iliac Spine',
    'MChest': 'Mid-Chest marker',
    'SENL': 'Sensor Left',
    'SENR': 'Sensor Right',
    'LANKM': 'Left Ankle Medial Malleolus',
    'RANKL': 'Right Ankle Lateral Malleolus'
}

def normalize_data(df, type='minmax'):
    """Normalizes data using MinMaxScaler or StandardScaler."""

    if type == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization type. Choose 'minmax' or 'standard'.")

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def cross_correlation_lag0(series1, series2):
    """Calculates cross-correlation at lag 0 between two pandas Series."""

    series1_numeric = pd.to_numeric(series1, errors='coerce')
    series2_numeric = pd.to_numeric(series2, errors='coerce')

    if not (series1_numeric.notna().sum() > 2 and series2_numeric.notna().sum() > 2 and series1_numeric.var() > 1e-6 and series2_numeric.var() > 1e-6):
        return np.nan
    else:
        corr_val = series1_numeric.corr(series2_numeric)
        return corr_val
    
def read_sto_file(file_path):
    """Reads an STO file into a Pandas DataFrame, handling header complexities."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find the start of the numeric data
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('time'):
                data_start = i + 1
                break

        if data_start == 0:
            raise ValueError("Could not find 'time' header in STO file")

        # Read the data
        data = []
        for line in lines[data_start:]:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                data.append([float(v) for v in parts])
            except ValueError:
                # Skip lines that can't be converted to float
                continue

        if not data:
            raise ValueError("No valid data found in STO file")

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Remove the 'time' column if it exists
        if len(df.columns) > 1:
            df = df.iloc[:, 1:]

        return df.values  # Return as numpy array for consistency

    except Exception as e:
        print(f"Error processing STO file {file_path}: {str(e)}")
        return None