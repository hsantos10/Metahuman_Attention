# eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import MARKER_NAME_MAP, cross_correlation_lag0  # Import marker name mapping and cross-correlation function

def calculate_correlation_matrix(df, file_type="TRC"):
    """Calculates and visualizes the correlation matrix."""

    correlation_matrix = df.corr()

    plt.figure(figsize=(40, 32))
    sns.set(font_scale=1.2)

    if file_type == "TRC":
        # Replace abbreviations in column names for TRC correlation matrix heatmap
        correlation_matrix.columns = pd.MultiIndex.from_tuples([(MARKER_NAME_MAP.get(marker, marker), axis) if isinstance(marker, str) else (marker, axis) for marker, axis in correlation_matrix.columns])
        correlation_matrix.index = pd.MultiIndex.from_tuples([(MARKER_NAME_MAP.get(marker, marker), axis)if isinstance(marker, str) else (marker, axis) for marker, axis in correlation_matrix.index])
        title = "Correlation Matrix of Trajectory Data with Full Marker Names"
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        file_name = "correlation_matrix_trc_full_names.png"
    else:
        title = f"Correlation Matrix of {file_type} Data"  # More generic title
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(rotation=0, fontsize=14)
        file_name = f"correlation_matrix_{file_type.lower()}.png"  # Generic filename

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5,
        cbar_kws={'shrink': .75}
    )

    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()
    sns.set(font_scale=1)

def calculate_cross_correlation(trc_data, mot_data):
    """Calculates and visualizes cross-correlation between TRC and MOT data."""

    cross_corr_results = pd.DataFrame(index=mot_data.columns, columns=trc_data.columns)

    trc_cols_subset = trc_data.columns.tolist()
    mot_cols_subset = mot_data.columns.tolist()

    for mot_col in mot_cols_subset:
        for trc_col in trc_cols_subset:
            corr_lag0 = cross_correlation_lag0(trc_data[trc_col], mot_data[mot_col])
            cross_corr_results.at[mot_col, trc_col] = corr_lag0

    top_markers_abs_corr = {}
    excel_output_data = []

    for mot_col in cross_corr_results.index:
        abs_correlations = cross_corr_results.loc[mot_col].abs().sort_values(ascending=False)

        top_10_markers = abs_correlations.head(10)
        top_markers_abs_corr[mot_col] = top_10_markers

        top_marker_data = []

        actual_correlations = cross_corr_results.loc[mot_col][top_10_markers.index]

        sorted_markers_with_values = sorted(actual_correlations.items(), key=lambda item: abs(item[1]), reverse=True)

        print(f"\n--- Top 10 TRC Markers (Absolute Correlation with MOT Column: {mot_col} at Lag 0) ---")
        print(f"MOT Column: {mot_col}")
        print(f"{'TRC Marker':<40} | {'Correlation Value (Lag 0)':<25}")
        print(f"{'-'*40:<40}-|{'-'*25:<25}")

        for marker_col, corr_value in sorted_markers_with_values:
            full_marker_name = MARKER_NAME_MAP.get(marker_col[0], marker_col[0])
            axis = marker_col[1]
            marker_with_axis = f"{full_marker_name} ({axis})"
            print(f"{str(marker_with_axis):<40} | {corr_value:.2f}")
            top_marker_data.append({'MOT Column': mot_col, 'TRC Marker': marker_with_axis, 'Correlation Value (Lag 0)': corr_value})
        excel_output_data.extend(top_marker_data)

    top_markers_df = pd.DataFrame(excel_output_data)

    plt.figure(figsize=(30, 45))
    sns.heatmap(cross_corr_results.astype(float).transpose(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Lag 0 Cross-Correlation Between MOT and TRC Columns (Full Marker Names)', fontsize=16)
    plt.ylabel('MOT Columns', fontsize=14)
    plt.xlabel('TRC Markers', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig("cross_correlation_heatmap_lag0_full_names_v5.png", dpi=300)
    plt.show()

    # Save DataFrames to TXT and CSV files
    txt_file_path = "top_trc_markers.txt"
    csv_file_path = "cross_correlation_lag0.csv"

    # Save top markers table to TXT
    with open(txt_file_path, 'w') as f:
        f.write("--- Top 10 TRC Markers (Absolute Correlation with MOT Columns at Lag 0) ---\n\n")
        for mot_col, top_markers in top_markers_abs_corr.items():
            f.write(f"MOT Column: {mot_col}\n")
            f.write(f"{'TRC Marker':<40} | {'Correlation Value (Lag 0)':<25}\n")
            f.write(f"{'-'*40:<40}-|{'-'*25:<25}\n")

            # Get actual correlation values (not absolute) for sorting
            actual_correlations = cross_corr_results.loc[mot_col][top_markers.index]

            # Sort markers by absolute correlation value (descending)
            sorted_markers_with_values = sorted(actual_correlations.items(), key=lambda item: abs(item[1]), reverse=True)

            for marker_col, corr_value in sorted_markers_with_values:
                full_marker_name = MARKER_NAME_MAP.get(marker_col[0], marker_col[0])
                axis = marker_col[1]
                marker_with_axis = f"{full_marker_name} ({axis})"
                f.write(f"{str(full_marker_name):<40} | {corr_value:.2f}\n")
            f.write("\n") # Add extra newline for separation between MOT columns


    # Save cross_corr_results DataFrame to CSV
    cross_corr_results.to_csv(csv_file_path)

    print(f"\nResults tables saved to TXT file: {txt_file_path}")
    print(f"Cross-correlation matrix saved to CSV file: {csv_file_path}")