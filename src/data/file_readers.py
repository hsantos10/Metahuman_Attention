import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from src.config import MARKER_NAME_MAP

_PRINTED_TRC_UNIT_MSG = False

# Custom Exceptions
class TRCFileProcessingError(Exception):
    """Base class for errors during TRC file processing."""
    pass

class EssentialMarkerMissingError(TRCFileProcessingError):
    """Custom exception for missing essential markers."""
    def __init__(self, message, missing_markers=None):
        super().__init__(message)
        self.missing_markers = missing_markers if missing_markers is not None else []

class TRCFileReadingError(TRCFileProcessingError):
    """Custom exception for general TRC file reading issues."""
    pass

@dataclass
class FileReadResult:
    """Standardized return object for file readers."""
    dataframe: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g. NaN stats
    issues: List[str] = field(default_factory=list)

def read_trc_file(file_path, essential_marker_short_names_list=None) -> FileReadResult:
    """
    Reads a TRC file, processes it, calculates NaN statistics, and handles errors by raising exceptions.
    """
    problems_encountered = []

    try:
        with open(file_path, 'r') as f:
            try:
                header_lines = [f.readline().strip() for _ in range(5)]
                if len(header_lines) < 5 or not header_lines[3]:
                    raise TRCFileReadingError(f"TRC file '{file_path}' has an incomplete or malformed header.")
            except Exception as e:
                raise TRCFileReadingError(f"Error reading header from TRC file '{file_path}': {e}")

        # Parse TRC declared units from the first 3 header lines when available
        declared_units = None
        try:
            header_keys = [x.strip() for x in header_lines[1].split('\t') if x.strip()]
            header_vals = [x.strip() for x in header_lines[2].split('\t') if x.strip()]
            if 'Units' in header_keys:
                uidx = header_keys.index('Units')
                if uidx < len(header_vals):
                    declared_units = header_vals[uidx].strip().lower()
        except Exception:
            declared_units = None

        raw_header_marker_names_unfiltered = header_lines[3].split('\t')
        actual_header_labels = [lbl.strip() for lbl in raw_header_marker_names_unfiltered if lbl.strip()]

        file_marker_names_in_order = []
        if actual_header_labels:
            start_idx = 0
            if actual_header_labels[0] == 'Frame#':
                start_idx += 1
            if len(actual_header_labels) > start_idx and actual_header_labels[start_idx] == 'Time':
                start_idx += 1
            file_marker_names_in_order = actual_header_labels[start_idx:]

        # Optional marker aliases, e.g. TRC_MARKER_ALIASES="LKNE:LKNEL,RKNE:RKNEL" maps a requested
        # (canonical) name to the label used in this file when the canonical one is absent.
        alias_env = os.getenv("TRC_MARKER_ALIASES", "").strip()
        if alias_env and file_marker_names_in_order:
            for pair in alias_env.split(","):
                if ":" not in pair: continue
                want, have = [s.strip() for s in pair.split(":", 1)]
                if want and have and want not in file_marker_names_in_order and have in file_marker_names_in_order:
                    file_marker_names_in_order = [want if m == have else m for m in file_marker_names_in_order]

        if not file_marker_names_in_order:
            problems_encountered.append("No marker names found in the TRC header after 'Frame#' and 'Time'.")

        essential_set = set(essential_marker_short_names_list) if essential_marker_short_names_list else None
        if essential_set:
            missing_essential_in_header = [m for m in essential_set if m not in file_marker_names_in_order]
            if missing_essential_in_header:
                raise EssentialMarkerMissingError(
                    f"Essential markers missing from TRC file '{file_path}' header: {missing_essential_in_header}",
                    missing_markers=missing_essential_in_header
                )

        final_labels_to_process = file_marker_names_in_order if essential_set is None else \
                                 [lbl for lbl in file_marker_names_in_order if lbl in essential_set]

        df = pd.read_csv(file_path, sep='\t', skiprows=5, header=None, low_memory=False)

        if df.empty and not final_labels_to_process:
             nan_stats = {'nan_count': 0, 'total_points': 0, 'nan_percentage': 0.0}
             return FileReadResult(pd.DataFrame(), nan_stats, ["TRC file is empty and no specific markers requested."])
        if df.empty and final_labels_to_process:
            raise TRCFileReadingError(f"TRC file '{file_path}' contains no data rows, but markers were expected.")


        num_data_cols_in_df = df.shape[1]

        columns_to_select_indices = []
        if num_data_cols_in_df > 0: columns_to_select_indices.append(0) # Frame#
        if num_data_cols_in_df > 1: columns_to_select_indices.append(1) # Time

        successfully_selected_marker_labels_for_multiindex = []
        incomplete_essential_markers_data = []

        for marker_name in final_labels_to_process:
            try:
                marker_k_index = file_marker_names_in_order.index(marker_name)
                base_data_col_idx = 2 + marker_k_index * 3

                marker_data_indices = [base_data_col_idx, base_data_col_idx + 1, base_data_col_idx + 2]

                has_all_components = True
                for i_coord in range(3):
                    if (base_data_col_idx + i_coord) >= num_data_cols_in_df:
                        if essential_set and marker_name in essential_set:
                            incomplete_essential_markers_data.append(
                                f"{marker_name} (Component {['X','Y','Z'][i_coord]} missing; df has {num_data_cols_in_df} cols, expected up to {base_data_col_idx + i_coord})"
                            )
                        has_all_components = False
                        break

                if essential_set and marker_name in essential_set and not has_all_components:
                    continue

                if has_all_components:
                    columns_to_select_indices.extend(marker_data_indices)
                    successfully_selected_marker_labels_for_multiindex.append(marker_name)
                elif not essential_set:
                    partial_indices = [idx for idx in marker_data_indices if idx < num_data_cols_in_df]
                    columns_to_select_indices.extend(partial_indices)
                    if len(partial_indices) == 3 :
                        successfully_selected_marker_labels_for_multiindex.append(marker_name)
                    elif partial_indices:
                         problems_encountered.append(f"Marker '{marker_name}' has incomplete data (loaded {len(partial_indices)}/3 components).")


            except ValueError:
                if essential_set and marker_name in essential_set:
                     incomplete_essential_markers_data.append(f"{marker_name} (Marked essential, but not found in file's marker list after initial check - logical error)")


        if incomplete_essential_markers_data:
            raise EssentialMarkerMissingError(
                f"Essential markers in '{file_path}' have incomplete data columns: {'; '.join(incomplete_essential_markers_data)}",
                missing_markers=incomplete_essential_markers_data
            )

        columns_to_select_indices = sorted(list(set(columns_to_select_indices)))

        if final_labels_to_process and not any(idx >= 2 for idx in columns_to_select_indices):
            raise TRCFileProcessingError(f"No data columns selected for any of the requested markers in '{file_path}'. Processed labels: {final_labels_to_process}. Selected indices: {columns_to_select_indices}")

        if not columns_to_select_indices and final_labels_to_process:
             raise TRCFileProcessingError(f"No data columns (including Frame#/Time) could be selected for '{file_path}' though markers were specified.")

        df_processed = df.iloc[:, [idx for idx in columns_to_select_indices if idx < num_data_cols_in_df]]


        multiindex_tuples = []
        idx_counter_for_marker_data = 0
        if 0 in columns_to_select_indices and df_processed.shape[1] > idx_counter_for_marker_data:
            multiindex_tuples.append(('Frame#', ''))
            idx_counter_for_marker_data +=1
        if 1 in columns_to_select_indices and df_processed.shape[1] > idx_counter_for_marker_data:
            multiindex_tuples.append(('Time', ''))
            idx_counter_for_marker_data +=1

        final_marker_data_columns_for_df = []

        for marker_short_name in successfully_selected_marker_labels_for_multiindex:
            full_marker_name = MARKER_NAME_MAP.get(marker_short_name, marker_short_name)
            for axis in ['X', 'Y', 'Z']:
                multiindex_tuples.append((full_marker_name, axis))
                final_marker_data_columns_for_df.append(df_processed.columns[idx_counter_for_marker_data])
                idx_counter_for_marker_data +=1

        if df_processed.shape[1] != len(multiindex_tuples):
            problems_encountered.append(f"Column count mismatch for MultiIndex: df has {df_processed.shape[1]}, expected {len(multiindex_tuples)}. Using raw column names from read_csv.")
        else:
            df_processed.columns = pd.MultiIndex.from_tuples(multiindex_tuples, names=['Marker', 'Axis'])

        if isinstance(df_processed.columns, pd.MultiIndex):
            marker_data_for_nan_stats = df_processed.xs(key='X', axis=1, level='Axis', drop_level=False)
            for axis_to_add in ['Y', 'Z']:
                 if axis_to_add in df_processed.columns.get_level_values('Axis'):
                    marker_data_for_nan_stats = pd.concat([marker_data_for_nan_stats, df_processed.xs(key=axis_to_add, axis=1, level='Axis', drop_level=False)], axis=1)
        else:
            cols_for_nan = [c for c in df_processed.columns if str(c).lower() not in ['frame#', 'time']]
            if not cols_for_nan and df_processed.shape[1] > 2:
                cols_for_nan = df_processed.columns[2:]
            elif not cols_for_nan and df_processed.shape[1] <=2 and df_processed.shape[1] > 0:
                 cols_for_nan = []
            elif not cols_for_nan and df_processed.shape[1] == 0:
                 cols_for_nan = []
            else:
                 cols_for_nan = df_processed.columns
            marker_data_for_nan_stats = df_processed[cols_for_nan]


        nan_count = 0
        total_points = 0
        if not marker_data_for_nan_stats.empty:
            numeric_marker_data = marker_data_for_nan_stats.apply(pd.to_numeric, errors='coerce')
            nan_count = int(numeric_marker_data.isnull().sum().sum())
            total_points = int(numeric_marker_data.size)

        nan_percentage = (nan_count / total_points) if total_points > 0 else 0.0
        nan_stats = {'nan_count': nan_count, 'total_points': total_points, 'nan_percentage': nan_percentage}

        if isinstance(df_processed.columns, pd.MultiIndex):
            df_processed.columns = ['_'.join(col).strip('_') for col in df_processed.columns.values]

        df_final_markers = df_processed.drop(columns=['Frame#_', 'Time_'], errors='ignore')

        # Ensure numeric marker matrix for downstream scaling/normalization
        df_final_markers = df_final_markers.apply(pd.to_numeric, errors='coerce')

        # Unit harmonization for marker trajectories
        # Keep marker coordinates in millimeters by default.
        # Override with env TRC_MARKER_UNITS_MODE in {"auto", "mm", "m"}.
        trc_units_mode = os.getenv("TRC_MARKER_UNITS_MODE", "mm").strip().lower()
        if trc_units_mode not in {"auto", "mm", "m"}:
            trc_units_mode = "mm"

        if trc_units_mode == "mm":
            marker_scale = 1.0
            resolved_units = "mm"
        elif trc_units_mode == "m":
            marker_scale = 0.001
            resolved_units = "m->mm"
        else:
            # Auto: output in millimeters. Trust declared units first; fallback to magnitude heuristic.
            if declared_units == 'mm':
                marker_scale = 1.0
                resolved_units = 'auto(header:mm)'
            elif declared_units in {'m', 'meter', 'meters'}:
                marker_scale = 1000.0
                resolved_units = 'auto(header:m->mm)'
            else:
                numeric_values = df_final_markers.values
                finite = numeric_values[np.isfinite(numeric_values)]
                abs_p95 = float(np.percentile(np.abs(finite), 95)) if finite.size else 0.0
                if abs_p95 > 10.0:
                    marker_scale = 1.0
                    resolved_units = 'auto(heuristic:mm)'
                else:
                    marker_scale = 1000.0
                    resolved_units = 'auto(heuristic:m->mm)'

        if marker_scale != 1.0:
            df_final_markers = df_final_markers * marker_scale

        # Factor that converts the returned marker values to millimetres. Consumers that mix marker
        # positions with quantities in fixed units (the CoP transform works in mm) need it, because
        # mode "mm" keeps metre files in metres. Unknown declared units: "m" mode means the file is in
        # metres, an auto x1000 means it was in metres, otherwise assume millimetres.
        declared_to_mm = {'mm': 1.0, 'm': 1000.0, 'meter': 1000.0, 'meters': 1000.0}.get(declared_units)
        if declared_to_mm is None:
            declared_to_mm = 1000.0 if (trc_units_mode == "m" or marker_scale == 1000.0) else 1.0
        nan_stats['marker_units_to_mm'] = declared_to_mm / marker_scale

        global _PRINTED_TRC_UNIT_MSG
        if not _PRINTED_TRC_UNIT_MSG:
            print(
                f"TRC marker units: mode={trc_units_mode}, declared={declared_units}, "
                f"resolved={resolved_units}, scale={marker_scale}"
            )
            _PRINTED_TRC_UNIT_MSG = True

        df_final_markers = df_final_markers.reset_index(drop=True)
        df_final_markers = df_final_markers.fillna(0.0)

        return FileReadResult(df_final_markers, nan_stats, problems_encountered)

    except FileNotFoundError:
        raise TRCFileReadingError(f"TRC file not found at path: {file_path}")
    except pd.errors.EmptyDataError:
        nan_stats = {'nan_count': 0, 'total_points': 0, 'nan_percentage': 0.0}
        return FileReadResult(pd.DataFrame(), nan_stats, [f"TRC file '{file_path}' is empty or contains no parsable data rows."])
    except TRCFileProcessingError as e:
        raise e
    except Exception as e:
        raise TRCFileProcessingError(f"An unexpected error occurred while processing TRC file {file_path}: {e}")


def read_mot_file(file_path) -> FileReadResult:
    problems = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        header_end_line = 0
        for i, line in enumerate(lines):
            if "endheader" in line.lower():
                header_end_line = i
                break
        else:
            raise TRCFileReadingError(f"'endheader' not found in MOT file: {file_path}")


        header_line_index = header_end_line + 1
        if header_line_index >= len(lines):
            raise TRCFileReadingError(f"No header line found after 'endheader' in MOT file: {file_path}")

        header = lines[header_line_index].strip().split('\t')

        data_skip_rows = header_end_line + 2
        if data_skip_rows < len(lines) and not lines[data_skip_rows].strip():
            data_skip_rows +=1

        try:
            data = pd.read_csv(file_path, delimiter='\t', skiprows=data_skip_rows, names=header, low_memory=False)
        except Exception as e:
            raise TRCFileReadingError(f"Error reading data from MOT file {file_path} using pandas: {e}")

        if data.empty:
            problems.append(f"Warning: read_mot_file for {file_path} resulted in an empty DataFrame after header processing.")

        return FileReadResult(data, {}, problems)

    except FileNotFoundError:
        raise TRCFileReadingError(f"MOT file not found at path: {file_path}")
    except Exception as e:
        raise TRCFileReadingError(f"Error processing MOT file {file_path}: {e}")


def read_sto_file(file_path) -> FileReadResult:
    """Reads an STO file into a Pandas DataFrame, handling header complexities."""
    problems = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        data_start = 0
        header_names = []
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('endheader'):
                if i + 1 < len(lines):
                    header_names = lines[i+1].strip().split('\t')
                    data_start = i + 2
                else:
                    raise TRCFileReadingError(f"No header line found after 'endheader' in STO file: {file_path}")
                break
        else:
            for i, line in enumerate(lines):
                if line.strip().lower().startswith('time'):
                    header_names = line.strip().split('\t')
                    data_start = i + 1
                    break
            else:
                raise TRCFileReadingError(f"Could not find 'endheader' or a 'time'-prefixed header line in STO file: {file_path}")

        if not header_names:
             raise TRCFileReadingError(f"Header names could not be determined for STO file: {file_path}")

        df = pd.read_csv(file_path, sep=r'\s+', skiprows=data_start, header=None)

        if df.empty:
            return FileReadResult(pd.DataFrame(columns=header_names if header_names else None), {}, [])


        if len(df.columns) == len(header_names):
            df.columns = header_names
        else:
            df.columns = [f'col_{i}' for i in range(len(df.columns))]

        if 'time' in df.columns:
            df = df.drop(columns=['time'])
        elif df.shape[1] > 0 and 'time' not in [str(c).lower() for c in df.columns]:
            df = df.iloc[:, 1:]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return FileReadResult(df, {}, problems)

    except FileNotFoundError:
        raise TRCFileReadingError(f"STO file not found at path: {file_path}")
    except Exception as e:
        raise TRCFileReadingError(f"Error processing STO file {file_path}: {str(e)}")
