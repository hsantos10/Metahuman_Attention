"""Project-wide constants: marker sets, target columns, and processing config."""

from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class ProcessingConfig:
    """Configuration for processing a single trial."""
    trc_filepath: str
    data_filepath: str
    use_metadata: bool
    metadata_df: Any # pandas DataFrame
    max_length: int
    data_columns_to_use: List[str]
    split_name: str
    trc_file_prefix: str
    trc_file_suffix: str
    subject_id_col_in_metadata: str
    weight_col_in_metadata: str
    height_col_in_metadata: str
    canonical_marker_columns: List[str]
    essential_marker_short_names_list: List[str]
    nan_threshold: float = 0.10
    apply_coordinate_corrections: bool = True
    strict_target_columns: bool = False
    enable_data_diagnostics: bool = False
    force_target_mode: str = "newton"
    enable_biomech_preprocessing: bool = True  # translational/rotational invariance + height scaling

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

ESSENTIAL_MARKER_SHORT_NAMES_WIND_TURBINE = [
    'LPH', 'RPH', 'LSHO', 'RSHO', 'MChest', 'LUPA', 'LELBL', 'LELBM',
    'LFRM', 'LWRM', 'RUPA', 'RELBL', 'RELBM', 'RFRM', 'RWRL', 'RWRM',
    'LASI', 'LPSI', 'RPSI', 'LTHI', 'LTOE', 'LHEE', 'LANKL', 'LANKM',
    'LTIB', 'RTHI', 'RTOE', 'RHEE', 'RANKL', 'RANKM', 'RTIB', 'RKNEM',
    'C7', 'LWRL', 'RKNEL', 'LKNEM', 'RASI', 'LKNEL', 'RAH', 'LAH'
]

ESSENTIAL_MARKER_SHORT_NAMES_PATIENT_HANDLING = [
    'LSHO', 'RSHO', 'LELBL', 'LELBM', 'LWRL', 'LWRM', 'RELBL', 'RELBM',
    'RWRM', 'LKNE', 'RKNE', 'LTOE', 'LHEE', 'LANKL', 'RTOE',
    'RHEE', 'RANKM', 'LASI', 'LANKM', 'RANKL'
]

ESSENTIAL_MARKER_SHORT_NAMES_GROUNDLINK = [
    'LSHO', 'RSHO', 'LELBL', 'LELBM', 'LWRL', 'LWRM', 'RELBL', 'RELBM',
    'RWRL', 'RWRM', 'LKNE', 'RKNE', 'LTOE', 'LHEE', 'LANKL', 'RTOE',
    'RHEE', 'RANKM', 'LASI', 'LPSI', 'RPSI', 'RASI', 'LANKM', 'RANKL'
]

# Intersection of GroundLink and Patient Handling essential marker sets.
# Used for COMBINED multi-lab training so the feature space is identical
# regardless of which lab's data a trial comes from.
ESSENTIAL_MARKER_SHORT_NAMES_COMBINED = [
    'LSHO', 'RSHO', 'LELBL', 'LELBM', 'LWRL', 'LWRM', 'RELBL', 'RELBM',
    'RWRM', 'LKNE', 'RKNE', 'LTOE', 'LHEE', 'LANKL', 'RTOE',
    'RHEE', 'RANKM', 'LASI', 'LANKM', 'RANKL'
]

# Left-right marker short name pairs used for bilateral flip augmentation.
# Flipping mirrors the ML axis (X), swaps these pairs, and also swaps
# force plate 1 ↔ 2 while negating _ml force components.
LEFT_RIGHT_MARKER_PAIRS = [
    ('LSHO',  'RSHO'),
    ('LELBL', 'RELBL'),
    ('LELBM', 'RELBM'),
    ('LWRL',  'RWRL'),
    ('LWRM',  'RWRM'),
    ('LKNE',  'RKNE'),
    ('LANKL', 'RANKL'),
    ('LANKM', 'RANKM'),
    ('LTOE',  'RTOE'),
    ('LHEE',  'RHEE'),
    ('LASI',  'RASI'),
    ('LPSI',  'RPSI'),
    # 40-marker wind-turbine set (WT_MARKER_SET=full40); without these, 16 markers were mirrored but not
    # swapped and ended up on the opposite limb in flipped training samples (found 2026-10-01)
    ('LPH',   'RPH'),
    ('LAH',   'RAH'),
    ('LUPA',  'RUPA'),
    ('LFRM',  'RFRM'),
    ('LTHI',  'RTHI'),
    ('LTIB',  'RTIB'),
    ('LKNEL', 'RKNEL'),
    ('LKNEM', 'RKNEM'),
]


# ── Target force columns (10 GRF channels) ──────────────────────────────

TARGET_COLUMNS = [
    "ground_force_1_ml", "ground_force_1_vert", "ground_force_1_ap",
    "ground_force_1_px", "ground_force_1_pz",
    "ground_force_2_ml", "ground_force_2_vert", "ground_force_2_ap",
    "ground_force_2_px", "ground_force_2_pz",
]

# ── Human-readable short names for force features ───────────────────────

FEATURE_SHORT_NAMES = {
    "ground_force_1_ml": "F1 ML",
    "ground_force_1_vert": "F1 Vert",
    "ground_force_1_ap": "F1 AP",
    "ground_force_1_px": "F1 CoP-X",
    "ground_force_1_pz": "F1 CoP-Z",
    "ground_force_2_ml": "F2 ML",
    "ground_force_2_vert": "F2 Vert",
    "ground_force_2_ap": "F2 AP",
    "ground_force_2_px": "F2 CoP-X",
    "ground_force_2_pz": "F2 CoP-Z",
    # plate-sum layout (TARGET_LAYOUT=plate_sum): summed force, vertical-force-weighted combined CoP
    "ground_force_s_ml": "Fs ML",
    "ground_force_s_vert": "Fs Vert",
    "ground_force_s_ap": "Fs AP",
    "ground_force_s_px": "Fs CoP-X",
    "ground_force_s_pz": "Fs CoP-Z",
}

FEATURE_SHORT_NAMES_LIST = list(FEATURE_SHORT_NAMES.values())
