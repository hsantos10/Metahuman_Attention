"""One adapter per dataset: pairs marker files with force files, applies subject/task filters, and loads
subject mass/height. GroundLink (C3D + .npy), Patient Handling (TRC + MOT), WindTurbine (TRC + MOT,
Chapter 2) and Combined (GL + PH with namespaced subject IDs). `get_dataset_adapter(name)` is the entry point."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import re

import pandas as pd

import os

from src.config import (
    ESSENTIAL_MARKER_SHORT_NAMES_COMBINED,
    ESSENTIAL_MARKER_SHORT_NAMES_WIND_TURBINE,
)
from src.utils.utils import load_metadata

# Data roots, overridable so the same code runs on the workstation and on HPRC.
TAMU_DATA_ROOT = Path(os.getenv("TAMU_DATA_ROOT", "e:/data/TAMU Data"))
GROUNDLINK_ROOT = Path(os.getenv("GROUNDLINK_ROOT", "e:/research/GroundLink"))


@dataclass
class AdapterMetadataConfig:
    metadata_path: Optional[Path]
    weight_col: Optional[str]
    height_col: Optional[str]
    subject_id_col: Optional[str]


@dataclass
class AdapterResult:
    trc_files: List[str]
    data_files: List[str]


class BaseDatasetAdapter:
    name: str
    essential_markers: List[str]
    use_metadata: bool

    def pair_files(self) -> AdapterResult:
        raise NotImplementedError

    def load_metadata(self) -> Tuple[Optional[pd.DataFrame], AdapterMetadataConfig]:
        return None, AdapterMetadataConfig(None, None, None, None)


class GroundLinkAdapter(BaseDatasetAdapter):
    # Subject codes to exclude from all training/evaluation (data quality).
    # "s001" = GL subject 1 — confirmed systemic force anomalies (ML bias,
    # AP sign flips, CoP offset) across all trial types.
    EXCLUDED_SUBJECTS: List[str] = ["s001"]

    def __init__(self, dataset_root: Optional[str] = None):
        self.name = "GROUNDLINK"
        self.essential_markers = ESSENTIAL_MARKER_SHORT_NAMES_COMBINED
        self.use_metadata = True

        base = Path(dataset_root) if dataset_root else GROUNDLINK_ROOT
        self.c3d_root = base / "c3d" / "c3d"
        self.force_root = base / "force" / "force"
        self.metadata_path = Path("data/groundlink_metadata.csv")

    @staticmethod
    def _normalize_subject_id(raw: str) -> str:
        match = re.search(r"s(\d{3})", raw.lower())
        if match:
            return str(int(match.group(1)))
        digits = re.search(r"(\d+)", raw)
        return str(int(digits.group(1))) if digits else raw

    def pair_files(self) -> AdapterResult:
        trc_files = sorted(str(p) for p in self.c3d_root.glob("*/*.c3d"))
        paired_trc, paired_force = [], []
        excluded_count = 0

        for trc_path in trc_files:
            trc_path_obj = Path(trc_path)
            match = re.search(r"(s\d{3})", trc_path_obj.name)
            if not match:
                continue

            subject_code = match.group(1)

            # Skip excluded subjects (data quality filter)
            if subject_code in self.EXCLUDED_SUBJECTS:
                excluded_count += 1
                continue

            force_path = self.force_root / f"{subject_code}_force" / subject_code / f"{trc_path_obj.stem}.npy"

            if force_path.is_file():
                paired_trc.append(trc_path)
                paired_force.append(str(force_path))

        if excluded_count > 0:
            print(f"  [GroundLinkAdapter] Excluded {excluded_count} trials from subjects "
                  f"{self.EXCLUDED_SUBJECTS} (data quality filter)")

        return AdapterResult(trc_files=paired_trc, data_files=paired_force)

    def load_metadata(self) -> Tuple[Optional[pd.DataFrame], AdapterMetadataConfig]:
        if not self.metadata_path.exists():
            return None, AdapterMetadataConfig(None, None, None, None)

        df = pd.read_csv(self.metadata_path)
        if "Participant ID" not in df.columns:
            return None, AdapterMetadataConfig(None, None, None, None)

        df["participant_key"] = df["Participant ID"].apply(self._normalize_subject_id)
        df = df.set_index("participant_key")

        return df, AdapterMetadataConfig(
            metadata_path=self.metadata_path,
            weight_col="Mass (kg)",
            height_col="Height (m)",
            subject_id_col="participant_key",
        )


class PatientHandlingAdapter(BaseDatasetAdapter):
    # Subject directories to exclude (data quality / plate-swap issues).
    # Values are directory names like "subj13", "subj14".
    EXCLUDED_SUBJECTS: List[str] = []

    def __init__(self, dataset_root: Optional[str] = None):
        self.name = "PATIENT_HANDLING"
        self.essential_markers = ESSENTIAL_MARKER_SHORT_NAMES_COMBINED
        self.use_metadata = True

        base = Path(dataset_root) if dataset_root else TAMU_DATA_ROOT / "01.PatientHandlingTasks" / "03.OpenSim"
        self.base_path = base
        self.metadata_path = TAMU_DATA_ROOT / "01.PatientHandlingTasks" / "Experiment Detail.xlsx"
        self.force_subdir = "ForceFile(50Hz)"

    def pair_files(self) -> AdapterResult:
        trc_files = sorted(str(p) for p in self.base_path.glob("subj*/MarkerData/*.trc"))
        paired_trc, paired_force = [], []
        excluded_count = 0

        for trc_path in trc_files:
            trc_path_obj = Path(trc_path)
            # Check if this trial belongs to an excluded subject
            subj_dir = trc_path_obj.parent.parent.name  # e.g. "subj13"
            if subj_dir in self.EXCLUDED_SUBJECTS:
                excluded_count += 1
                continue

            expected_force = trc_path_obj.parent.parent / "ForceData" / self.force_subdir / f"{trc_path_obj.stem}_Resampled.mot"
            if expected_force.is_file():
                paired_trc.append(trc_path)
                paired_force.append(str(expected_force))

        if excluded_count > 0:
            print(f"  [PatientHandlingAdapter] Excluded {excluded_count} trials from subjects "
                  f"{self.EXCLUDED_SUBJECTS} (data quality filter)")

        return AdapterResult(trc_files=paired_trc, data_files=paired_force)

    def load_metadata(self) -> Tuple[Optional[pd.DataFrame], AdapterMetadataConfig]:
        df = load_metadata(str(self.metadata_path), "Subject", "Weight", "Height")
        return df, AdapterMetadataConfig(
            metadata_path=self.metadata_path,
            weight_col="Weight/kg",
            height_col="Height/m",
            subject_id_col="Subject",
        )


class WindTurbineAdapter(BaseDatasetAdapter):
    """Wind-turbine maintenance captures (same laboratory and OpenSim convention as Patient Handling).

    Layout: <root>/subjXX/MarkerData/<trial>.trc (44 markers incl. 4 box markers, metres, 50 Hz) and
    <root>/subjXX/ForceData/ForceFile(50Hz)/<trial>_Resampled.mot written by tools/wt_prepare_force.py.
    Trials whose task group is 'exclude' in data/wind_turbine_trials.csv (sync, poses, unlabeled) are
    dropped; WT_TASK_GROUPS (comma list of direct/ladder) restricts further, WT_TASKS to task names,
    WT_SUBJECTS to subject folders (e.g. the first six: subj01,subj02,subj03,subj14,subj24,subj25).
    Marker set: WT_MARKER_SET=common20 (default; Chapter 1 set with LKNE->LKNEL, RKNE->RKNEL aliases)
    or full40. Sets TRC_MARKER_UNITS_MODE=m and TRC_MARKER_ALIASES unless already set.
    """
    EXCLUDED_SUBJECTS: List[str] = []

    def __init__(self, dataset_root: Optional[str] = None):
        self.name = "WIND_TURBINE"
        marker_set = os.getenv("WT_MARKER_SET", "common20").strip().lower()
        self.essential_markers = (ESSENTIAL_MARKER_SHORT_NAMES_WIND_TURBINE if marker_set == "full40"
                                  else ESSENTIAL_MARKER_SHORT_NAMES_COMBINED)
        self.use_metadata = True
        base = Path(dataset_root) if dataset_root else TAMU_DATA_ROOT / "02.WindTurbineMaintenanceTasks" / "03.OpenSim"
        self.base_path = base
        self.metadata_path = TAMU_DATA_ROOT / "02.WindTurbineMaintenanceTasks" / "Detailed Experimental Data (1).xlsx"
        self.force_subdir = "ForceFile(50Hz)"
        self.trials_csv = Path(__file__).resolve().parents[2] / "data" / "wind_turbine_trials.csv"
        os.environ.setdefault("TRC_MARKER_UNITS_MODE", "m")
        if marker_set != "full40":
            os.environ.setdefault("TRC_MARKER_ALIASES", "LKNE:LKNEL,RKNE:RKNEL")

    def _allowed_trials(self):
        """(subject, trial) -> True for trials that enter the run; None if no trials csv."""
        if not self.trials_csv.is_file():
            return None
        df = pd.read_csv(self.trials_csv)
        groups = [g for g in os.getenv("WT_TASK_GROUPS", "direct,ladder").split(",") if g.strip()]
        tasks = [t for t in os.getenv("WT_TASKS", "").split(",") if t.strip()]
        keep = df[df["group"].isin(groups)]
        if tasks:
            keep = keep[keep["task"].isin(tasks)]
        return {(r.subject, str(r.trial)) for r in keep.itertuples()}

    def _header_markers(self, trc_path: Path):
        """Marker labels in a TRC header, with TRC_MARKER_ALIASES applied (canonical name if the file
        carries the aliased label instead)."""
        try:
            with open(trc_path, "r", encoding="utf-8", errors="replace") as f:
                for _ in range(3): f.readline()
                labels = [x.strip() for x in f.readline().split("\t") if x.strip()]
        except Exception:
            return set()
        names = set(labels)
        for pair in os.getenv("TRC_MARKER_ALIASES", "").split(","):
            if ":" in pair:
                want, have = [s.strip() for s in pair.split(":", 1)]
                if want not in names and have in names:
                    names.add(want)
        return names

    def pair_files(self) -> AdapterResult:
        trc_files = sorted(str(p) for p in self.base_path.glob("subj*/MarkerData/*.trc"))
        allowed = self._allowed_trials()
        essential = set(self.essential_markers)
        paired_trc, paired_force = [], []
        excluded_subj = excluded_task = 0
        missing_markers = {}
        # WT_SUBJECTS (comma list of subject folders, e.g. subj01,subj02): restrict to those subjects
        only = {s.strip() for s in os.getenv("WT_SUBJECTS", "").split(",") if s.strip()}
        for trc_path in trc_files:
            p = Path(trc_path)
            subj_dir = p.parent.parent.name
            if subj_dir in self.EXCLUDED_SUBJECTS or (only and subj_dir not in only):
                excluded_subj += 1; continue
            if allowed is not None and (subj_dir, p.stem) not in allowed:
                excluded_task += 1; continue
            expected_force = p.parent.parent / "ForceData" / self.force_subdir / f"{p.stem}_Resampled.mot"
            if not expected_force.is_file():
                continue
            # Skip trials whose export lacks an essential marker (e.g. subj24: RHEE missing in 22 trials),
            # so the paired list matches what the loader will keep and per-trial bookkeeping stays aligned.
            miss = essential - self._header_markers(p)
            if miss:
                missing_markers[subj_dir] = missing_markers.get(subj_dir, 0) + 1
                continue
            paired_trc.append(trc_path); paired_force.append(str(expected_force))
        print(f"  [WindTurbineAdapter] {len(paired_trc)} trials paired; excluded {excluded_subj} by subject, "
              f"{excluded_task} by task filter (groups={os.getenv('WT_TASK_GROUPS', 'direct,ladder')}, "
              f"tasks={os.getenv('WT_TASKS', '') or 'all'}), {sum(missing_markers.values())} for missing essential "
              f"markers {missing_markers or ''}; marker set {os.getenv('WT_MARKER_SET', 'common20')}")
        return AdapterResult(trc_files=paired_trc, data_files=paired_force)

    def load_metadata(self) -> Tuple[Optional[pd.DataFrame], AdapterMetadataConfig]:
        df = load_metadata(str(self.metadata_path), "Subject", "Weight", "Height", sheet_name="Design")
        return df, AdapterMetadataConfig(
            metadata_path=self.metadata_path,
            weight_col="Weight/kg",
            height_col="Height/m",
            subject_id_col="Subject",
        )


class AddBioAdapter(WindTurbineAdapter):
    """AddBiomechanics studies converted to the wind-turbine OpenSim layout by chapter2/tools/addbio_to_opensim.py
    (Chapter 2, Part B). Root = $ADDBIO_OSIM_ROOT/$ADDBIO_STUDY with subjNN/MarkerData/<trial>.trc (standard marker
    names, metres, 50 Hz, AddBiomechanics world frame, Y up) and subjNN/ForceData/ForceFile(50Hz)/<trial>_Resampled.mot
    (plate 1 = right foot, plate 2 = left foot). metadata.csv and trials.csv sit in the study root. Markers and forces
    share one frame, so no sign corrections apply (configs/addbio.yaml: coordinate_corrections false) and the force
    co-rotation treats model X as the ML label (FORCE_MODEL_X_IS_ML=1, set by the Chapter 2 driver).
    """

    def __init__(self, dataset_root: Optional[str] = None):
        self.name = "ADDBIO"
        self.essential_markers = ESSENTIAL_MARKER_SHORT_NAMES_COMBINED
        self.use_metadata = True
        study = os.environ.get("ADDBIO_STUDY", "Han2023")
        root = Path(dataset_root or os.environ.get("ADDBIO_OSIM_ROOT", "e:/data/AddBio_osim")) / study
        # ADDBIO_EXCLUDE: comma list of subject folders to drop. Default for Han2023 (= GroundLink): subj01, the
        # subject our GroundLink runs exclude for force anomalies, so the two GroundLink paths stay comparable.
        self.EXCLUDED_SUBJECTS = [s.strip() for s in os.environ.get("ADDBIO_EXCLUDE", "subj01" if study == "Han2023" else "").split(",") if s.strip()]
        self.base_path = root
        self.metadata_path = root / "metadata.csv"
        self.force_subdir = "ForceFile(50Hz)"
        self.trials_csv = root / "trials.csv"
        os.environ.setdefault("TRC_MARKER_UNITS_MODE", "m")

    def load_metadata(self) -> Tuple[Optional[pd.DataFrame], AdapterMetadataConfig]:
        df = load_metadata(str(self.metadata_path), "Subject", "Weight", "Height")
        return df, AdapterMetadataConfig(metadata_path=self.metadata_path, weight_col="Weight/kg",
                                         height_col="Height/m", subject_id_col="Subject")


def get_dataset_adapter(name: str) -> BaseDatasetAdapter:
    name_upper = name.upper()
    if name_upper == "GROUNDLINK":
        return GroundLinkAdapter()
    if name_upper == "PATIENT_HANDLING":
        return PatientHandlingAdapter()
    if name_upper == "WIND_TURBINE":
        return WindTurbineAdapter()
    if name_upper == "ADDBIO":
        return AddBioAdapter()
    if name_upper == "COMBINED":
        return CombinedAdapter()
    raise ValueError(f"Unsupported dataset {name}. Supported: GROUNDLINK, PATIENT_HANDLING, WIND_TURBINE, ADDBIO, COMBINED.")


class CombinedAdapter(BaseDatasetAdapter):
    """
    Combines GroundLink (7 subjects, lab GL) and Patient Handling (12 subjects,
    lab PH) into a single dataset for cross-lab / LOSO training.

    Subject IDs are namespaced: gl_1 … gl_7  and  ph_1 … ph_14 so the LOSO
    split can hold out one subject from either lab without ID collisions.

    The feature space is restricted to the 20-marker intersection of both
    labs' marker sets (ESSENTIAL_MARKER_SHORT_NAMES_COMBINED).
    """

    def __init__(self):
        self.name = "COMBINED"
        self.essential_markers = ESSENTIAL_MARKER_SHORT_NAMES_COMBINED
        self.use_metadata = True
        self._gl = GroundLinkAdapter()
        self._ph = PatientHandlingAdapter()

    def pair_files(self) -> AdapterResult:
        gl_result = self._gl.pair_files()
        ph_result = self._ph.pair_files()
        return AdapterResult(
            trc_files  = gl_result.trc_files  + ph_result.trc_files,
            data_files = gl_result.data_files + ph_result.data_files,
        )

    def load_metadata(self) -> Tuple[Optional[pd.DataFrame], AdapterMetadataConfig]:
        """
        Merge both metadata tables.  Subject indices are namespaced with 'gl_'
        and 'ph_' prefixes so the pipeline can look up each trial's metadata
        regardless of which lab it came from.
        """
        import re

        gl_df, gl_cfg = self._gl.load_metadata()
        ph_df, ph_cfg = self._ph.load_metadata()

        frames = []

        if gl_df is not None:
            gl_renamed = gl_df.copy()
            gl_renamed.index = [f"gl_{idx}" for idx in gl_df.index]
            gl_renamed = gl_renamed.rename(columns={
                gl_cfg.weight_col: "Mass (kg)",
                gl_cfg.height_col: "Height (m)",
            })
            frames.append(gl_renamed[["Mass (kg)", "Height (m)"]])

        if ph_df is not None:
            ph_renamed = ph_df.copy()
            ph_renamed.index = [f"ph_{idx}" for idx in ph_df.index]
            ph_renamed = ph_renamed.rename(columns={
                ph_cfg.weight_col: "Mass (kg)",
                ph_cfg.height_col: "Height (m)",
            })
            frames.append(ph_renamed[["Mass (kg)", "Height (m)"]])

        if not frames:
            return None, AdapterMetadataConfig(None, None, None, None)

        combined_df = pd.concat(frames)

        return combined_df, AdapterMetadataConfig(
            metadata_path=None,
            weight_col="Mass (kg)",
            height_col="Height (m)",
            subject_id_col="participant_key",
        )