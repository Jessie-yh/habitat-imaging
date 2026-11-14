"""Case discovery and NIfTI / CSV I-O helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Case:
    """One patient: a tumour ROI plus one or more parameter sources."""

    case_id: str
    roi_path: Path
    maps: dict[str, Path] = field(default_factory=dict)
    series_path: Path | None = None


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def _fmt(pattern: str, case_dir: Path) -> Path:
    return Path(pattern.replace("{case}", str(case_dir)).replace("{case_id}", case_dir.name))


@dataclass
class Discovery:
    """Result of scanning the dataset root."""

    cases: list[Case] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)


def scan_cases(cfg: dict[str, Any]) -> Discovery:
    """Resolve every case directory, recording why unusable cases were skipped.

    Missing files are reported and the case is skipped -- a partially populated
    export directory is the normal situation in retrospective cohorts.
    """
    root = Path(cfg["dataset"]["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    case_glob = cfg["dataset"].get("case_glob") or "*"
    case_dirs = sorted(p for p in root.glob(case_glob) if p.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"no case directories matched {root}/{case_glob}")

    roi_pattern = cfg["dataset"]["roi_pattern"]
    series_pattern = cfg["dataset"].get("series_pattern")
    map_specs = cfg["dataset"].get("maps") or []

    report = Discovery()

    for case_dir in case_dirs:
        roi = _fmt(roi_pattern, case_dir)
        if not roi.is_file():
            report.skipped.append({"case_id": case_dir.name, "stage": "discovery",
                                   "error": f"roi missing: {roi}"})
            continue

        case = Case(case_id=case_dir.name, roi_path=roi)
        reasons: list[str] = []

        if series_pattern:
            series = _fmt(series_pattern, case_dir)
            if not series.is_file():
                reasons.append(f"series missing: {series}")
            else:
                case.series_path = series

        for spec in map_specs:
            path = _fmt(spec["pattern"], case_dir)
            if not path.is_file():
                reasons.append(f"map '{spec['name']}' missing: {path}")
            else:
                case.maps[spec["name"]] = path

        if reasons:
            report.skipped.append({"case_id": case_dir.name, "stage": "discovery",
                                   "error": "; ".join(reasons)})
        else:
            report.cases.append(case)

    for item in report.skipped:
        log.warning("skip %s: %s", item["case_id"], item["error"])
    if not report.cases:
        raise RuntimeError("no usable cases after file discovery; see warnings above")
    log.info("discovered %d usable case(s), %d skipped",
             len(report.cases), len(report.skipped))
    return report


def discover_cases(cfg: dict[str, Any]) -> list[Case]:
    """Usable cases only (thin wrapper around :func:`scan_cases`)."""
    return scan_cases(cfg).cases


# --------------------------------------------------------------------------- #
# NIfTI
# --------------------------------------------------------------------------- #

def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, Any]:
    """Load a NIfTI file as float32 data + affine + header."""
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    return data, img.affine, img.header


def save_nifti(path: str | Path, data: np.ndarray, affine: np.ndarray,
               header: Any | None = None, dtype: Any = np.float32) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data, dtype=dtype), affine, header)
    img.set_data_dtype(dtype)
    nib.save(img, str(path))
    return path


def voxel_volume_mm3(affine: np.ndarray) -> float:
    """Voxel volume in mm^3 from the affine (works with non-orthogonal affines)."""
    return float(abs(np.linalg.det(affine[:3, :3])))


def temporal_resolution(header: Any, fallback: float | None = None) -> float:
    """Repetition time in seconds, from ``pixdim[4]`` when available."""
    try:
        tr = float(header.get_zooms()[3])
    except Exception:  # pragma: no cover - header without 4th dim
        tr = 0.0
    if not np.isfinite(tr) or tr <= 0:
        if fallback is None:
            raise ValueError(
                "4D DCE series has no valid TR (pixdim[4]); set dce.temporal_resolution_s"
            )
        return float(fallback)
    # NIfTI stores TR in seconds in most DICOM->NIfTI converters, but some
    # tools write milliseconds.  Values > 100 are treated as milliseconds.
    if tr > 100.0:
        tr /= 1000.0
    return tr


# --------------------------------------------------------------------------- #
# Tabular output
# --------------------------------------------------------------------------- #

def write_csv(df: pd.DataFrame, path: str | Path, encoding: str = "utf-8-sig",
              also_gbk: bool = False, float_format: str = "%.6g") -> list[Path]:
    """Write a CSV; optionally emit a GBK twin for Excel on Chinese Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=encoding, float_format=float_format)
    written = [path]
    if also_gbk and encoding.lower().replace("-", "") not in {"gbk", "gb2312", "gb18030"}:
        gbk_path = path.with_name(f"{path.stem}_gbk{path.suffix}")
        try:
            df.to_csv(gbk_path, index=False, encoding="gbk", float_format=float_format)
            written.append(gbk_path)
        except UnicodeEncodeError:  # non-GBK characters (e.g. Greek letters)
            log.warning("cannot encode %s as GBK; skipped", path.name)
    return written


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
    return path


def concat_rows(rows: Iterable[dict[str, Any]], columns: Sequence[str] | None = None) -> pd.DataFrame:
    rows = list(rows)
    df = pd.DataFrame(rows)
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[list(columns) + [c for c in df.columns if c not in columns]]
    return df
