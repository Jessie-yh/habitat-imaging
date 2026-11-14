"""Configuration loading and validation.

The pipeline is driven by a single YAML file (see ``configs/``).  Every key has
a documented default so that a minimal config only needs to describe the data
layout.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULTS: dict[str, Any] = {
    "project": {
        "name": "cervical_habitat",
        "seed": 42,
        "output_dir": "results/run",
    },
    "dataset": {
        "root": "data/raw",
        "case_glob": "*",
        "roi_pattern": "{case}/roi.nii.gz",
        # Exactly one of ``series`` / ``maps`` must be usable per run.
        "series_pattern": None,
        "maps": [],
        # DWI/IVIM acquisition (only needed for the optional IVIM fit).
        "ivim_bvalues": None,
    },
    "resample": {
        "enabled": True,
        "target_spacing": [1.0, 1.0, 3.0],
        "order_param": 3,
        "order_mask": 0,
    },
    "preprocessing": {
        "percentile_clip": [1.0, 99.0],
        "minmax_scale": True,
        "fill_nonfinite": "roi_median",  # roi_median | zero | drop
        "min_roi_voxels": 30,
    },
    "dce": {
        # Used only when dataset.series_pattern is given.
        "baseline_frames": 3,
        "temporal_resolution_s": None,  # None -> read from NIfTI header
        "time_unit": "min",  # unit of the reported rate constants: min | s
        "smooth_frames_sigma": 0.0,
        "robust_peak": False,
    },
    "ivim": {
        "fit": False,  # True only if you supply a 4D DWI series
        "b_value_threshold": 200.0,
        "d_bounds": [0.0, 0.005],
        "dstar_bounds": [0.005, 0.5],
        "f_bounds": [0.0, 1.0],
    },
    "clustering": {
        "algorithm": "kmeans",
        "n_clusters": None,  # None -> use ``selection`` on k_range
        "k_range": [2, 10],
        "selection": "mode",  # mode | elbow | silhouette | ch | db
        "evaluate": True,  # always report the k-selection curves
        "n_init": 20,
        "init": "k-means++",
        "max_samples_for_eval": 50000,
        "max_voxels_for_fit": 300000,
        "feature_space": "normalized",  # normalized | raw
    },
    "metrics": {
        "volume": True,
        "parameter_stats": True,
        "msi": True,
        "msi_matrix_long": False,
        "ithscore": False,
    },
    "output": {
        "csv_encoding": "utf-8-sig",
        "also_gbk": False,
        "save_plots": True,
        "save_habitat_maps": True,
        "save_normalized_maps": False,
        "max_case_figures": 20,
    },
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, merged over :data:`DEFAULTS`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    cfg = _deep_merge(DEFAULTS, raw)
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_sha256"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    _validate(cfg)
    return cfg


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _validate(cfg: dict[str, Any]) -> None:
    ds = cfg["dataset"]
    _require(bool(ds.get("series_pattern")) or bool(ds.get("maps")),
             "dataset: provide either 'series_pattern' (4D DCE) or a non-empty 'maps' list")

    if ds.get("maps"):
        names = [m.get("name") for m in ds["maps"]]
        _require(len(names) >= 2,
                 "dataset.maps: at least two parametric maps are required for 2-D habitat clustering")
        _require(all(names), "dataset.maps: every entry needs a 'name'")
        _require(len(set(names)) == len(names), "dataset.maps: duplicate map names")

    cl = cfg["clustering"]
    k_range = cl["k_range"]
    _require(len(k_range) == 2 and 2 <= int(k_range[0]) <= int(k_range[1]),
             "clustering.k_range must be [k_min, k_max] with 2 <= k_min <= k_max")
    if cl["n_clusters"] is not None:
        _require(int(cl["n_clusters"]) >= 2, "clustering.n_clusters must be >= 2")
    _require(cl["selection"] in {"mode", "elbow", "silhouette", "ch", "db"},
             "clustering.selection must be one of mode|elbow|silhouette|ch|db")
    _require(cl["feature_space"] in {"normalized", "raw"},
             "clustering.feature_space must be 'normalized' or 'raw'")

    pp = cfg["preprocessing"]
    lo, hi = pp["percentile_clip"]
    _require(0.0 <= lo < hi <= 100.0, "preprocessing.percentile_clip must satisfy 0 <= lo < hi <= 100")

    if ds.get("series_pattern"):
        _require(int(cfg["dce"]["baseline_frames"]) >= 1, "dce.baseline_frames must be >= 1")
        _require(cfg["dce"]["time_unit"] in {"min", "s"}, "dce.time_unit must be 'min' or 's'")


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Make dataset/output paths absolute relative to the config file directory."""
    base = Path(cfg["_config_path"]).parent.parent  # configs/ lives at repo root
    ds = cfg["dataset"]
    root = Path(ds["root"]).expanduser()
    cfg["dataset"]["root"] = str(root if root.is_absolute() else (base / root).resolve())

    out = Path(cfg["project"]["output_dir"]).expanduser()
    cfg["project"]["output_dir"] = str(out if out.is_absolute() else (base / out).resolve())
    return cfg
