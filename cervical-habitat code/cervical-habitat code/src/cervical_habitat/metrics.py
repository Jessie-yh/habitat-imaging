"""Post-clustering quantification.

Three families of descriptors are produced per patient:

1. **Volume descriptors** -- voxel count, volume in mm^3 and percentage of the
   whole tumour for every habitat.
2. **Parameter descriptors** -- distribution of each parametric map *in its
   original units* inside each habitat (this is what goes into group-level
   statistics; the [0, 1] values are only used for clustering).
3. **Spatial descriptors** -- the Multiregional Spatial Interaction (MSI)
   matrix over the 26-neighbourhood, including the tumour/background boundary,
   summarised by 22 features (18 first-order + 4 second-order).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .habitat import PreparedCase

log = logging.getLogger(__name__)

_STATS = ("mean", "sd", "median", "q1", "q3", "iqr", "min", "max", "p10", "p90", "cv")

_FIRST_ORDER_STATS = ("sum", "mean", "var", "max", "min", "range")


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #

def volume_rows(case: PreparedCase, habitat_volume: np.ndarray,
                n_habitats: int, voxel_mm3: float) -> dict[str, Any]:
    row: dict[str, Any] = {"case_id": case.case_id}
    counts = {h: int((habitat_volume == h).sum()) for h in range(1, n_habitats + 1)}
    total = int(sum(counts.values()))
    row["total_voxels"] = total
    row["total_volume_mm3"] = total * voxel_mm3
    for h in range(1, n_habitats + 1):
        row[f"habitat_{h}_voxels"] = counts[h]
        row[f"habitat_{h}_volume_mm3"] = counts[h] * voxel_mm3
        row[f"habitat_{h}_percent"] = (100.0 * counts[h] / total) if total else np.nan
    row["n_habitats_present"] = int(sum(c > 0 for c in counts.values()))
    return row


# --------------------------------------------------------------------------- #
# Per-habitat parameter distributions (original units)
# --------------------------------------------------------------------------- #

def parameter_stat_rows(case: PreparedCase, habitat_volume: np.ndarray,
                        n_habitats: int) -> list[dict[str, Any]]:
    """Long-format rows: one per (habitat, parameter)."""
    rows: list[dict[str, Any]] = []
    labels = habitat_volume[case.mask]  # 1-based, aligned with case.raw rows

    for h in range(1, n_habitats + 1):
        sel = labels == h
        if not sel.any():
            continue
        for j, name in enumerate(case.feature_names):
            vals = case.raw[sel, j].astype(np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            q1, q3 = np.percentile(vals, [25, 75])
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            rows.append({
                "case_id": case.case_id,
                "habitat": h,
                "parameter": name,
                "n_voxels": int(vals.size),
                "mean": mean,
                "sd": sd,
                "median": float(np.median(vals)),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
                "cv": float(abs(sd / mean)) if abs(mean) > 1e-12 else np.nan,
            })
    return rows


# --------------------------------------------------------------------------- #
# MSI matrix
# --------------------------------------------------------------------------- #

def _offsets_26() -> list[tuple[int, int, int]]:
    return [(dx, dy, dz)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if not (dx == dy == dz == 0)]


def msi_matrix(habitat_volume: np.ndarray, n_habitats: int) -> np.ndarray:
    """Adjacency-count matrix of size (n_habitats + 1) x (n_habitats + 1).

    Class 0 is the surrounding tissue; class ``h`` is habitat ``h``.  Voxel
    pairs where both members are background are ignored.  Because all 26
    offsets in both directions are visited, the matrix is symmetric.
    """
    lab = np.rint(np.asarray(habitat_volume)).astype(np.int16)
    lab[(lab < 0) | (lab > n_habitats)] = 0

    padded = np.pad(lab, 1, mode="constant", constant_values=0)
    core = padded[1:-1, 1:-1, 1:-1]
    counts = np.zeros((n_habitats + 1, n_habitats + 1), dtype=np.int64)

    for dx, dy, dz in _offsets_26():
        nb = padded[1 + dx:padded.shape[0] - 1 + dx,
                    1 + dy:padded.shape[1] - 1 + dy,
                    1 + dz:padded.shape[2] - 1 + dz]
        keep = (core > 0) | (nb > 0)
        if not keep.any():
            continue
        np.add.at(counts, (core[keep].astype(np.int64), nb[keep].astype(np.int64)), 1)

    return counts


def msift_features(counts: np.ndarray) -> tuple[float, dict[str, float]]:
    """Summarise an MSI matrix with 18 first-order + 4 second-order features."""
    p = counts.astype(np.float64)
    total = p.sum()
    feats: dict[str, float] = {}
    if total <= 0:
        return 0.0, {k: float("nan") for k in _feature_names()}

    p /= total
    n = counts.shape[0]

    diag_vals = np.diag(counts).astype(np.float64)
    off = counts[~np.eye(n, dtype=bool)].astype(np.float64)
    all_vals = counts.astype(np.float64).ravel()

    for label, values in (("diag", diag_vals), ("offdiag", off), ("all", all_vals)):
        feats[f"msi_{label}_sum"] = float(values.sum())
        feats[f"msi_{label}_mean"] = float(values.mean())
        feats[f"msi_{label}_var"] = float(values.var())
        feats[f"msi_{label}_max"] = float(values.max())
        feats[f"msi_{label}_min"] = float(values.min())
        feats[f"msi_{label}_range"] = float(values.max() - values.min())

    nz = p[p > 0]
    feats["msi_entropy"] = float(-(nz * np.log2(nz)).sum())
    feats["msi_energy"] = float((p ** 2).sum())

    idx = np.arange(n)
    dist = (idx[:, None] - idx[None, :]).astype(np.float64)
    feats["msi_contrast"] = float((p * dist ** 2).sum())
    feats["msi_homogeneity"] = float((p / (1.0 + np.abs(dist))).sum())

    # composite score: entropy normalised by its maximum (log2 of non-zero pairs)
    max_entropy = np.log2(max(len(nz), 2))
    score = float(feats["msi_entropy"] / max_entropy) if max_entropy > 0 else float("nan")

    return score, feats


def _feature_names() -> list[str]:
    names: list[str] = []
    for label in ("diag", "offdiag", "all"):
        names.extend(f"msi_{label}_{s}" for s in _FIRST_ORDER_STATS)
    names.extend(["msi_entropy", "msi_energy", "msi_contrast", "msi_homogeneity"])
    return names


MSI_FEATURE_NAMES = _feature_names()


def msi_rows(case: PreparedCase, habitat_volume: np.ndarray,
             n_habitats: int) -> dict[str, Any]:
    counts = msi_matrix(habitat_volume, n_habitats)
    score, feats = msift_features(counts)
    row: dict[str, Any] = {"case_id": case.case_id, "msi_heterogeneity_score": score}
    row.update(feats)
    return row


def msi_matrix_rows(case: PreparedCase, habitat_volume: np.ndarray,
                    n_habitats: int) -> list[dict[str, Any]]:
    """Long-format dump of the raw MSI matrix (supplementary material)."""
    counts = msi_matrix(habitat_volume, n_habitats)
    rows = []
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            rows.append({"case_id": case.case_id, "row": i, "col": j,
                         "count": int(counts[i, j])})
    return rows


# --------------------------------------------------------------------------- #
# Cohort-level summary
# --------------------------------------------------------------------------- #

def wide_parameter_table(long_df: pd.DataFrame, stat: str = "mean") -> pd.DataFrame:
    """Pivot (case, habitat, parameter) -> one column per habitat/parameter."""
    if long_df.empty:
        return long_df
    sub = long_df[long_df.get("stat", stat) == stat] if "stat" in long_df else long_df
    sub = sub.copy()
    sub["column"] = sub["parameter"] + "_habitat" + sub["habitat"].astype(str)
    return sub.pivot_table(index="case_id", columns="column", values=stat).reset_index()
