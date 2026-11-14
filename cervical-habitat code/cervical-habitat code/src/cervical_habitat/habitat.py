"""Turning pooled voxel labels back into 3-D habitat maps."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .preprocess import NormalisationParams

log = logging.getLogger(__name__)


@dataclass
class PreparedCase:
    """ROI voxels of one patient on the common analysis grid.

    Full 3-D volumes are not kept in memory: only the ROI voxel matrix and the
    geometry needed to write results back to disk.
    """

    case_id: str
    mask: np.ndarray  # bool (x, y, z)
    affine: np.ndarray
    header: Any
    feature_names: list[str]
    raw: np.ndarray  # (n_vox, n_params) in original units
    norm: np.ndarray  # (n_vox, n_params) in [0, 1]
    norm_params: dict[str, NormalisationParams] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_voxels(self) -> int:
        return int(self.norm.shape[0])

    def features(self, space: str = "normalized") -> np.ndarray:
        if space == "raw":
            return self.raw
        if space == "normalized":
            return self.norm
        raise ValueError(f"unknown feature space: {space}")


def labels_to_volume(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scatter per-voxel labels (0..k-1) into a uint8 volume, 1-based.

    Background stays 0; habitat ``i`` is written as ``i + 1`` following NIfTI
    label-map conventions.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size != int(mask.sum()):
        raise ValueError(f"{labels.size} labels for {int(mask.sum())} ROI voxels")
    volume = np.zeros(mask.shape, dtype=np.uint8)
    volume[mask] = (labels + 1).astype(np.uint8)
    return volume


def remap_labels(labels: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Apply a 1-based renumbering (``mapping[old_label] -> new_label``)."""
    return mapping[labels]


def rank_mapping_from_centers(centers: np.ndarray) -> np.ndarray:
    """Order clusters by mean normalised intensity: 0 -> habitat 1 ... k -> k."""
    order = np.argsort(centers.sum(axis=1))
    mapping = np.empty(order.size, dtype=np.int64)
    mapping[order] = np.arange(1, order.size + 1)
    return mapping


def habitat_volumes(prepared: Sequence[PreparedCase], all_labels: Sequence[np.ndarray],
                    mapping: np.ndarray, n_habitats: int) -> dict[str, np.ndarray]:
    """case_id -> 1-based habitat volume."""
    out: dict[str, np.ndarray] = {}
    for case, labels in zip(prepared, all_labels):
        out[case.case_id] = labels_to_volume(remap_labels(labels, mapping), case.mask)
    return out
