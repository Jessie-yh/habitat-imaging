"""Image preprocessing: grid alignment, resampling, non-finite handling and
ROI-guided intensity normalisation.

Normalisation follows the study protocol:

1. take the intensity distribution **inside the tumour ROI** (not the whole
   image, which would be dominated by air/fat),
2. clip at the 1st/99th ROI percentiles to remove outliers,
3. min-max scale so every parametric map lives on the same [0, 1] scale,
4. clip to [0, 1].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output

log = logging.getLogger(__name__)


@dataclass
class NormalisationParams:
    """Everything needed to reproduce (or invert) a normalisation."""

    lower: float
    upper: float
    percentile_low: float
    percentile_high: float
    roi_voxels: int
    roi_min_raw: float
    roi_max_raw: float
    fallback: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower_bound": self.lower,
            "upper_bound": self.upper,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "roi_voxels": self.roi_voxels,
            "roi_min_raw": self.roi_min_raw,
            "roi_max_raw": self.roi_max_raw,
            "fallback_used": self.fallback,
        }


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _same_grid(a_affine: np.ndarray, a_shape: tuple[int, ...],
               b_affine: np.ndarray, b_shape: tuple[int, ...],
               tol: float = 1e-4) -> bool:
    return a_shape[:3] == b_shape[:3] and np.allclose(a_affine, b_affine, atol=tol)


def align_to_reference(data: np.ndarray, affine: np.ndarray, header: Any,
                       ref_data: np.ndarray, ref_affine: np.ndarray,
                       ref_header: Any, order: int = 3) -> np.ndarray:
    """Resample ``data`` onto the reference grid (usually the ROI grid).

    Uses nearest neighbour for label maps (``order=0``) so habitat/ROI labels
    are never interpolated into non-existent values.
    """
    if _same_grid(affine, data.shape, ref_affine, ref_data.shape):
        return np.asarray(data, dtype=np.float32)

    log.info("  resampling to ROI grid: %s -> %s", data.shape[:3], ref_data.shape[:3])
    src = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine, header)
    out = resample_from_to(src, (ref_data.shape[:3], ref_affine), order=order,
                           mode="constant", cval=0.0)
    return np.asanyarray(out.dataobj, dtype=np.float32)


def resample_spacing(data: np.ndarray, affine: np.ndarray, header: Any,
                     target_spacing: tuple[float, float, float],
                     order: int = 3) -> tuple[np.ndarray, np.ndarray, Any]:
    """Resample to an isotropic-in-plane / thick-slice grid, e.g. 1x1x3 mm."""
    zooms = np.abs(np.diag(affine[:3, :3]))
    if np.allclose(zooms, target_spacing, atol=1e-3):
        return np.asarray(data, dtype=np.float32), affine, header

    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine, header)
    out = resample_to_output(img, voxel_sizes=tuple(target_spacing), order=order,
                             mode="constant", cval=0.0)
    log.info("  resampled %s -> %s (target %s mm)", tuple(round(z, 2) for z in zooms),
             out.shape[:3], target_spacing)
    return (np.asanyarray(out.dataobj, dtype=np.float32), out.affine, out.header)


def grid_spacing(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.abs(np.diag(affine[:3, :3])))  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Intensity
# --------------------------------------------------------------------------- #

def fill_nonfinite(data: np.ndarray, mask: np.ndarray, strategy: str = "roi_median") -> tuple[np.ndarray, int]:
    """Replace NaN/Inf.

    ``roi_median`` (default) keeps the voxel in the analysis with a neutral
    value instead of silently shrinking the tumour; ``zero`` writes 0; ``drop``
    leaves the values in place and lets the caller remove them.
    """
    bad = ~np.isfinite(data)
    n_bad = int(bad.sum())
    if n_bad == 0 or strategy == "drop":
        return data, n_bad
    if strategy == "zero":
        fill = 0.0
    else:
        roi_vals = data[mask & np.isfinite(data)]
        fill = float(np.median(roi_vals)) if roi_vals.size else 0.0
    out = np.array(data, dtype=np.float32, copy=True)
    out[bad] = fill
    return out, n_bad


def roi_guided_normalize(data: np.ndarray, mask: np.ndarray,
                         percentile_low: float = 1.0,
                         percentile_high: float = 99.0,
                         minmax: bool = True) -> tuple[np.ndarray, NormalisationParams]:
    """Clip ROI percentiles, min-max scale and clip to [0, 1]."""
    vals = data[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("ROI is empty after removing non-finite voxels")

    lower = float(np.percentile(vals, percentile_low))
    upper = float(np.percentile(vals, percentile_high))
    fallback = False

    if upper - lower < 1e-6:
        fallback = True
        if abs(upper) < 1e-6:
            lower, upper = -1.0, 1.0
        else:
            lower, upper = upper * 0.5, upper * 1.5
        log.warning("  ROI intensity range is degenerate; fallback bounds [%.3g, %.3g]",
                    lower, upper)

    if minmax:
        norm = (np.asarray(data, dtype=np.float32) - lower) / (upper - lower)
        np.clip(norm, 0.0, 1.0, out=norm)
    else:
        norm = np.asarray(data, dtype=np.float32).copy()

    params = NormalisationParams(
        lower=lower,
        upper=upper,
        percentile_low=float(percentile_low),
        percentile_high=float(percentile_high),
        roi_voxels=int(vals.size),
        roi_min_raw=float(vals.min()),
        roi_max_raw=float(vals.max()),
        fallback=fallback,
    )
    return norm.astype(np.float32), params
