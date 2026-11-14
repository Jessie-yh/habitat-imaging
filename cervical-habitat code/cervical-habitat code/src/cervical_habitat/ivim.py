"""Optional IVIM parameter estimation from a multi-b DWI series.

Most clinical sites export vendor-generated ``D`` / ``D*`` / ``f`` maps, which
can be fed to the pipeline directly (``dataset.maps``).  This module exists for
sites that only have the raw multi-b DWI and want a reproducible fit.

Segmented (two-step) bi-exponential fit::

    S(b) / S0 = (1 - f) * exp(-b * D) + f * exp(-b * D*)

1. ``D`` and ``(1 - f)`` from a log-linear fit of the high-b points,
2. ``D*`` from the low-b residuals after subtracting the perfusion-free term.

The segmented approach is deliberately used instead of a full non-linear fit:
it is the standard, stable choice for clinical IVIM and avoids the well-known
D*/f trade-off in noisy voxels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

IVIM_OUTPUT_NAMES = ("D", "D*", "f")
EPS = 1e-8


@dataclass
class IVIMFit:
    maps: dict[str, np.ndarray]  # full 3-D arrays (0 outside mask)
    b_values: np.ndarray
    n_voxels: int
    n_failed: int


def _linear_fit(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised least-squares slope/intercept of y = a + m*x over valid rows."""
    n = valid.sum(axis=1).astype(np.float64)
    xm = np.where(valid, x, 0.0).sum(axis=1) / np.maximum(n, 1)
    ym = np.where(valid, y, 0.0).sum(axis=1) / np.maximum(n, 1)
    dx = np.where(valid, x - xm[:, None], 0.0)
    dy = np.where(valid, y - ym[:, None], 0.0)
    denom = (dx * dx).sum(axis=1)
    slope = np.where(denom > EPS, (dx * dy).sum(axis=1) / np.maximum(denom, EPS), 0.0)
    intercept = ym - slope * xm
    return slope, intercept


def fit_ivim(dwi: np.ndarray, b_values: np.ndarray, mask: np.ndarray,
             b_value_threshold: float = 200.0,
             d_bounds: tuple[float, float] = (0.0, 0.005),
             dstar_bounds: tuple[float, float] = (0.005, 0.5),
             f_bounds: tuple[float, float] = (0.0, 1.0)) -> IVIMFit:
    """Fit D, D* and f voxel-wise inside ``mask``.

    ``dwi`` is a 4-D array ``(x, y, z, b)``; ``b_values`` are in s/mm^2 and
    must be in the same order as the 4th dimension of ``dwi``.
    """
    if dwi.ndim != 4:
        raise ValueError(f"DWI must be 4-D (x,y,z,b), got {dwi.shape}")
    b_values = np.asarray(b_values, dtype=np.float64)
    if b_values.size != dwi.shape[3]:
        raise ValueError(f"{b_values.size} b-values for {dwi.shape[3]} volumes")
    if np.any(b_values < 0):
        raise ValueError("b-values must be non-negative")

    order = np.argsort(b_values)
    b_sorted = b_values[order]

    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        raise ValueError("empty ROI mask")

    signals = np.asarray(dwi, dtype=np.float64).reshape(-1, dwi.shape[3])[idx][:, order]
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalise by the b=0 (or lowest-b) volume
    b0_col = int(np.argmin(b_sorted))
    s0 = signals[:, b0_col]
    s0_safe = np.where(s0 <= EPS, np.nan, s0)
    ratio = signals / s0_safe[:, None]
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ratio = np.clip(ratio, EPS, 1.0)

    log_s = np.log(ratio)

    high = b_sorted >= b_value_threshold
    if high.sum() < 2:
        raise ValueError(
            f"need >= 2 b-values >= {b_value_threshold} for the D fit; got {int(high.sum())}"
        )

    # Step 1: D from the perfusion-free high-b regime
    slope, intercept = _linear_fit(b_sorted[high], log_s[:, high],
                                   np.ones((len(idx), int(high.sum())), dtype=bool))
    d = np.clip(-slope, d_bounds[0], d_bounds[1])
    a_high = np.clip(np.exp(intercept), EPS, 1.0)  # ~ (1 - f)
    f = np.clip(1.0 - a_high, f_bounds[0], f_bounds[1])

    # Step 2: D* from the low-b residual after removing the extravascular term
    low = b_sorted < b_value_threshold
    if low.any():
        b_low = b_sorted[low]
        residual = ratio[:, low] - (1.0 - f)[:, None] * np.exp(-b_low[None, :] * d[:, None])
        residual = np.clip(residual, EPS, None)
        log_res = np.log(residual)
        valid = residual > EPS
        slope_low, intercept_low = _linear_fit(b_low[None, :].repeat(len(idx), axis=0) * np.ones_like(log_res),
                                               log_res, valid)
        dstar = np.clip(-slope_low, dstar_bounds[0], dstar_bounds[1])
        # voxels where the perfusion term was undetectable -> no reliable D*
        dstar = np.where(valid.sum(axis=1) >= 1, dstar, 0.0)
    else:
        dstar = np.zeros_like(d)

    failed = int((s0 <= EPS).sum())

    def scatter(values: np.ndarray) -> np.ndarray:
        out = np.zeros(mask.size, dtype=np.float32)
        out[idx] = values.astype(np.float32)
        return out.reshape(mask.shape)

    log.info("IVIM fit: %d voxels, %d with non-positive S0, b=%s",
             idx.size, failed, b_sorted.tolist())

    return IVIMFit(
        maps={"D": scatter(d), "D*": scatter(dstar), "f": scatter(f)},
        b_values=b_sorted,
        n_voxels=int(idx.size),
        n_failed=failed,
    )
