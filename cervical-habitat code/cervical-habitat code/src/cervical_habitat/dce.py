"""DCE-MRI semi-quantitative parameters: wash-in / wash-out rates.

Conventions (documented in ``docs/methods.md``)
----------------------------------------------
With ``S_0`` the mean of the pre-contrast frames, ``S_peak`` the peak
post-contrast signal, ``S_end`` the last frame and times measured from the
first post-contrast frame:

    wash-in  (%)      = (S_peak - S_0) / S_0 * 100
    wash-out (%)      = (S_peak - S_end) / S_peak * 100
    wash-in  rate     = (S_peak - S_0) / (S_0 * t_peak)          [1/min or 1/s]
    wash-out rate     = (S_peak - S_end) / (S_peak * (t_end - t_peak))

Rates are signal-slope-per-unit-time, which makes them independent of the
frame count and comparable across patients with different sampling schemes.

Because ``S_peak`` is the maximum post-contrast value of the voxel itself,
the wash-out rate is >= 0 for the default per-voxel peak: a voxel cannot end
above its own maximum.  A *negative* wash-out rate therefore only arises with
``robust_peak=True``, where the peak time is taken from the ROI-mean curve and
an individual voxel may still be enhancing at that moment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

log = logging.getLogger(__name__)

OUTPUT_NAMES = ("wash-in rate", "wash-out rate")
EPS = 1e-6


@dataclass
class DCEResult:
    maps: dict[str, np.ndarray]  # full-size 3-D arrays, zeros outside the ROI
    time_axis: np.ndarray
    peak_index: np.ndarray  # full-size int array
    baseline_frames: int
    temporal_resolution_s: float


def compute_semi_quantitative(series: np.ndarray, mask: np.ndarray,
                              tr_s: float,
                              baseline_frames: int = 3,
                              time_unit: str = "min",
                              smooth_sigma_frames: float = 0.0,
                              robust_peak: bool = False,
                              include_percentages: bool = False) -> DCEResult:
    """Compute wash-in/wash-out rate maps from a 4-D DCE series.

    Parameters
    ----------
    series
        4-D array ``(x, y, z, t)`` of signal intensities.
    mask
        3-D tumour mask on the same grid.
    tr_s
        Temporal resolution in **seconds**.
    baseline_frames
        Number of pre-contrast frames averaged into ``S_0``.
    time_unit
        Unit of the returned rate constants (``"min"`` or ``"s"``).
    smooth_sigma_frames
        Gaussian smoothing along time before peak picking (0 = off).
    robust_peak
        Use the 95th percentile frame of the ROI time-activity curve instead of
        the per-voxel argmax.  Reduces noise sensitivity at the cost of a
        slightly earlier peak.
    include_percentages
        Also return the classic ``wash-in`` / ``wash-out`` percentage maps.
        They are **not** used for clustering (the design uses the two rates).
    """
    if series.ndim != 4:
        raise ValueError(f"DCE series must be 4-D (x,y,z,t), got shape {series.shape}")
    n_frames = series.shape[3]
    if n_frames <= baseline_frames + 1:
        raise ValueError(f"need >{baseline_frames + 1} frames, got {n_frames}")
    if time_unit not in {"min", "s"}:
        raise ValueError("time_unit must be 'min' or 's'")

    unit_scale = 60.0 if time_unit == "min" else 1.0  # seconds -> chosen unit

    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        raise ValueError("empty ROI mask")

    # (n_vox, t) matrix keeps memory bounded for whole-cohort runs
    ts = np.asarray(series, dtype=np.float32).reshape(-1, n_frames)[idx, :]
    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)

    if smooth_sigma_frames and smooth_sigma_frames > 0:
        ts = gaussian_filter1d(ts, sigma=float(smooth_sigma_frames), axis=1,
                              mode="nearest")

    s0 = ts[:, :baseline_frames].mean(axis=1)
    post = ts[:, baseline_frames:]
    t_post = (np.arange(post.shape[1], dtype=np.float32) + 1.0) * tr_s / unit_scale

    if robust_peak:
        roi_curve = post.mean(axis=0)
        peak_col = int(np.argmax(roi_curve))
        peak_idx = np.full(ts.shape[0], peak_col, dtype=np.int32)
    else:
        peak_idx = np.argmax(post, axis=1).astype(np.int32)

    rows = np.arange(ts.shape[0])
    s_peak = post[rows, peak_idx]
    s_end = post[:, -1]
    t_peak = t_post[peak_idx]
    t_end = np.full_like(t_peak, t_post[-1])

    s0_safe = np.where(np.abs(s0) < EPS, np.nan, s0)
    s_peak_safe = np.where(np.abs(s_peak) < EPS, np.nan, s_peak)

    wash_in_rate = (s_peak - s0) / (s0_safe * np.maximum(t_peak, EPS))
    dt_out = np.maximum(t_end - t_peak, EPS)
    wash_out_rate = (s_peak - s_end) / (s_peak_safe * dt_out)

    # Voxels whose baseline or peak signal is ~0 carry no enhancement information
    wash_in_rate = np.nan_to_num(wash_in_rate, nan=0.0, posinf=0.0, neginf=0.0)
    wash_out_rate = np.nan_to_num(wash_out_rate, nan=0.0, posinf=0.0, neginf=0.0)

    def scatter(values: np.ndarray) -> np.ndarray:
        out = np.zeros(mask.size, dtype=np.float32)
        out[idx] = values.astype(np.float32)
        return out.reshape(mask.shape)

    peak_full = np.zeros(mask.size, dtype=np.int32)
    peak_full[idx] = peak_idx + baseline_frames
    peak_full = peak_full.reshape(mask.shape)

    log.info("DCE semi-quantitative maps: %d ROI voxels, %d frames, TR=%.2fs, unit=%s",
             idx.size, n_frames, tr_s, time_unit)

    maps = {
        "wash-in rate": scatter(wash_in_rate),
        "wash-out rate": scatter(wash_out_rate),
    }
    if include_percentages:
        maps["wash-in"] = scatter((s_peak - s0) / s0_safe * 100.0)
        maps["wash-out"] = scatter((s_peak - s_end) / s_peak_safe * 100.0)

    return DCEResult(
        maps=maps,
        time_axis=t_post,
        peak_index=peak_full,
        baseline_frames=int(baseline_frames),
        temporal_resolution_s=float(tr_s),
    )


def dce_metadata(result: DCEResult, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_frames": result.baseline_frames,
        "temporal_resolution_s": result.temporal_resolution_s,
        "time_unit": cfg.get("time_unit", "min"),
        "smooth_frames_sigma": float(cfg.get("smooth_frames_sigma", 0.0)),
        "robust_peak": bool(cfg.get("robust_peak", False)),
        "n_post_contrast_frames": int(result.time_axis.size),
        "last_time_point": float(result.time_axis[-1]),
    }
