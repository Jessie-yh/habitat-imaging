"""Optional ITHscore computation.

ITHscore (Li et al., European Radiology 2019; see also Radiology 2025) is an
intra-tumour heterogeneity score built from multi-scale radiomics clustering of
the largest tumour slice.  It is *not* reimplemented here -- the upstream
package is used when installed, so results match the published definition.

    pip install ITHscore

If the package is missing the pipeline logs a warning and leaves the columns
empty instead of silently substituting a different metric.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_AVAILABLE: bool | None = None


def available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        try:  # pragma: no cover - depends on optional dependency
            import ITHscore  # noqa: F401
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return bool(_AVAILABLE)


def compute_ithscore(image: np.ndarray, mask: np.ndarray,
                     feature_type: str = "original", n_jobs: int = 1,
                     **kwargs: Any) -> dict[str, float]:
    """Return ``{"ithscore": ...}`` or NaNs when the dependency is unavailable."""
    if not available():
        log.warning("ITHscore is not installed; skipping (pip install ITHscore)")
        return {"ithscore": float("nan"), "ithscore_n_clusters": float("nan")}

    import ITHscore  # pragma: no cover

    try:
        score, n_clusters = ITHscore.ith_score(
            image, mask, feature_type=feature_type, n_jobs=n_jobs, **kwargs
        )
        return {"ithscore": float(score), "ithscore_n_clusters": float(n_clusters)}
    except Exception as exc:  # pragma: no cover - upstream signature drift
        log.warning("ITHscore failed (%s); writing NaN", exc)
        return {"ithscore": float("nan"), "ithscore_n_clusters": float("nan")}
