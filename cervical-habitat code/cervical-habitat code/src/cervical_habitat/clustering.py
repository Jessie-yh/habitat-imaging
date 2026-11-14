"""Voxel-wise K-means habitat clustering and cluster-number selection.

Protocol
--------
* features: the normalised parametric maps (one column per map),
* algorithm: K-means with k-means++ initialisation, ``n_init`` restarts,
* cluster number: fixed by the study design (DCE k=3, IVIM k=4) or chosen from
  four criteria -- elbow (max curvature), silhouette (max), Calinski-Harabasz
  (max) and Davies-Bouldin (min) -- combined by majority vote ("mode").

Tie-breaking is deterministic (elbow > silhouette > CH > DB > smaller k) so the
same data always yields the same ``k``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

log = logging.getLogger(__name__)

METHODS = ("elbow", "silhouette", "ch", "db")
_TIE_ORDER = {m: i for i, m in enumerate(METHODS)}


@dataclass
class ClusterEvaluation:
    table: pd.DataFrame  # one row per k
    suggested: dict[str, int]  # method -> k
    chosen_k: int
    strategy: str
    votes: dict[int, int] = field(default_factory=dict)

    def as_records(self) -> list[dict[str, Any]]:
        return self.table.to_dict(orient="records")


def find_elbow(distortions: Sequence[float], k_values: Sequence[int]) -> int:
    """Elbow via maximum distance to the chord of the normalised inertia curve.

    This is the standard Kneedle formulation: normalise both axes to [0, 1],
    draw the straight line between the first and last point, and take the point
    furthest from that line.  It is more stable than a discrete-curvature
    estimate, which is biased towards the point *after* the bend.
    """
    k = np.asarray(k_values, dtype=float)
    d = np.asarray(distortions, dtype=float)
    if k.size < 3:
        return int(k_values[0])

    k_n = (k - k.min()) / max(k.max() - k.min(), 1e-12)
    d_n = (d - d.min()) / max(d.max() - d.min(), 1e-12)

    # |(y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1| / ||(x2 - x1, y2 - y1)||
    x1, y1 = k_n[0], d_n[0]
    x2, y2 = k_n[-1], d_n[-1]
    num = np.abs((y2 - y1) * k_n - (x2 - x1) * d_n + x2 * y1 - y2 * x1)
    den = float(np.hypot(y2 - y1, x2 - x1))
    if den <= 1e-12:  # flat curve -> no elbow
        return int(k_values[0])

    return int(k_values[int(np.argmax(num / den))])


def _subsample(x: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(x.shape[0], size=max_samples, replace=False))


def evaluate_cluster_numbers(x: np.ndarray, k_range: Sequence[int], seed: int = 42,
                             n_init: int = 20, max_samples_for_eval: int = 50000,
                             strategy: str = "mode") -> ClusterEvaluation:
    """Score every k in ``k_range`` and pick one."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"feature matrix must be 2-D, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("feature matrix contains NaN/Inf")

    k_min, k_max = int(min(k_range)), int(max(k_range))
    # a cluster needs at least ~10 voxels to be meaningful
    k_max = min(k_max, max(2, x.shape[0] // 10))
    if k_max < k_min:
        raise ValueError(f"not enough voxels ({x.shape[0]}) for k >= {k_min}")

    rows: list[dict[str, Any]] = []
    sub = _subsample(x, max_samples_for_eval, seed)
    x_eval = x[sub]

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                    random_state=seed).fit(x)
        labels_eval = km.predict(x_eval)
        counts = np.bincount(labels_eval, minlength=k)

        if counts.min() >= 2 and len(np.unique(labels_eval)) == k:
            sil = float(silhouette_score(x_eval, labels_eval))
        else:  # degenerate split, silhouette undefined
            sil = float("nan")

        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": sil,
            "calinski_harabasz": float(calinski_harabasz_score(x_eval, labels_eval)),
            "davies_bouldin": float(davies_bouldin_score(x_eval, labels_eval)),
        })

    table = pd.DataFrame(rows)
    k_values = table["k"].to_numpy()

    suggested = {
        "elbow": find_elbow(table["inertia"].to_numpy(), k_values),
        "silhouette": int(k_values[int(np.nanargmax(table["silhouette"].to_numpy()))]),
        "ch": int(k_values[int(np.argmax(table["calinski_harabasz"].to_numpy()))]),
        "db": int(k_values[int(np.argmin(table["davies_bouldin"].to_numpy()))]),
    }

    if strategy in suggested:
        chosen = suggested[strategy]
        votes: dict[int, int] = {}
    else:
        votes = {}
        for method in METHODS:
            votes[suggested[method]] = votes.get(suggested[method], 0) + 1
        chosen = _vote_winner(votes, suggested)

    table["selected"] = table["k"] == chosen
    log.info("cluster evaluation k=%d..%d -> %s | chosen k=%d (%s)",
             k_min, k_max, suggested, chosen, strategy)

    return ClusterEvaluation(table=table, suggested=suggested, chosen_k=int(chosen),
                             strategy=strategy, votes=votes)


def _vote_winner(votes: dict[int, int], suggested: dict[str, int]) -> int:
    """Majority vote; ties resolved by method priority, then by smaller k."""
    best = max(votes.values())
    tied = [k for k, v in votes.items() if v == best]
    if len(tied) == 1:
        return tied[0]

    def rank(k: int) -> tuple[int, int]:
        priority = min(_TIE_ORDER[m] for m, v in suggested.items() if v == k)
        return priority, k

    return min(tied, key=rank)


def fit_habitats(x: np.ndarray, n_clusters: int, seed: int = 42,
                 n_init: int = 20) -> KMeans:
    """Final model on all pooled voxels."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < n_clusters * 5:
        raise ValueError(
            f"only {x.shape[0]} voxels for k={n_clusters}; too few for a stable fit"
        )
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=n_init,
                random_state=seed).fit(x)
    log.info("final K-means: %d voxels, k=%d, inertia=%.3f",
             x.shape[0], n_clusters, km.inertia_)
    return km


def label_habitats_by_intensity(centers: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Renumber clusters so habitat 1 is the least-enhanced and k the most.

    Habitat numbering is otherwise arbitrary; a fixed rule makes group-level
    statistics and figures comparable across patients.
    """
    score = centers.sum(axis=1)
    order = np.argsort(score)  # ascending: low signal -> habitat 1
    remap = np.empty_like(order)
    remap[order] = np.arange(1, len(order) + 1)
    return remap


def cluster_summary(x: np.ndarray, labels: np.ndarray, feature_names: Sequence[str],
                    n_clusters: int) -> pd.DataFrame:
    rows = []
    total = len(labels)
    for c in range(n_clusters):
        sel = labels == c
        row: dict[str, Any] = {"cluster": c + 1, "n_voxels": int(sel.sum()),
                               "percent": 100.0 * sel.sum() / total}
        for i, name in enumerate(feature_names):
            vals = x[sel, i]
            row[f"{name}_mean"] = float(vals.mean())
            row[f"{name}_sd"] = float(vals.std())
            row[f"{name}_min"] = float(vals.min())
            row[f"{name}_max"] = float(vals.max())
        rows.append(row)
    return pd.DataFrame(rows)
