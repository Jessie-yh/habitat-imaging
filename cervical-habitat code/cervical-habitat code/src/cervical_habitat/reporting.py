"""Figures produced by the pipeline.

All figures are optional (``output.save_plots``) and use the non-interactive
Agg backend, so the pipeline runs head-less on a cluster.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def plot_k_selection(table: pd.DataFrame, chosen_k: int, path: str | Path) -> Path | None:
    if table is None or table.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("inertia", "Inertia (elbow)", axes[0, 0], "min"),
        ("silhouette", "Silhouette score", axes[0, 1], "max"),
        ("calinski_harabasz", "Calinski-Harabasz", axes[1, 0], "max"),
        ("davies_bouldin", "Davies-Bouldin", axes[1, 1], "min"),
    ]
    for col, title, ax, _ in panels:
        ax.plot(table["k"], table[col], "o-", color="#1f77b4", lw=1.8, ms=6)
        if chosen_k in set(table["k"]):
            row = table.loc[table["k"] == chosen_k].iloc[0]
            ax.axvline(chosen_k, color="#d62728", ls="--", lw=1.2, label=f"k = {chosen_k}")
            ax.plot([chosen_k], [row[col]], "X", color="#d62728", ms=11)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("number of clusters k")
        ax.grid(alpha=0.3)
        ax.set_xticks(list(table["k"]))

    fig.suptitle("Cluster-number evaluation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_space(x: np.ndarray, labels: np.ndarray, names: Sequence[str],
                       centers: np.ndarray, path: str | Path,
                       max_points: int = 20000, seed: int = 42) -> Path | None:
    """Pairwise feature-space panels, coloured by habitat."""
    n_feat = len(names)
    if n_feat < 2:
        return None

    pairs = [(i, j) for i in range(n_feat) for j in range(i + 1, n_feat)]
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)

    rng = np.random.default_rng(seed)
    if x.shape[0] > max_points:
        idx = rng.choice(x.shape[0], size=max_points, replace=False)
    else:
        idx = np.arange(x.shape[0])

    for ax, (i, j) in zip(axes.ravel(), pairs):
        for c in range(int(labels.max()) + 1):
            sel = labels[idx] == c
            ax.scatter(x[idx][sel, i], x[idx][sel, j], s=3, alpha=0.35,
                       color=COLORS[c % len(COLORS)], label=f"habitat {c + 1}",
                       linewidths=0)
        ax.scatter(centers[:, i], centers[:, j], marker="X", s=110,
                   c=[COLORS[c % len(COLORS)] for c in range(len(centers))],
                   edgecolors="black", linewidths=1.0, zorder=5)
        ax.set_xlabel(names[i], fontsize=10)
        ax.set_ylabel(names[j], fontsize=10)
        ax.grid(alpha=0.25)

    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")

    handles, labs = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=min(len(labs), 6), fontsize=9,
               frameon=False)
    fig.suptitle("Habitat clusters in parameter space (X = cluster centre)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_case_montage(image: np.ndarray, habitat: np.ndarray, mask: np.ndarray,
                      path: str | Path, n_slices: int = 5,
                      n_habitats: int = 3, margin: int = 4) -> Path | None:
    """Axial slices cropped to the tumour bounding box, with habitat contours."""
    zs = np.where(mask.any(axis=(0, 1)))[0]
    if zs.size == 0:
        return None
    pick = np.linspace(zs.min(), zs.max(), num=min(n_slices, zs.size)).round().astype(int)
    pick = np.unique(pick)

    # crop to the tumour bounding box (+margin) so the lesion fills the panel
    xs = np.where(mask.any(axis=(1, 2)))[0]
    ys = np.where(mask.any(axis=(0, 2)))[0]
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, mask.shape[0])
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, mask.shape[1])

    # window on tumour voxels for display only
    vals = image[mask]
    vmin, vmax = (np.percentile(vals, 1), np.percentile(vals, 99)) if vals.size else (0, 1)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(1, len(pick), figsize=(2.9 * len(pick), 3.3), squeeze=False)
    for ax, z in zip(axes.ravel(), pick):
        ax.imshow(np.rot90(image[x0:x1, y0:y1, z]), cmap="gray", vmin=vmin, vmax=vmax)
        for h in range(1, n_habitats + 1):
            layer = np.ma.masked_where(habitat[x0:x1, y0:y1, z] != h,
                                       habitat[x0:x1, y0:y1, z])
            if layer.count() == 0:
                continue
            ax.contour(np.rot90(layer.filled(0)), levels=[0.5],
                       colors=[COLORS[(h - 1) % len(COLORS)]], linewidths=1.3)
        ax.set_title(f"z = {z}", fontsize=9)
    ax.axis("off")

    handles = [plt.Line2D([], [], color=COLORS[(h - 1) % len(COLORS)], lw=1.8,
                          label=f"habitat {h}") for h in range(1, n_habitats + 1)]
    fig.legend(handles=handles, loc="lower center", ncol=n_habitats, fontsize=9,
               frameon=False)
    fig.suptitle("Habitat contours", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_msi(counts: np.ndarray, path: str | Path) -> Path | None:
    p = counts.astype(float)
    if p.sum() <= 0:
        return None
    p = p / p.sum()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(p, cmap="viridis")
    ax.set_xlabel("neighbour class")
    ax.set_ylabel("voxel class")
    ax.set_title("MSI matrix (0 = surrounding tissue)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
