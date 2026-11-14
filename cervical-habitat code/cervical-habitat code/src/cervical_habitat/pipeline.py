"""End-to-end pipeline: read -> preprocess -> cluster -> quantify -> report."""

from __future__ import annotations

import logging
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__, ith
from .clustering import cluster_summary, evaluate_cluster_numbers, fit_habitats
from .config import resolve_paths
from .dce import compute_semi_quantitative, dce_metadata
from .habitat import PreparedCase, rank_mapping_from_centers
from .io import (
    Case,
    load_nifti,
    save_json,
    save_nifti,
    scan_cases,
    temporal_resolution,
    voxel_volume_mm3,
    write_csv,
)
from .metrics import (
    MSI_FEATURE_NAMES,
    msi_matrix,
    msi_rows,
    parameter_stat_rows,
    volume_rows,
    wide_parameter_table,
)
from .preprocess import (
    align_to_reference,
    fill_nonfinite,
    grid_spacing,
    resample_spacing,
    roi_guided_normalize,
)
from .reporting import plot_case_montage, plot_feature_space, plot_k_selection

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    output_dir: Path
    n_cases: int
    n_clusters: int
    weights: dict[str, float] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Single-case preparation
# --------------------------------------------------------------------------- #

def _resample_4d(series: np.ndarray, affine: np.ndarray, header: Any,
                 target: Sequence[float], order: int = 1) -> tuple[np.ndarray, np.ndarray, Any]:
    frames = []
    out_affine = None
    out_header = None
    for t in range(series.shape[3]):
        frame, out_affine, out_header = resample_spacing(
            series[..., t], affine, header, tuple(target), order=order)
        frames.append(frame)
    return np.stack(frames, axis=3), out_affine, out_header  # type: ignore[arg-type]


def _to_analysis_grid(data: np.ndarray, affine: np.ndarray, header: Any,
                      roi_data: np.ndarray, roi_affine: np.ndarray, roi_header: Any,
                      cfg: dict[str, Any], order: int) -> tuple[np.ndarray, np.ndarray, Any]:
    """Align a map to the ROI grid, then both to the target spacing."""
    data = align_to_reference(data, affine, header, roi_data, roi_affine, roi_header,
                              order=order)
    if cfg["resample"]["enabled"]:
        data, affine, header = resample_spacing(
            data, roi_affine, roi_header, tuple(cfg["resample"]["target_spacing"]),
            order=order)
    else:
        affine, header = roi_affine, roi_header
    return data, affine, header


def prepare_case(case: Case, cfg: dict[str, Any]) -> PreparedCase:
    """Load one case, put every map on the analysis grid and normalise it."""
    log.info("[%s] loading ROI", case.case_id)
    roi_data, roi_affine, roi_header = load_nifti(case.roi_path)
    if roi_data.ndim != 3:
        raise ValueError(f"ROI must be 3-D, got shape {roi_data.shape}")

    mask = roi_data > 0
    if not mask.any():
        raise ValueError("ROI contains no positive voxels")

    # --- ROI -> target grid ------------------------------------------------
    if cfg["resample"]["enabled"]:
        target = tuple(cfg["resample"]["target_spacing"])
        roi_rs, roi_affine, roi_header = resample_spacing(
            roi_data, roi_affine, roi_header, target,
            order=int(cfg["resample"]["order_mask"]))
        mask = roi_rs > 0.5
        if not mask.any():
            raise ValueError("ROI vanished after resampling (check target spacing)")

    meta: dict[str, Any] = {
        "roi_path": str(case.roi_path),
        "analysis_spacing_mm": list(grid_spacing(roi_affine)),
        "roi_voxels_after_resample": int(mask.sum()),
    }

    # --- parameter maps ----------------------------------------------------
    raw_maps: dict[str, np.ndarray] = {}
    if case.series_path is not None:
        log.info("[%s] computing DCE semi-quantitative maps", case.case_id)
        series, s_affine, s_header = load_nifti(case.series_path)
        tr = temporal_resolution(s_header, cfg["dce"].get("temporal_resolution_s"))
        if cfg["resample"]["enabled"]:
            series, s_affine, s_header = _resample_4d(
                series, s_affine, s_header, cfg["resample"]["target_spacing"],
                order=int(cfg["resample"]["order_param"]))
        else:
            series = align_to_reference(series, s_affine, s_header, roi_data,
                                        roi_affine, roi_header, order=1)
        if series.shape[:3] != mask.shape:
            raise ValueError(
                f"DCE series grid {series.shape[:3]} does not match ROI grid {mask.shape}")
        dce = compute_semi_quantitative(
            series, mask, tr,
            baseline_frames=int(cfg["dce"]["baseline_frames"]),
            time_unit=cfg["dce"]["time_unit"],
            smooth_sigma_frames=float(cfg["dce"].get("smooth_frames_sigma", 0.0)),
            robust_peak=bool(cfg["dce"].get("robust_peak", False)),
        )
        raw_maps.update(dce.maps)
        meta["dce"] = dce_metadata(dce, cfg["dce"])
    else:
        for name, path in case.maps.items():
            log.info("[%s] loading map '%s'", case.case_id, name)
            data, affine, header = load_nifti(path)
            data, _, _ = _to_analysis_grid(
                data, affine, header, roi_data, roi_affine, roi_header, cfg,
                order=int(cfg["resample"]["order_param"]))
            if data.shape != mask.shape:
                raise ValueError(
                    f"map '{name}' grid {data.shape} does not match ROI grid {mask.shape}")
            raw_maps[name] = data

    feature_names = list(raw_maps.keys())
    if len(feature_names) < 2:
        raise ValueError("at least two parametric maps are required")

    # --- intensity cleanup + normalisation --------------------------------
    lower_p, upper_p = cfg["preprocessing"]["percentile_clip"]
    strategy = cfg["preprocessing"]["fill_nonfinite"]
    norm_cols, raw_cols = [], []
    norm_params = {}
    n_bad_total = 0

    for name in feature_names:
        data, n_bad = fill_nonfinite(raw_maps[name], mask, strategy=strategy)
        n_bad_total += n_bad
        norm, params = roi_guided_normalize(
            data, mask, float(lower_p), float(upper_p),
            minmax=bool(cfg["preprocessing"]["minmax_scale"]))
        norm_params[name] = params
        raw_cols.append(data[mask].astype(np.float32))
        norm_cols.append(norm[mask].astype(np.float32))

    meta["nonfinite_voxels_filled"] = int(n_bad_total)

    prepared = PreparedCase(
        case_id=case.case_id,
        mask=mask,
        affine=roi_affine,
        header=roi_header,
        feature_names=feature_names,
        raw=np.column_stack(raw_cols),
        norm=np.column_stack(norm_cols),
        norm_params=norm_params,
        meta=meta,
    )

    min_vox = int(cfg["preprocessing"]["min_roi_voxels"])
    if prepared.n_voxels < min_vox:
        raise ValueError(f"ROI too small ({prepared.n_voxels} < {min_vox} voxels)")

    log.info("[%s] ready: %d ROI voxels, maps=%s", case.case_id,
             prepared.n_voxels, feature_names)
    return prepared


# --------------------------------------------------------------------------- #
# Cohort run
# --------------------------------------------------------------------------- #

def _stack_features(prepared: Sequence[PreparedCase], space: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    mats = [c.features(space) for c in prepared]
    slices, start = [], 0
    for m in mats:
        slices.append((start, start + m.shape[0]))
        start += m.shape[0]
    return np.vstack(mats).astype(np.float64), slices


def run_pipeline(cfg: dict[str, Any], limit: int | None = None) -> RunResult:
    t0 = time.time()
    cfg = resolve_paths(cfg)
    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["project"]["seed"])
    np.random.seed(seed)

    cases_scan = scan_cases(cfg)
    cases = cases_scan.cases
    if limit:
        cases = cases[:limit]

    prepared: list[PreparedCase] = []
    failures: list[dict[str, str]] = list(cases_scan.skipped)
    for case in cases:
        try:
            prepared.append(prepare_case(case, cfg))
        except Exception as exc:
            log.exception("[%s] failed: %s", case.case_id, exc)
            failures.append({"case_id": case.case_id, "stage": "preprocessing",
                             "error": f"{type(exc).__name__}: {exc}"})

    if not prepared:
        raise RuntimeError("every case failed during preprocessing; see log")

    feature_names = prepared[0].feature_names
    n_maps = len(feature_names)
    for c in prepared:
        if c.feature_names != feature_names:
            raise ValueError(
                f"inconsistent map set for {c.case_id}: {c.feature_names} vs {feature_names}")

    space = cfg["clustering"]["feature_space"]
    x, slices = _stack_features(prepared, space)
    log.info("pooled feature matrix: %s (%s space)", x.shape, space)

    # --- cluster number ---------------------------------------------------
    clustering_cfg = cfg["clustering"]
    evaluation = None
    if clustering_cfg.get("evaluate", True) or clustering_cfg["n_clusters"] is None:
        evaluation = evaluate_cluster_numbers(
            x, clustering_cfg["k_range"], seed=seed,
            n_init=int(clustering_cfg["n_init"]),
            max_samples_for_eval=int(clustering_cfg["max_samples_for_eval"]),
            strategy=clustering_cfg["selection"])
        if evaluation is not None:
            write_csv(evaluation.table, out_dir / "cluster_evaluation.csv",
                      encoding=cfg["output"]["csv_encoding"],
                      also_gbk=cfg["output"]["also_gbk"])

    k = int(clustering_cfg["n_clusters"]) if clustering_cfg["n_clusters"] else int(evaluation.chosen_k)
    if evaluation is not None and clustering_cfg["n_clusters"]:
        log.info("fixed k=%d from config (evaluation suggested %s)", k, evaluation.suggested)
    if k > n_maps * 4:
        log.warning("k=%d is large relative to %d features; clusters may be unstable", k, n_maps)

    # --- fit on a capped subsample, predict everything --------------------
    max_fit = int(clustering_cfg.get("max_voxels_for_fit", 300000))
    if x.shape[0] > max_fit:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(x.shape[0], size=max_fit, replace=False))
        log.info("fitting K-means on %d/%d voxels (max_voxels_for_fit)", max_fit, x.shape[0])
    else:
        idx = np.arange(x.shape[0])

    model = fit_habitats(x[idx], k, seed=seed, n_init=int(clustering_cfg["n_init"]))
    labels_all = model.predict(x).astype(np.int64)
    centers = model.cluster_centers_

    mapping = rank_mapping_from_centers(centers)
    labels_all = mapping[labels_all] - 1  # 0-based, ordered by intensity

    per_case_labels = [labels_all[a:b] for a, b in slices]
    hab_volumes = {c.case_id: _vol_from_labels(lab, c.mask, k)
                   for c, lab in zip(prepared, per_case_labels)}

    result = RunResult(output_dir=out_dir, n_cases=len(prepared), n_clusters=k)

    # --- outputs ----------------------------------------------------------
    _write_cluster_outputs(out_dir, cfg, evaluation, model, centers, x, labels_all,
                           feature_names, prepared, k)

    voxel_rows, stat_rows, msi_rows_all, msi_matrices = [], [], [], {}
    figured: list[str] = []
    for case in prepared:
        vol = hab_volumes[case.case_id]
        vv = voxel_volume_mm3(case.affine)

        if cfg["metrics"]["volume"]:
            voxel_rows.append(volume_rows(case, vol, k, vv))
        if cfg["metrics"]["parameter_stats"]:
            stat_rows.extend(parameter_stat_rows(case, vol, k))
        if cfg["metrics"]["msi"]:
            msi_rows_all.append(msi_rows(case, vol, k))
            msi_matrices[case.case_id] = msi_matrix(vol, k)

        # habitat maps
        if cfg["output"]["save_habitat_maps"]:
            save_nifti(out_dir / "habitat_maps" / f"{case.case_id}_habitat.nii.gz",
                       vol, case.affine, case.header, dtype=np.uint8)

        # per-case figure
        if cfg["output"]["save_plots"] and _case_figure_allowed(cfg, len(figured)):
            figured.append(case.case_id)
            img = np.zeros(case.mask.shape, dtype=np.float32)
            img[case.mask] = case.raw[:, 0]
            plot_case_montage(img, vol, case.mask,
                              out_dir / "figures" / "cases" / f"{case.case_id}_habitat.png",
                              n_habitats=k)

        if cfg["metrics"].get("ithscore") and msi_rows_all and \
                msi_rows_all[-1]["case_id"] == case.case_id:
            msi_rows_all[-1].update(ith.compute_ithscore(_first_map(case), case.mask))

    if voxel_rows:
        write_csv(pd.DataFrame(voxel_rows), out_dir / "habitat_volumes.csv",
                  encoding=cfg["output"]["csv_encoding"], also_gbk=cfg["output"]["also_gbk"])

    if stat_rows:
        stats_df = pd.DataFrame(stat_rows)
        write_csv(stats_df, out_dir / "habitat_parameter_stats.csv",
                  encoding=cfg["output"]["csv_encoding"], also_gbk=cfg["output"]["also_gbk"])
        wide = wide_parameter_table(stats_df, stat="mean")
        write_csv(wide, out_dir / "habitat_parameter_stats_wide.csv",
                  encoding=cfg["output"]["csv_encoding"], also_gbk=cfg["output"]["also_gbk"])

    if msi_rows_all:
        msi_df = pd.DataFrame(msi_rows_all)
        cols = ["case_id", "msi_heterogeneity_score"] + MSI_FEATURE_NAMES
        for extra in ("ithscore", "ithscore_n_clusters"):
            if extra in msi_df.columns:
                cols.append(extra)
        msi_df = msi_df[[c for c in cols if c in msi_df.columns]]
        write_csv(msi_df, out_dir / "msi_features.csv",
                  encoding=cfg["output"]["csv_encoding"], also_gbk=cfg["output"]["also_gbk"])

        if cfg["output"].get("save_plots") and msi_matrices:
            from .reporting import plot_msi
            first_id = prepared[0].case_id
            plot_msi(msi_matrices[first_id], out_dir / "figures" / "msi_example.png")

        if cfg["metrics"].get("msi_matrix_long"):
            long_rows = []
            for case in prepared:
                for i in range(k + 1):
                    for j in range(k + 1):
                        long_rows.append({"case_id": case.case_id, "row": i, "col": j,
                                          "count": int(msi_matrices[case.case_id][i, j])})
            write_csv(pd.DataFrame(long_rows), out_dir / "msi_matrices_long.csv",
                      encoding=cfg["output"]["csv_encoding"])

    # normalisation parameters, per-case metadata, failures
    save_json({c.case_id: {name: p.as_dict() for name, p in c.norm_params.items()}
               for c in prepared}, out_dir / "normalization_params.json")
    write_csv(pd.DataFrame([{"case_id": c.case_id, "n_voxels": c.n_voxels,
                             **{k2: str(v2) for k2, v2 in c.meta.items()}}
                            for c in prepared]),
              out_dir / "cases_processed.csv", encoding=cfg["output"]["csv_encoding"])
    if failures:
        write_csv(pd.DataFrame(failures), out_dir / "cases_failed.csv",
                  encoding=cfg["output"]["csv_encoding"])

    manifest = _manifest(cfg, prepared, k, evaluation, feature_names, space,
                         model, time.time() - t0, failures)
    save_json(manifest, out_dir / "run_manifest.json")

    result.outputs = sorted(p.name for p in out_dir.iterdir())
    log.info("done in %.1fs -> %s", time.time() - t0, out_dir)
    return result


# --------------------------------------------------------------------------- #
# Helpers used above
# --------------------------------------------------------------------------- #

def _vol_from_labels(labels: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    vol = np.zeros(mask.shape, dtype=np.uint8)
    vol[mask] = (labels + 1).astype(np.uint8)
    if vol.max() > k:
        raise ValueError("label index exceeds number of habitats")
    return vol


def _first_map(case: PreparedCase) -> np.ndarray:
    out = np.zeros(case.mask.shape, dtype=np.float32)
    out[case.mask] = case.raw[:, 0]
    return out


def _case_figure_allowed(cfg: dict[str, Any], n_done: int) -> bool:
    limit = int(cfg["output"].get("max_case_figures", 20))
    return n_done < limit


def _write_cluster_outputs(out_dir: Path, cfg: dict[str, Any], evaluation: Any,
                           model: Any, centers: np.ndarray, x: np.ndarray,
                           labels: np.ndarray, feature_names: list[str],
                           prepared: Sequence[PreparedCase], k: int) -> None:
    enc = cfg["output"]["csv_encoding"]
    gbk = cfg["output"]["also_gbk"]

    summary = cluster_summary(x, labels, feature_names, k)
    write_csv(summary, out_dir / "cluster_summary.csv", encoding=enc, also_gbk=gbk)

    centers_df = pd.DataFrame(centers, columns=feature_names)
    centers_df.insert(0, "habitat", np.arange(1, k + 1))
    write_csv(centers_df, out_dir / "cluster_centers_normalized.csv", encoding=enc, also_gbk=gbk)

    # raw-unit means per habitat (pooled over patients)
    raw_all = np.vstack([c.raw for c in prepared])
    rows = []
    for h in range(k):
        sel = labels == h
        row = {"habitat": h + 1, "n_voxels": int(sel.sum())}
        for j, name in enumerate(feature_names):
            row[f"{name}_mean_raw"] = float(raw_all[sel, j].mean())
            row[f"{name}_sd_raw"] = float(raw_all[sel, j].std())
        rows.append(row)
    write_csv(pd.DataFrame(rows), out_dir / "cluster_centers_raw.csv", encoding=enc, also_gbk=gbk)

    lines = [
        "Habitat clustering summary",
        "=" * 60,
        f"pooled voxels : {x.shape[0]}",
        f"features      : {', '.join(feature_names)}",
        f"habitats (k)  : {k}",
    ]
    if evaluation is not None:
        lines += [
            f"selection     : {evaluation.strategy}",
            f"suggested k   : {evaluation.suggested}",
            f"votes         : {evaluation.votes}",
        ]
        lines.append("k\tinertia\tsilhouette\tCH\tDB")
        for rec in evaluation.as_records():
            sil = rec["silhouette"]
            sil_s = "n/a" if (isinstance(sil, float) and np.isnan(sil)) else f"{sil:.4f}"
            lines.append(f"{rec['k']}\t{rec['inertia']:.1f}\t{sil_s}\t"
                         f"{rec['calinski_harabasz']:.1f}\t{rec['davies_bouldin']:.4f}")
    (out_dir / "cluster_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if cfg["output"]["save_plots"]:
        if evaluation is not None:
            plot_k_selection(evaluation.table, k, out_dir / "figures" / "k_selection.png")
        plot_feature_space(x, labels, feature_names, centers,
                           out_dir / "figures" / "feature_space.png",
                           seed=int(cfg["project"]["seed"]))


def _manifest(cfg: dict[str, Any], prepared: Sequence[PreparedCase], k: int,
              evaluation: Any, feature_names: list[str], space: str,
              model: Any, elapsed: float, failures: list[dict[str, str]]) -> dict[str, Any]:
    import nibabel
    import scipy
    import sklearn

    return {
        "package": "cervical-habitat",
        "package_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nibabel": nibabel.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "config_path": cfg.get("_config_path"),
        "config_sha256": cfg.get("_config_sha256"),
        "seed": int(cfg["project"]["seed"]),
        "n_cases_requested": len(prepared) + len(failures),
        "n_cases_used": len(prepared),
        "failures": failures,
        "features": feature_names,
        "feature_space": space,
        "n_clusters": int(k),
        "k_selection": None if evaluation is None else {
            "strategy": evaluation.strategy,
            "suggested": evaluation.suggested,
            "votes": evaluation.votes,
        },
        "kmeans": {
            "init": "k-means++",
            "n_init": int(cfg["clustering"]["n_init"]),
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
        },
        "preprocessing": {
            "target_spacing_mm": cfg["resample"]["target_spacing"] if cfg["resample"]["enabled"] else None,
            "percentile_clip": cfg["preprocessing"]["percentile_clip"],
            "minmax": bool(cfg["preprocessing"]["minmax_scale"]),
            "nonfinite_strategy": cfg["preprocessing"]["fill_nonfinite"],
        },
        "dce": cfg["dce"] if prepared and "dce" in prepared[0].meta else None,
        "elapsed_seconds": round(elapsed, 2),
    }
