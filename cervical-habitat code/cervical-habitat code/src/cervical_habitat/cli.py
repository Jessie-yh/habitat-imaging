"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import load_config
from .pipeline import run_pipeline


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("nibabel").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cervical-habitat",
        description="Voxel-wise K-means habitat analysis of cervical cancer on multi-parametric MRI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full pipeline")
    run.add_argument("-c", "--config", required=True, help="YAML configuration file")
    run.add_argument("-o", "--output-dir", help="override project.output_dir")
    run.add_argument("-k", "--n-clusters", type=int, help="override clustering.n_clusters")
    run.add_argument("-n", "--limit", type=int, help="only process the first N cases (debug)")
    run.add_argument("--no-plots", action="store_true", help="skip all figures")
    run.add_argument("--gbk", action="store_true", help="also write GBK copies of every CSV")
    run.add_argument("--evaluate-only", action="store_true",
                     help="stop after cluster-number evaluation (no habitat maps)")
    run.add_argument("-v", "--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["project"]["output_dir"] = args.output_dir
    if args.n_clusters:
        cfg["clustering"]["n_clusters"] = args.n_clusters
        cfg["clustering"]["evaluate"] = False if args.evaluate_only else cfg["clustering"]["evaluate"]
    if args.no_plots:
        cfg["output"]["save_plots"] = False
    if args.gbk:
        cfg["output"]["also_gbk"] = True
    if args.evaluate_only:
        cfg["clustering"]["evaluate"] = True

    result = run_pipeline(cfg, limit=args.limit)
    print(f"\nfinished: {result.n_cases} cases, k={result.n_clusters}")
    print(f"output: {result.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
