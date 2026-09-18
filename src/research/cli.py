#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, masterofray/AI_Stunting_model"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.2.0"
__maintainer__ = "Aryanto, M.Si"
__email__      = "aryanto.dandan@gmail.com"
__created__    = "2026-08-31"
__modified__   = "2026-09-18"



"""Command-line entry point for the modular stunting research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import MODEL, OUTPUT
from .pipeline import ResearchPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stunting research pipeline")
    parser.add_argument("--source", type=Path, required=True, help="CSV or RawData SQL file")
    parser.add_argument("--output", type=Path, default=OUTPUT.root)
    parser.add_argument("--trials", type=int, default=MODEL.n_trials)
    parser.add_argument("--test-size", type=float, default=MODEL.test_size)
    parser.add_argument("--seed", type=int, default=MODEL.seed)
    parser.add_argument("--xai-samples", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pipeline = ResearchPipeline(output_root=args.output, seed=args.seed)
    report = pipeline.run(
        source=args.source,
        n_trials=args.trials,
        test_size=args.test_size,
        xai_local_samples=args.xai_samples,
    )
    zip_path = pipeline.make_zip()
    print(f"Tuned ROC-AUC: {report['tuned']['roc_auc']:.4f}")
    print(f"Artifacts: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
