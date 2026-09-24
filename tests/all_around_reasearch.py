#!/usr/bin/env python3
from __future__ import annotations

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, masterofray/AI_Stunting_model"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.2.0"
__maintainer__ = "Aryanto, M.Si"
__email__      = "aryanto.dandan@gmail.com"
__created__    = "2026-08-31"
__modified__   = "2026-09-18"


"""Run the complete stunting research workflow outside a notebook.

Flow: raw stunting data -> data/EDA analysis -> XGBoost baseline -> Optuna tuning
-> tuned model evaluation -> SHAP/LIME/native XAI -> ZIP artifact.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from research.pipeline import ResearchPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the all-around stunting research pipeline")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "src" / "query" / "RawData.sql",
        help="CSV or SQL RawData source",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artefact")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xai-samples", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = ResearchPipeline(output_root=args.output, seed=args.seed)
    report = pipeline.run(
        source=args.source,
        n_trials=args.trials,
        test_size=args.test_size,
        xai_local_samples=args.xai_samples,
    )
    zip_path = pipeline.make_zip()
    tuned = report["tuned"]
    print("=" * 72)
    print("All-around research completed")
    print(f"Rows:              {report['rows']}")
    print(f"Train/Test:        {report['train_rows']}/{report['test_rows']}")
    print(f"Tuned ROC-AUC:     {tuned['roc_auc']:.4f}")
    print(f"Tuned F1:          {tuned['f1']:.4f}")
    print(f"Optuna trials:     {args.trials}")
    print(f"Artifact directory: {args.output}")
    print(f"Artifact ZIP:       {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
