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



import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .analysis import ResearchAnalyzer
from .config import MODEL, OUTPUT
from .data import load_raw_data, prepare_model_frame, save_json, stratified_split, validate_dataset
from .visualization import plot_binary_summary, plot_district_prevalence
from .xgboost_model import XGBoostResearchModel


class ResearchPipeline:
    def __init__(self, output_root: str | Path = OUTPUT.root, seed: int = MODEL.seed):
        self.output = Path(output_root)
        self.seed = seed
        self.output_config = OUTPUT
        self.output_config = type(OUTPUT)(root=self.output)
        self.output_config.create()

    def _copy_csv_outputs(self, df: pd.DataFrame, X_train, X_test, y_train, y_test) -> None:
        df.to_csv(self.output_config.data / "stunting_raw.csv", index=False)
        X_train.to_csv(self.output_config.data / "x_train.csv", index=False)
        X_test.to_csv(self.output_config.data / "x_test.csv", index=False)
        pd.DataFrame({"Stunting": y_train}).to_csv(self.output_config.data / "y_train.csv", index=False)
        pd.DataFrame({"Stunting": y_test}).to_csv(self.output_config.data / "y_test.csv", index=False)

    def run(
        self,
        source: str | Path,
        n_trials: int = MODEL.n_trials,
        test_size: float = MODEL.test_size,
        xai_local_samples: int = 5,
    ) -> Dict[str, Any]:
        started = datetime.now(timezone.utc)
        raw_df = load_raw_data(source)
        validation = validate_dataset(raw_df)
        save_json(validation, self.output_config.json / "data_quality.json")
        if not validation["valid"]:
            raise ValueError(f"Dataset validation failed: {validation['errors']}")

        analyzer = ResearchAnalyzer(raw_df)
        overview = analyzer.save_all(self.output_config.json, self.output_config.data)
        plot_district_prevalence(raw_df, self.output_config.graphics / "district_prevalence.png")
        plot_binary_summary(raw_df, self.output_config.graphics / "binary_covariates_by_stunting.png")

        X, y = prepare_model_frame(raw_df)
        X_train, X_test, y_train, y_test = stratified_split(
            X, y, test_size=test_size, seed=self.seed
        )
        self._copy_csv_outputs(raw_df, X_train, X_test, y_train, y_test)

        model = XGBoostResearchModel(seed=self.seed, n_trials=n_trials)
        model.fit_baseline(X_train, y_train)
        baseline_pred, baseline_prob = model.predict(X_test)
        baseline_metrics = model.evaluate(y_test, baseline_pred, baseline_prob)
        save_json(baseline_metrics, self.output_config.json / "xgboost_baseline_metrics.json")

        model.tune(X_train, y_train)
        model.save_tuning_report(self.output_config.json / "xgboost_optuna_best.json")
        model.fit_best(X_train, y_train)
        pred, prob = model.predict(X_test)
        tuned_metrics = model.save_report(
            y_test,
            pred,
            prob,
            self.output_config.graphics,
            self.output_config.json,
            self.output_config.data,
            prefix="xgboost_tuned",
        )
        model.save(self.output_config.model / "xgboost_pipeline.joblib")
        # XAI is isolated so the model pipeline stays usable without interactive notebooks.
        from xai.explain import XAIExplainer

        xai = XAIExplainer(model.pipeline, seed=self.seed)
        xai_summary = xai.run_all(
            X_train,
            X_test,
            self.output_config.xai,
            max_local_samples=xai_local_samples,
        )

        report = {
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "rows": int(len(raw_df)),
            "features_before_encoding": list(X.columns),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "baseline": baseline_metrics,
            "tuned": tuned_metrics,
            "optuna_best_value": float(model.study.best_value),
            "optuna_best_params": dict(model.study.best_params),
            "xai": xai_summary,
            "overview": overview,
        }
        save_json(report, self.output_config.json / "all_around_reasearch_summary.json")
        return report

    def make_zip(self) -> Path:
        """Package generated outputs without recursively embedding the ZIP itself."""
        archive_path = self.output_config.zip_path
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if archive_path.exists():
            archive_path.unlink()
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in self.output.rglob("*"):
                if not path.is_file() or path.resolve() == archive_path.resolve():
                    continue
                archive.write(path, path.relative_to(self.output))
        return archive_path
