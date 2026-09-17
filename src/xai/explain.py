"""SHAP, LIME, and native XGBoost explanations with file outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LimeTabularExplainer = None
    LIME_AVAILABLE = False

from ..research.data import save_json


class XAIExplainer:
    """Produce global and local explanations for the fitted pipeline."""

    def __init__(self, fitted_pipeline, seed: int = 42):
        self.pipeline = fitted_pipeline
        self.seed = seed
        self.preprocessor = fitted_pipeline.named_steps["preprocessor"]
        self.classifier = fitted_pipeline.named_steps["classifier"]
        self._feature_names = [str(x) for x in self.preprocessor.get_feature_names_out()]

    def transformed(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(X), dtype=float)

    def native_importance(self, output_dir: str | Path) -> pd.DataFrame:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        booster = self.classifier.get_booster()
        records = []
        for importance_type in ("gain", "weight", "cover", "total_gain", "total_cover"):
            score = booster.get_score(importance_type=importance_type)
            for idx, feature in enumerate(self._feature_names):
                records.append(
                    {
                        "feature": feature,
                        "importance_type": importance_type,
                        "score": float(score.get(f"f{idx}", 0.0)),
                    }
                )
        result = pd.DataFrame(records)
        result.to_csv(output_dir / "native_xgboost_importance.csv", index=False)
        return result

    def shap_global(self, X_train: pd.DataFrame, X_test: pd.DataFrame, output_dir: str | Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        background = self.transformed(X_train)
        test_matrix = self.transformed(X_test)
        explainer = shap.TreeExplainer(self.classifier, background)
        values = explainer.shap_values(test_matrix)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]

        np.save(output_dir / "shap_values.npy", values)
        pd.DataFrame(values, columns=self._feature_names).to_csv(
            output_dir / "shap_values.csv", index=False
        )
        mean_abs = np.abs(values).mean(axis=0)
        importance = pd.DataFrame(
            {"feature": self._feature_names, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(output_dir / "shap_global_importance.csv", index=False)

        plt.figure(figsize=(11, 7))
        shap.summary_plot(
            values,
            test_matrix,
            feature_names=self._feature_names,
            plot_type="bar",
            show=False,
            max_display=min(25, len(self._feature_names)),
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_bar.png", dpi=220, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(11, 7))
        shap.summary_plot(
            values,
            test_matrix,
            feature_names=self._feature_names,
            show=False,
            max_display=min(25, len(self._feature_names)),
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=220, bbox_inches="tight")
        plt.close()

        return {
            "feature_count": len(self._feature_names),
            "n_test_rows": int(len(X_test)),
            "top_features": importance.head(15).to_dict(orient="records"),
            "shap_output_space": "raw_model_output",
        }

    def shap_local(
        self,
        X_test: pd.DataFrame,
        output_dir: str | Path,
        max_samples: int = 5,
    ) -> pd.DataFrame:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix = self.transformed(X_test)
        explainer = shap.TreeExplainer(self.classifier)
        values = explainer.shap_values(matrix)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        rows = []
        sample_count = min(max_samples, len(X_test))
        for position in range(sample_count):
            contribution = values[position]
            top = np.argsort(np.abs(contribution))[::-1][:10]
            for idx in top:
                rows.append(
                    {
                        "sample_index": int(X_test.index[position]),
                        "feature": self._feature_names[idx],
                        "shap_value": float(contribution[idx]),
                        "abs_shap_value": float(abs(contribution[idx])),
                    }
                )
            fig, ax = plt.subplots(figsize=(10, 6))
            local = pd.DataFrame(
                {
                    "feature": np.array(self._feature_names)[top],
                    "shap_value": contribution[top],
                }
            ).sort_values("shap_value")
            ax.barh(local["feature"], local["shap_value"])
            ax.axvline(0, linewidth=1)
            ax.set_title(f"SHAP local explanation: test row {X_test.index[position]}")
            ax.set_xlabel("SHAP value")
            fig.tight_layout()
            fig.savefig(
                output_dir / f"shap_local_{X_test.index[position]}.png",
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(fig)
        result = pd.DataFrame(rows)
        result.to_csv(output_dir / "shap_local_explanations.csv", index=False)
        return result

    def lime_local(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        output_dir: str | Path,
        max_samples: int = 5,
    ) -> pd.DataFrame:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_matrix = self.transformed(X_train)
        test_matrix = self.transformed(X_test)
        if not LIME_AVAILABLE:
            return pd.DataFrame(
                [{"status": "skipped", "reason": "LIME is not installed"}]
            )
        explainer = LimeTabularExplainer(
            training_data=train_matrix,
            feature_names=self._feature_names,
            class_names=["Not stunting", "Stunting"],
            mode="classification",
            discretize_continuous=True,
            random_state=self.seed,
        )
        rows = []
        for position in range(min(max_samples, len(X_test))):
            instance = test_matrix[position]
            explanation = explainer.explain_instance(
                instance,
                self.classifier.predict_proba,
                num_features=min(15, len(self._feature_names)),
            )
            for feature, weight in explanation.as_list(label=1):
                rows.append(
                    {
                        "sample_index": int(X_test.index[position]),
                        "feature_rule": str(feature),
                        "weight": float(weight),
                    }
                )
            fig = explanation.as_pyplot_figure(label=1)
            fig.tight_layout()
            fig.savefig(
                output_dir / f"lime_local_{X_test.index[position]}.png",
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(fig)
        result = pd.DataFrame(rows)
        result.to_csv(output_dir / "lime_local_explanations.csv", index=False)
        return result

    def run_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        output_dir: str | Path,
        max_local_samples: int = 5,
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        native = self.native_importance(output_dir)
        shap_summary = self.shap_global(X_train, X_test, output_dir)
        shap_local = self.shap_local(X_test, output_dir, max_samples=max_local_samples)
        lime_local = self.lime_local(X_train, X_test, output_dir, max_samples=max_local_samples)
        summary = {
            "method": ["XGBoost native importance", "SHAP", "LIME"],
            "lime_available": LIME_AVAILABLE,
            "native_importance_rows": int(len(native)),
            "shap_local_rows": int(len(shap_local)),
            "lime_local_rows": int(len(lime_local)),
            "shap": shap_summary,
            "transformed_feature_names": self._feature_names,
        }
        save_json(summary, output_dir / "xai_summary.json")
        return summary
