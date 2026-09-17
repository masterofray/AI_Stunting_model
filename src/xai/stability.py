"""Stability and robustness diagnostics for post-hoc explanations.

Resamples the test/training set B times, recomputes SHAP values, and reports:
* Rank correlation of global feature importances across bootstraps.
* Top-k Jaccard overlap.
* Per-feature coefficient of variation.
* A "stability score" in [0, 1] per feature.

This is the kind of analysis reviewers ask for when you claim "feature X
is the most important driver".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from ..research.data import save_json


class StabilityExplainer:
    def __init__(self, fitted_pipeline, seed: int = 42):
        self.pipeline = fitted_pipeline
        self.preprocessor = fitted_pipeline.named_steps["preprocessor"]
        self.classifier = fitted_pipeline.named_steps["classifier"]
        self.feature_names = [
            str(x) for x in self.preprocessor.get_feature_names_out()
        ]
        self.seed = int(seed)

    def transformed(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(X), dtype=float)

    def shap_stability(
        self,
        X_test: pd.DataFrame,
        output_dir: str | Path,
        n_bootstrap: int = 30,
        subsample_frac: float = 0.8,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self.seed)
        matrix = self.transformed(X_test)
        explainer = shap.TreeExplainer(self.classifier)

        importances: List[np.ndarray] = []
        rank_matrix: List[np.ndarray] = []
        for _ in range(n_bootstrap):
            idx = rng.choice(
                len(matrix), size=max(2, int(subsample_frac * len(matrix))), replace=True
            )
            values = explainer.shap_values(matrix[idx])
            if isinstance(values, list):
                values = values[-1]
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, -1]
            imp = np.abs(values).mean(axis=0)
            importances.append(imp)
            rank_matrix.append((-imp).argsort().argsort() + 1)  # rank 1 = most important

        imp_arr = np.vstack(importances)
        rank_arr = np.vstack(rank_matrix)

        mean_imp = imp_arr.mean(axis=0)
        std_imp = imp_arr.std(axis=0)
        cv = std_imp / (np.abs(mean_imp) + 1e-12)
        mean_rank = rank_arr.mean(axis=0)
        rank_std = rank_arr.std(axis=0)

        # Top-k Jaccard across pairs of bootstraps
        top_sets = [set(np.argsort(-imp)[:top_k]) for imp in importances]
        jaccards = []
        for a in range(len(top_sets)):
            for b in range(a + 1, len(top_sets)):
                inter = len(top_sets[a] & top_sets[b])
                union = len(top_sets[a] | top_sets[b])
                jaccards.append(inter / union if union else 0.0)
        mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

        # Pairwise Spearman across bootstraps
        spearman_pairs = []
        for a in range(len(importances)):
            for b in range(a + 1, len(importances)):
                rho, _ = spearmanr(importances[a], importances[b])
                if not np.isnan(rho):
                    spearman_pairs.append(rho)
        mean_spearman = float(np.mean(spearman_pairs)) if spearman_pairs else 0.0

        # Per-feature stability score = 1 / (1 + cv)
        stability_score = 1.0 / (1.0 + cv)

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": mean_imp,
                "std_abs_shap": std_imp,
                "cv": cv,
                "mean_rank": mean_rank,
                "rank_std": rank_std,
                "stability_score": stability_score,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        df.to_csv(output_dir / "shap_stability.csv", index=False)

        self._plot_stability(df, output_dir / "shap_stability.png")
        summary = {
            "n_bootstrap": n_bootstrap,
            "subsample_frac": subsample_frac,
            "top_k": top_k,
            "mean_topk_jaccard": mean_jaccard,
            "mean_spearman_rank_correlation": mean_spearman,
            "features_below_stability_0_5": int((stability_score < 0.5).sum()),
            "most_stable_features": df.head(10)["feature"].tolist(),
        }
        save_json(summary, output_dir / "shap_stability_summary.json")
        return summary

    def _plot_stability(self, df: pd.DataFrame, path: Path) -> None:
        top = df.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(top) + 1)))
        ax.barh(top["feature"], top["stability_score"], color="#8172B2")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("stability score  (1 / (1 + CV))")
        ax.set_title("Feature-importance stability across bootstraps")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)