"""Feature interaction analysis for the stunting XGBoost model.

Two complementary views:
1. TreeSHAP interaction values -> exact pairwise interaction strength.
2. Friedman's H-statistic -> model-agnostic interaction strength.

The output is a ranked edge list (a "social determinants network") plus
heatmaps that can be dropped into a paper as figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ..research.data import save_json


class InteractionExplainer:
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

    # ---------- TreeSHAP interactions ----------

    def shap_interactions(
        self,
        X_test: pd.DataFrame,
        output_dir: str | Path,
        top_k: int = 20,
        max_rows: int = 400,
    ) -> pd.DataFrame:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix = self.transformed(X_test)[:max_rows]
        explainer = shap.TreeExplainer(self.classifier)
        inter = explainer.shap_interaction_values(matrix)
        inter = np.asarray(inter)
        if inter.ndim == 4:
            inter = inter[:, :, :, -1]
        # absolute mean over samples
        strength = np.abs(inter).mean(axis=0)

        edges: List[Dict[str, Any]] = []
        n = len(self.feature_names)
        for i in range(n):
            for j in range(i + 1, n):
                edges.append(
                    {
                        "feature_i": self.feature_names[i],
                        "feature_j": self.feature_names[j],
                        "interaction_strength": float(strength[i, j]),
                    }
                )
        edges_df = (
            pd.DataFrame(edges)
            .sort_values("interaction_strength", ascending=False)
            .reset_index(drop=True)
        )
        edges_df.to_csv(output_dir / "shap_interactions.csv", index=False)

        self._plot_heatmap(strength, output_dir / "shap_interaction_heatmap.png")
        self._plot_top_edges(edges_df.head(top_k), output_dir / "shap_interaction_top.png")
        save_json(
            {
                "n_features": n,
                "n_test_rows_used": int(matrix.shape[0]),
                "top_edges": edges_df.head(top_k).to_dict(orient="records"),
            },
            output_dir / "shap_interactions_summary.json",
        )
        return edges_df

    # ---------- Friedman's H-statistic ----------

    def friedman_h(
        self,
        X_test: pd.DataFrame,
        output_dir: str | Path,
        pairs: Optional[List[tuple]] = None,
        grid_resolution: int = 12,
        max_rows: int = 1500,
    ) -> pd.DataFrame:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix = self.transformed(X_test)[:max_rows]
        n = len(self.feature_names)
        if pairs is None:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        pred = lambda M: self.classifier.predict_proba(M)[:, 1]
        records: List[Dict[str, Any]] = []
        for i, j in pairs:
            h2 = self._h_statistic(matrix, i, j, pred, grid_resolution)
            records.append(
                {
                    "feature_i": self.feature_names[i],
                    "feature_j": self.feature_names[j],
                    "h_squared": h2,
                }
            )
        df = (
            pd.DataFrame(records)
            .sort_values("h_squared", ascending=False)
            .reset_index(drop=True)
        )
        df.to_csv(output_dir / "friedman_h.csv", index=False)

        self._plot_top_edges(
            df.head(20), output_dir / "friedman_h_top.png", value_col="h_squared"
        )
        return df

    # ---------- internals ----------

    def _h_statistic(
        self,
        matrix: np.ndarray,
        i: int,
        j: int,
        pred,
        grid: int,
    ) -> float:
        qs = np.linspace(0.05, 0.95, grid)
        xi = np.quantile(matrix[:, i], qs)
        xj = np.quantile(matrix[:, j], qs)

        def partial_dependence(fixed_i: Optional[np.ndarray], fixed_j: Optional[np.ndarray]) -> np.ndarray:
            grid_out = np.zeros((len(xi), len(xj)))
            base = matrix.copy()
            for a, vi in enumerate(xi):
                for b, vj in enumerate(xj):
                    M = base
                    if fixed_i is not None:
                        M = M.copy()
                        M[:, i] = vi
                    if fixed_j is not None:
                        M = M.copy()
                        M[:, j] = vj
                    grid_out[a, b] = pred(M).mean()
            return grid_out

        pd_ij = partial_dependence(True, True)
        pd_i = partial_dependence(True, None)
        pd_j = partial_dependence(None, True)
        numerator = np.var(pd_ij - pd_i - pd_j + pd_ij.mean())
        denominator = np.var(pd_ij) + 1e-12
        return float(numerator / denominator)

    def _plot_heatmap(self, matrix: np.ndarray, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(self.feature_names)),
                                        max(5, 0.35 * len(self.feature_names))))
        im = ax.imshow(matrix, cmap="magma")
        ax.set_xticks(range(len(self.feature_names)))
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_xticklabels(self.feature_names, rotation=90, fontsize=7)
        ax.set_yticklabels(self.feature_names, fontsize=7)
        fig.colorbar(im, ax=ax, label="mean |SHAP interaction|")
        ax.set_title("SHAP interaction strength")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_top_edges(
        self,
        edges: pd.DataFrame,
        path: Path,
        value_col: str = "interaction_strength",
    ) -> None:
        if edges.empty:
            return
        labels = [f"{a} ↔ {b}" for a, b in zip(edges["feature_i"], edges["feature_j"])]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(labels) + 1)))
        ax.barh(labels[::-1], edges[value_col].values[::-1], color="#C44E52")
        ax.set_xlabel(value_col)
        ax.set_title(f"Top interaction pairs ({value_col})")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)