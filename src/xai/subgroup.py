"""Subgroup-aware explanations.

Asks the question: *do different subpopulations receive the same reasons?*
Two complementary analyses:

1. ICE (Individual Conditional Expectation) curves per subgroup, with
   variance bands — reveals heterogeneity in the learned response.
2. SHAP-importance divergence between subgroups — a JSD-style distance
   between normalised importance vectors, revealing features whose
   importance is subgroup-specific.
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

from ..research.data import save_json


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    )


class SubgroupExplainer:
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

    # ---------- SHAP importance divergence across subgroups ----------

    def shap_divergence(
        self,
        X_test: pd.DataFrame,
        subgroup_col: str,
        output_dir: str | Path,
        max_rows: int = 800,
    ) -> pd.DataFrame:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        X_sub = X_test.iloc[:max_rows]
        matrix = self.transformed(X_sub)
        explainer = shap.TreeExplainer(self.classifier)
        values = explainer.shap_values(matrix)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]

        groups = X_sub[subgroup_col].astype(str).values
        unique_groups = sorted(pd.unique(groups))
        importance_by_group: Dict[str, np.ndarray] = {}
        for g in unique_groups:
            mask = groups == g
            if mask.sum() < 10:
                continue
            importance_by_group[g] = np.abs(values[mask]).mean(axis=0)

        if len(importance_by_group) < 2:
            return pd.DataFrame(
                [{"status": "skipped", "reason": "fewer than 2 valid subgroups"}]
            )

        rows: List[Dict[str, Any]] = []
        group_names = list(importance_by_group.keys())
        for i, gi in enumerate(group_names):
            for j, gj in enumerate(group_names):
                if j <= i:
                    continue
                jsd = _js_divergence(importance_by_group[gi], importance_by_group[gj])
                rows.append(
                    {
                        "group_a": gi,
                        "group_b": gj,
                        "js_divergence": jsd,
                    }
                )
        divergence = pd.DataFrame(rows).sort_values("js_divergence", ascending=False)
        divergence.to_csv(output_dir / "subgroup_shap_divergence.csv", index=False)

        # Per-feature importance table for plotting
        importance_df = pd.DataFrame(importance_by_group, index=self.feature_names)
        importance_df.to_csv(output_dir / "subgroup_shap_importance.csv")

        self._plot_group_importance(
            importance_df, output_dir / "subgroup_shap_importance.png"
        )
        save_json(
            {
                "subgroup_col": subgroup_col,
                "groups": group_names,
                "max_js_divergence": float(divergence["js_divergence"].max())
                if not divergence.empty
                else 0.0,
                "pairwise": divergence.to_dict(orient="records"),
            },
            output_dir / "subgroup_shap_divergence_summary.json",
        )
        return divergence

    # ---------- ICE curves ----------

    def ice_curves(
        self,
        X_test: pd.DataFrame,
        feature: str,
        subgroup_col: str,
        output_dir: str | Path,
        grid_resolution: int = 25,
        max_rows: int = 400,
    ) -> pd.DataFrame:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix = self.transformed(X_test.iloc[:max_rows])
        f_idx = self.feature_names.index(feature) if feature in self.feature_names else None
        if f_idx is None:
            # feature not in transformed space: try matching a prefix
            candidates = [i for i, n in enumerate(self.feature_names) if feature in n]
            if not candidates:
                return pd.DataFrame(
                    [{"status": "skipped", "reason": f"{feature} not in transformed space"}]
                )
            f_idx = candidates[0]

        grid = np.linspace(
            np.quantile(matrix[:, f_idx], 0.02),
            np.quantile(matrix[:, f_idx], 0.98),
            grid_resolution,
        )
        subgroup_labels = X_test.iloc[:max_rows][subgroup_col].astype(str).values

        curves = np.zeros((matrix.shape[0], grid_resolution))
        for k, v in enumerate(grid):
            M = matrix.copy()
            M[:, f_idx] = v
            curves[:, k] = self.classifier.predict_proba(M)[:, 1]

        rows: List[Dict[str, Any]] = []
        for g in sorted(pd.unique(subgroup_labels)):
            mask = subgroup_labels == g
            if mask.sum() < 5:
                continue
            mean_curve = curves[mask].mean(axis=0)
            std_curve = curves[mask].std(axis=0)
            for k, v in enumerate(grid):
                rows.append(
                    {
                        "subgroup": g,
                        "feature_value": float(v),
                        "mean_prediction": float(mean_curve[k]),
                        "std_prediction": float(std_curve[k]),
                    }
                )
        result = pd.DataFrame(rows)
        result.to_csv(output_dir / f"ice_{feature}_{subgroup_col}.csv", index=False)
        self._plot_ice(
            result,
            feature,
            subgroup_col,
            output_dir / f"ice_{feature}_{subgroup_col}.png",
        )
        return result

    # ---------- plotting ----------

    def _plot_group_importance(self, importance_df: pd.DataFrame, path: Path) -> None:
        top_features = importance_df.mean(axis=1).sort_values(ascending=False).head(15).index
        sub = importance_df.loc[top_features]
        fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(top_features) + 1)))
        width = 0.8 / max(len(sub.columns), 1)
        idx = np.arange(len(top_features))
        for k, col in enumerate(sub.columns):
            ax.barh(idx + k * width, sub[col].values, height=width, label=str(col))
        ax.set_yticks(idx + width * (len(sub.columns) - 1) / 2)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel("mean |SHAP|")
        ax.set_title("SHAP importance by subgroup")
        ax.legend(title="subgroup", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_ice(
        self,
        df: pd.DataFrame,
        feature: str,
        subgroup_col: str,
        path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for g, sub in df.groupby("subgroup"):
            ax.plot(sub["feature_value"], sub["mean_prediction"], label=str(g))
            ax.fill_between(
                sub["feature_value"],
                sub["mean_prediction"] - sub["std_prediction"],
                sub["mean_prediction"] + sub["std_prediction"],
                alpha=0.15,
            )
        ax.set_xlabel(feature)
        ax.set_ylabel("mean predicted P(stunting)")
        ax.set_title(f"ICE of {feature} by {subgroup_col}")
        ax.legend(title=subgroup_col, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)