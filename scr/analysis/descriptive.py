"""Descriptive Statistics Module"""
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from scipy import stats
from config.settings import REQUIRED_COLUMNS, BINARY_VARS, TARGET_VAR, DISTRICT_COL, MALNUTRITION_COL, analysis_config

class DescriptiveAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        for col in REQUIRED_COLUMNS:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def _wilson_ci(self, x: int, n: int, z: float = None) -> Tuple[float, float]:
        if z is None:
            z = analysis_config.Z_SCORE_95CI
        if n <= 0:
            return np.nan, np.nan
        p = x / n
        den = 1 + z * z / n
        center = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return max(0.0, center - half), min(1.0, center + half)

    def district_summary(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        out = df.groupby(DISTRICT_COL)[TARGET_VAR].agg(
            n="size", stunting_n="sum", stunting_prev="mean"
        ).reset_index()
        ci = [self._wilson_ci(int(r.stunting_n), int(r.n)) for _, r in out.iterrows()]
        out["ci95_low"] = [x[0] for x in ci]
        out["ci95_high"] = [x[1] for x in ci]
        return out.sort_values("stunting_prev", ascending=False)

    def malnutrition_summary(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if df is None:
            df = self.df
        tab = pd.crosstab(df[MALNUTRITION_COL], df[TARGET_VAR]).reindex(columns=[0, 1], fill_value=0)
        out = tab.reset_index().rename(columns={0: "stunting_0_n", 1: "stunting_1_n"})
        out["stunting_prev"] = out["stunting_1_n"] / (out["stunting_0_n"] + out["stunting_1_n"])
        out["total_n"] = out["stunting_0_n"] + out["stunting_1_n"]
        chi2, p, dof, _ = stats.chi2_contingency(tab.values)
        meta = {"chi2": float(chi2), "p": float(p), "dof": int(dof), "n_levels": int(df[MALNUTRITION_COL].nunique())}
        return out, meta

    def binary_variable_summary(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        summaries = []
        for var in BINARY_VARS:
            if var in df.columns:
                n_ones = int((df[var] == 1).sum())
                n_zeros = int((df[var] == 0).sum())
                n_total = n_ones + n_zeros
                by_stunting = df.groupby(TARGET_VAR)[var].mean().reset_index()
                mean_1 = float(by_stunting[by_stunting[TARGET_VAR] == 1]["mean"].iloc[0]) if len(by_stunting) > 1 else np.nan
                mean_0 = float(by_stunting[by_stunting[TARGET_VAR] == 0]["mean"].iloc[0]) if len(by_stunting) > 1 else np.nan
                summaries.append({
                    "variable": var, "n_ones": n_ones, "n_zeros": n_zeros,
                    "prop_ones": n_ones / n_total if n_total > 0 else 0.0,
                    "mean_when_stunting_1": mean_1, "mean_when_stunting_0": mean_0,
                })
        return pd.DataFrame(summaries)

    def overall_summary(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if df is None:
            df = self.df
        n_total = len(df)
        n_stunting = int(df[TARGET_VAR].sum())
        return {
            "n_total": n_total, "n_stunting": n_stunting,
            "stunting_prevalence": n_stunting / n_total if n_total > 0 else 0.0,
            "n_districts": int(df[DISTRICT_COL].nunique()),
        }

    def correlation_matrix(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[numeric_cols].corr()

    def save_all_summaries(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.district_summary().to_csv(output_dir / "district_summary.csv", index=False)
        mal_sum, mal_meta = self.malnutrition_summary()
        mal_sum.to_csv(output_dir / "malnutrition_summary.csv", index=False)
        with open(output_dir / "malnutrition_association.json", "w") as f:
            json.dump(mal_meta, f, indent=2)
        self.binary_variable_summary().to_csv(output_dir / "binary_variable_summary.csv", index=False)
        with open(output_dir / "overall_summary.json", "w") as f:
            json.dump(self.overall_summary(), f, indent=2)
        self.correlation_matrix().to_csv(output_dir / "correlation_matrix.csv")
