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


"""Notebook-derived exploratory, descriptive, and association analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import FEATURES
from .data import save_json


class ResearchAnalyzer:
    """Reusable analysis stage distilled from the original notebook."""

    binary_features = (
        "Gender",
        "Exclusive_Milky",
        "Smoke_Habit",
        "PureWater_Access",
        "Healthy_Toilet",
        "Difficult_Acess",
    )

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def overview(self) -> Dict[str, Any]:
        return {
            "rows": int(len(self.df)),
            "columns": int(len(self.df.columns)),
            "districts": int(self.df[FEATURES.district].nunique()) if FEATURES.district in self.df else 0,
            "stunting_prevalence": float(self.df[FEATURES.target].mean()),
            "missing_total": int(self.df.isna().sum().sum()),
            "exact_duplicates": int(self.df.duplicated().sum()),
        }

    def district_summary(self) -> pd.DataFrame:
        out = (
            self.df.groupby(FEATURES.district)[FEATURES.target]
            .agg(n="size", stunting_n="sum", stunting_prevalence="mean")
            .reset_index()
        )
        return out.sort_values("stunting_prevalence", ascending=False)

    def binary_associations(self) -> pd.DataFrame:
        rows = []
        for variable in self.binary_features:
            if variable not in self.df.columns:
                continue
            tab = pd.crosstab(self.df[variable], self.df[FEATURES.target]).reindex(
                index=[0, 1], columns=[0, 1], fill_value=0
            )
            a = int(tab.loc[1, 1])
            b = int(tab.loc[1, 0])
            c = int(tab.loc[0, 1])
            d = int(tab.loc[0, 0])
            odds_ratio = (a * d) / (b * c) if b * c else np.inf
            fisher_p = float(stats.fisher_exact([[a, b], [c, d]])[1])
            exposed_prev = a / (a + b) if a + b else np.nan
            unexposed_prev = c / (c + d) if c + d else np.nan
            rows.append(
                {
                    "variable": variable,
                    "exposed_stunting_n": a,
                    "exposed_nonstunting_n": b,
                    "unexposed_stunting_n": c,
                    "unexposed_nonstunting_n": d,
                    "stunting_prevalence_exposed": exposed_prev,
                    "stunting_prevalence_unexposed": unexposed_prev,
                    "odds_ratio": odds_ratio,
                    "fisher_p": fisher_p,
                }
            )
        return pd.DataFrame(rows)

    def chi_square_tests(self) -> pd.DataFrame:
        rows = []
        for variable in self.binary_features:
            if variable not in self.df.columns:
                continue
            tab = pd.crosstab(self.df[variable], self.df[FEATURES.target]).reindex(
                index=[0, 1], columns=[0, 1], fill_value=0
            )
            chi2, p, dof, _ = stats.chi2_contingency(tab.values, correction=False)
            denom = len(self.df) * min(tab.shape[0] - 1, tab.shape[1] - 1)
            cramer_v = float(np.sqrt(chi2 / denom)) if denom else np.nan
            rows.append(
                {"variable": variable, "chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": cramer_v}
            )
        return pd.DataFrame(rows)

    def malnutrition_crosstab(self) -> pd.DataFrame:
        table = pd.crosstab(self.df["Malnutrition_Level"], self.df[FEATURES.target]).reindex(
            columns=[0, 1], fill_value=0
        )
        table = table.rename(columns={0: "not_stunting_n", 1: "stunting_n"}).reset_index()
        table["total_n"] = table["not_stunting_n"] + table["stunting_n"]
        table["stunting_prevalence"] = table["stunting_n"] / table["total_n"]
        return table

    def save_all(self, json_dir: str | Path, data_dir: str | Path) -> Dict[str, Any]:
        json_dir = Path(json_dir)
        data_dir = Path(data_dir)
        json_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        overview = self.overview()
        district = self.district_summary()
        associations = self.binary_associations()
        chi = self.chi_square_tests()
        malnutrition = self.malnutrition_crosstab()

        save_json(overview, json_dir / "analysis_overview.json")
        district.to_csv(data_dir / "district_summary.csv", index=False)
        associations.to_csv(data_dir / "binary_associations.csv", index=False)
        chi.to_csv(data_dir / "chi_square_tests.csv", index=False)
        malnutrition.to_csv(data_dir / "malnutrition_crosstab.csv", index=False)
        return overview
