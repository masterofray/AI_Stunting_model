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



"""Statistical Analysis Module"""
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from scipy import stats
from config.settings import REQUIRED_COLUMNS, BINARY_VARS, TARGET_VAR, DISTRICT_COL, MALNUTRITION_COL, analysis_config

class StatisticalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        for col in REQUIRED_COLUMNS:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def binary_association(self, variable: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if df is None:
            df = self.df
        tab = pd.crosstab(df[variable], df[TARGET_VAR]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        a, b = int(tab.loc[1, 1]), int(tab.loc[1, 0])
        c, d = int(tab.loc[0, 1]), int(tab.loc[0, 0])
        rr = (a / (a + b)) / (c / (c + d)) if a + b and c + d and c else np.inf
        orr = (a * d) / (b * c) if b * c else np.inf
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        log_or = math.log((aa * dd) / (bb * cc))
        se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
        fisher_p = float(stats.fisher_exact([[a, b], [c, d]])[1])
        return {
            "variable": variable, "exposed_stunting_n": a, "exposed_nonstunting_n": b,
            "unexposed_stunting_n": c, "unexposed_nonstunting_n": d,
            "stunting_prev_exposed": a / (a + b) if a + b else np.nan,
            "stunting_prev_unexposed": c / (c + d) if c + d else np.nan,
            "risk_ratio": rr, "odds_ratio": orr,
            "or_ci_low": math.exp(log_or - analysis_config.Z_SCORE_95CI * se),
            "or_ci_high": math.exp(log_or + analysis_config.Z_SCORE_95CI * se),
            "fisher_p": fisher_p, "n_total": a + b + c + d,
        }

    def all_binary_associations(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        rows = [self.binary_association(var, df) for var in BINARY_VARS]
        return pd.DataFrame(rows)

    def chi_square_tests(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        chi_rows = []
        for var in BINARY_VARS:
            if var in df.columns:
                tab = pd.crosstab(df[var], df[TARGET_VAR]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                chi2, p, dof, _ = stats.chi2_contingency(tab.values, correction=False)
                v_cramer = math.sqrt(chi2 / (len(df) * min(tab.shape[0] - 1, tab.shape[1] - 1)))
                chi_rows.append({"variable": var, "chi2": chi2, "p": p, "dof": dof, "cramers_v": v_cramer})
        chi = pd.DataFrame(chi_rows)
        if hasattr(stats, "false_discovery_control"):
            chi["p_fdr_bh"] = stats.false_discovery_control(chi["p"].to_numpy(), method=analysis_config.FDR_METHOD)
        else:
            chi["p_fdr_bh"] = np.nan
        return chi

    def district_association(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if df is None:
            df = self.df
        tab = pd.crosstab(df[DISTRICT_COL], df[TARGET_VAR])
        chi2, p, dof, _ = stats.chi2_contingency(tab.values)
        v = math.sqrt(chi2 / (len(df) * min(tab.shape[0] - 1, tab.shape[1] - 1)))
        return {"chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": float(v)}

    def association_tables(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df is None:
            df = self.df
        return self.all_binary_associations(df), self.chi_square_tests(df)

    def deduplication_sensitivity(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        raw = self.all_binary_associations(df)
        dedup_df = df.drop_duplicates()
        dedup = self.all_binary_associations(dedup_df)
        sens_rows = []
        for var in BINARY_VARS:
            r = raw[raw["variable"] == var].iloc[0]
            d = dedup[dedup["variable"] == var].iloc[0]
            sens_rows.append({
                "variable": var, "raw_or": r["odds_ratio"], "raw_fisher_p": r["fisher_p"],
                "dedup_or": d["odds_ratio"], "dedup_fisher_p": d["fisher_p"],
            })
        return pd.DataFrame(sens_rows)

    def save_all_tests(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        binary, chi = self.association_tables()
        binary.to_csv(output_dir / "binary_associations.csv", index=False)
        chi.to_csv(output_dir / "chi_square_tests.csv", index=False)
        with open(output_dir / "district_association.json", "w") as f:
            json.dump(self.district_association(), f, indent=2)
        self.deduplication_sensitivity().to_csv(output_dir / "deduplication_sensitivity.csv", index=False)