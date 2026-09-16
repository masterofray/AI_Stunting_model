"""Bayesian Analysis Module"""
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize
from config.settings import REQUIRED_COLUMNS, TARGET_VAR, DISTRICT_COL, analysis_config

class BayesianAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        for col in REQUIRED_COLUMNS:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def bayesian_beta_binomial(self, df_district: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if df_district is None:
            df_district = self.df.groupby(DISTRICT_COL)[TARGET_VAR].agg(stunted="sum", n="size").reset_index()
        y = df_district["stunted"].to_numpy(dtype=float)
        n = df_district["n"].to_numpy(dtype=float)

        def neg_ll(theta):
            alpha, beta = np.exp(theta)
            ll = np.sum(
                gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
                + gammaln(y + alpha) + gammaln(n - y + beta) - gammaln(n + alpha + beta)
                + gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
            )
            return -ll

        res = minimize(neg_ll, np.log([2.0, 2.0]), method="Nelder-Mead")
        alpha, beta = np.exp(res.x)
        out = df_district.copy()
        out["prior_alpha"] = alpha
        out["prior_beta"] = beta
        out["posterior_alpha"] = out["stunted"] + alpha
        out["posterior_beta"] = out["n"] - out["stunted"] + beta
        out["posterior_mean"] = out["posterior_alpha"] / (out["posterior_alpha"] + out["posterior_beta"])
        out["posterior_ci_low"] = stats.beta.ppf(0.025, out["posterior_alpha"], out["posterior_beta"])
        out["posterior_ci_high"] = stats.beta.ppf(0.975, out["posterior_alpha"], out["posterior_beta"])
        out["raw_prev"] = out["stunted"] / out["n"]
        out["shrinkage"] = out["posterior_mean"] - out["raw_prev"]
        hyperparams = {"alpha": float(alpha), "beta": float(beta), "success": bool(res.success)}
        return out, hyperparams

    def save_all_bayesian(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        district_data = self.df.groupby(DISTRICT_COL)[TARGET_VAR].agg(stunted="sum", n="size").reset_index()
        bay_dist, bay_hyper = self.bayesian_beta_binomial(district_data)
        bay_dist.to_csv(output_dir / "bayesian_district_partial_pooling.csv", index=False)
        with open(output_dir / "bayesian_beta_binomial_hyperparameters.json", "w") as f:
            json.dump(bay_hyper, f, indent=2)