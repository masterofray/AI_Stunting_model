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



"""Bayesian Models Module"""
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize
from config.settings import REQUIRED_COLUMNS, TARGET_VAR, DISTRICT_COL, analysis_config

class BayesianModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        for col in REQUIRED_COLUMNS:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        self.model = None
        self.trace = None
        self.is_fitted = False

    def fit_beta_binomial_hierarchical(self, group_col: str = DISTRICT_COL) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        grouped = self.df.groupby(group_col)[TARGET_VAR].agg(stunted="sum", n="size").reset_index()
        y = grouped["stunted"].to_numpy(dtype=float)
        n = grouped["n"].to_numpy(dtype=float)

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
        out = grouped.copy()
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
        self.is_fitted = True
        return out, hyperparams

    def fit_mixed_logit_variational(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            formula = "Stunting ~ Gender + Exclusive_Milky + Smoke_Habit + PureWater_Access + Healthy_Toilet + Difficult_Acess"
            model = BinomialBayesMixedGLM.from_formula(
                formula, {"district": "0 + C(District_Name)"}, self.df
            )
            fit = model.fit_vb()
            rows = []
            for feature, mu, sd in zip(model.exog_names, fit.fe_mean, fit.fe_sd):
                rows.append({
                    "feature": feature,
                    "posterior_mean_beta": float(mu),
                    "posterior_sd_beta": float(sd),
                    "posterior_or": float(np.exp(mu)),
                    "or_ci_low": float(np.exp(mu - analysis_config.Z_SCORE_95CI * sd)),
                    "or_ci_high": float(np.exp(mu + analysis_config.Z_SCORE_95CI * sd)),
                    "posterior_prob_beta_gt_0": float(stats.norm.cdf(mu / sd)),
                })
            re_sd = float(np.exp(fit.vcp_mean[0]))
            re_var = re_sd**2
            logistic_icc = re_var / (re_var + math.pi**2 / 3)
            meta = {
                "random_effect_sd": re_sd,
                "random_effect_variance": re_var,
                "logistic_icc": float(logistic_icc)
            }
            self.model = model
            self.is_fitted = True
            return pd.DataFrame(rows), meta
        except ImportError:
            return self._fallback_logistic()

    def _fallback_logistic(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        X = self.df.drop(columns=[TARGET_VAR, DISTRICT_COL])
        y = self.df[TARGET_VAR]
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Malnutrition_Level"]),
            ("num", "passthrough", [c for c in X.columns if c != "Malnutrition_Level"])
        ])
        pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=5000))])
        pipe.fit(X, y)
        feature_names = pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(["Malnutrition_Level"]).tolist()
        feature_names += [c for c in X.columns if c != "Malnutrition_Level"]
        rows = []
        for feature, coef in zip(feature_names, pipe.named_steps["lr"].coef_[0]):
            rows.append({"feature": feature, "coefficient": float(coef), "odds_ratio": float(np.exp(coef))})
        meta = {"fit_method": "logistic_regression_fallback", "n_observations": len(self.df)}
        return pd.DataFrame(rows), meta