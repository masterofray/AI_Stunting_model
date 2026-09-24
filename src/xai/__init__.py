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



"""Explainable AI utilities for the stunting XGBoost model.

Modules
-------
explain           : vanilla SHAP / LIME / native XGBoost importance.
counterfactual    : diverse, feasibility-constrained counterfactuals (GA).
rules             : IF-THEN rule distillation from the XGBoost ensemble.
interactions      : SHAP interaction values + Friedman H-statistic.
stability         : bootstrap stability of SHAP explanations.
subgroup          : subgroup-aware SHAP divergence + ICE curves.
"""

from .counterfactual import CounterfactualExplainer, FeatureSpec
from .explain import XAIExplainer
from .interactions import InteractionExplainer
from .rules import RuleExtractor
from .stability import StabilityExplainer
from .subgroup import SubgroupExplainer

__all__ = [
    "XAIExplainer",
    "CounterfactualExplainer",
    "FeatureSpec",
    "RuleExtractor",
    "InteractionExplainer",
    "StabilityExplainer",
    "SubgroupExplainer",
]