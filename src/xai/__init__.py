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