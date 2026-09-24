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


"""
Settings and configuration constants for Mamberamo Raya stunting analysis.
"""
from pathlib import Path
from typing import List

# Required columns
REQUIRED_COLUMNS: List[str] = [
    "District_Name", "Stunting", "Gender", "Malnutrition_Level",
    "Exclusive_Milky", "Smoke_Habit", "PureWater_Access",
    "Healthy_Toilet", "Difficult_Acess",
]

BINARY_VARS: List[str] = [
    "Gender", "Exclusive_Milky", "Smoke_Habit",
    "PureWater_Access", "Healthy_Toilet", "Difficult_Acess",
]

TARGET_VAR: str = "Stunting"
DISTRICT_COL: str = "District_Name"
MALNUTRITION_COL: str = "Malnutrition_Level"

class AnalysisConfig:
    CONFIDENCE_LEVEL: float = 0.95
    Z_SCORE_95CI: float = 1.959963984540054
    CV_N_SPLITS: int = 5
    CV_RANDOM_STATE: int = 42
    ALPHA: float = 0.05
    FDR_METHOD: str = "bh"

class ModelConfig:
    LOGISTIC_MAX_ITER: int = 5000
    LOGISTIC_SOLVER: str = "lbfgs"
    EVALUATION_METRICS: List[str] = [
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "average_precision", "brier_score_loss",
    ]

class PlotConfig:
    FIG_SIZE_STANDARD: tuple = (10, 6)
    FIG_SIZE_WIDE: tuple = (12, 6)
    FIG_DPI: int = 180
    COLOR_STUNTING: str = "#d62728"
    COLOR_NON_STUNTING: str = "#2ca02c"

analysis_config = AnalysisConfig()
model_config = ModelConfig()
plot_config = PlotConfig()

__all__ = [
    "REQUIRED_COLUMNS", "BINARY_VARS", "TARGET_VAR", "DISTRICT_COL",
    "MALNUTRITION_COL", "AnalysisConfig", "ModelConfig", "PlotConfig",
    "analysis_config", "model_config", "plot_config",
]