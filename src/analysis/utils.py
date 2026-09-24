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


"""Utility Functions Module"""
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from config.settings import REQUIRED_COLUMNS, BINARY_VARS, TARGET_VAR, DISTRICT_COL

class AnalysisUtils:
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df

    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
        validation = {"is_valid": True, "errors": [], "warnings": []}
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            validation["is_valid"] = False
            validation["errors"].append(f"Missing required columns: {missing}")
        for var in BINARY_VARS:
            if var in df.columns:
                unique_vals = df[var].dropna().unique()
                if not all(v in [0, 1] for v in unique_vals):
                    validation["warnings"].append(f"Variable {var} contains non-binary values")
        if len(df) == 0:
            validation["is_valid"] = False
            validation["errors"].append("DataFrame is empty")
        return validation

    @staticmethod
    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        info = {
            "n_rows": len(df), "n_columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "n_unique_by_column": {col: int(df[col].nunique()) for col in df.columns},
            "missing_by_column": {col: int(df[col].isna().sum()) for col in df.columns},
        }
        if TARGET_VAR in df.columns:
            info["target_distribution"] = {
                "n_0": int((df[TARGET_VAR] == 0).sum()),
                "n_1": int((df[TARGET_VAR] == 1).sum()),
                "prop_1": float(df[TARGET_VAR].mean()),
            }
        if DISTRICT_COL in df.columns:
            info["district_info"] = {
                "n_districts": int(df[DISTRICT_COL].nunique()),
                "districts": df[DISTRICT_COL].unique().tolist(),
            }
        return info