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


"""Data Quality Analysis Module"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any
from config.settings import REQUIRED_COLUMNS, BINARY_VARS, TARGET_VAR, DISTRICT_COL

class DataQualityAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate_required_columns()
        self.quality_report: Dict[str, Any] = {}

    def _validate_required_columns(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def check_exact_duplicates(self) -> Tuple[Dict[str, Any], pd.Series]:
        fp = pd.util.hash_pandas_object(self.df, index=False)
        counts = fp.value_counts()
        summary = {
            "total_rows": int(len(self.df)),
            "unique_exact_records": int(len(counts)),
            "exact_duplicate_rows_beyond_first": int(self.df.duplicated().sum()),
            "duplicate_groups": int((counts > 1).sum()),
            "rows_in_duplicate_groups": int(fp.isin(counts[counts > 1].index).sum()),
            "max_repeat_count": int(counts.max()),
            "mean_repeat_among_duplicate_groups": float(counts[counts > 1].mean()) if (counts > 1).any() else 1.0,
        }
        self.quality_report["exact_duplicates"] = summary
        return summary, counts

    def check_missing_values(self) -> Dict[str, Any]:
        missing_total = int(self.df.isna().sum().sum())
        missing_by_column = {k: int(v) for k, v in self.df.isna().sum().items()}
        result = {
            "missing_values_total": missing_total,
            "missing_by_column": missing_by_column,
            "has_missing_values": missing_total > 0,
        }
        self.quality_report["missing_values"] = result
        return result

    def check_binary_variables(self) -> Dict[str, Any]:
        binary_validation = {}
        for var in BINARY_VARS:
            if var in self.df.columns:
                unique_values = self.df[var].dropna().unique().tolist()
                is_valid = all(v in [0, 1] for v in unique_values)
                n = len(self.df[var])
                n_ones = int((self.df[var] == 1).sum())
                binary_validation[var] = {
                    "unique_values": unique_values,
                    "is_valid_binary": is_valid,
                    "n_zeros": n - n_ones,
                    "n_ones": n_ones,
                    "prop_ones": n_ones / n if n > 0 else 0.0,
                }
        self.quality_report["binary_variables"] = binary_validation
        return binary_validation

    def check_target_distribution(self) -> Dict[str, Any]:
        if TARGET_VAR not in self.df.columns:
            return {}
        target_counts = self.df[TARGET_VAR].value_counts()
        n_stunting = int(target_counts.get(1, 0))
        n_non_stunting = int(target_counts.get(0, 0))
        n_total = n_stunting + n_non_stunting
        result = {
            "n_stunting": n_stunting,
            "n_non_stunting": n_non_stunting,
            "n_total": n_total,
            "stunting_prevalence": n_stunting / n_total if n_total > 0 else 0.0,
        }
        self.quality_report["target_distribution"] = result
        return result

    def check_district_distribution(self) -> Dict[str, Any]:
        if DISTRICT_COL not in self.df.columns:
            return {}
        district_counts = self.df[DISTRICT_COL].value_counts()
        result = {
            "n_districts": int(district_counts.shape[0]),
            "district_counts": {k: int(v) for k, v in district_counts.items()},
            "min_samples_per_district": int(district_counts.min()),
            "max_samples_per_district": int(district_counts.max()),
            "mean_samples_per_district": float(district_counts.mean()),
        }
        self.quality_report["district_distribution"] = result
        return result

    def generate_full_report(self) -> Dict[str, Any]:
        self.quality_report["dataset_info"] = {
            "n_rows": len(self.df),
            "n_columns": len(self.df.columns),
            "column_names": list(self.df.columns),
        }
        self.check_exact_duplicates()
        self.check_missing_values()
        self.check_binary_variables()
        self.check_target_distribution()
        self.check_district_distribution()
        return self.quality_report

    def save_report(self, output_path: Path) -> None:
        report = self.generate_full_report()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def get_deduplicated_data(self) -> pd.DataFrame:
        return self.df.drop_duplicates().copy()

class DataLoader:
    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df