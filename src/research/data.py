#!/usr/bin/env python3
from __future__ import annotations

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, masterofray/AI_Stunting_model"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.2.0"
__maintainer__ = "Aryanto, M.Si"
__email__      = "aryanto.dandan@gmail.com"
__created__    = "2026-08-31"
__modified__   = "2026-09-18"



"""Data loading, validation, splitting, and serialization utilities."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, MODEL


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def load_raw_data(
    source: str | Path,
    table_name: str = "RawData",
    source_type: Optional[str] = None,
) -> pd.DataFrame:
    """Load RawData from CSV or an SQL script containing CREATE/INSERT statements."""
    path = Path(source)
    source_type = (source_type or path.suffix.lstrip(".")).lower()

    if source_type in {"csv", "txt"}:
        df = pd.read_csv(path)
    elif source_type in {"sql", "sqlite"}:
        sql = path.read_text(encoding="utf-8")
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(sql)
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    else:
        raise ValueError(f"Unsupported source type: {source_type!r} for {path}")

    return _normalise_columns(df)


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a deterministic data-quality report without mutating the input."""
    missing = [col for col in FEATURES.required_columns if col not in df.columns]
    report: Dict[str, Any] = {
        "valid": not missing and not df.empty,
        "errors": [],
        "warnings": [],
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_by_column": {str(c): int(v) for c, v in df.isna().sum().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }
    if missing:
        report["valid"] = False
        report["errors"].append(f"Missing required columns: {missing}")
    if df.empty:
        report["valid"] = False
        report["errors"].append("Dataset is empty")

    if FEATURES.target in df.columns:
        values = sorted(df[FEATURES.target].dropna().unique().tolist())
        report["target_values"] = [int(v) if isinstance(v, (np.integer, int)) else v for v in values]
        if not set(values).issubset({0, 1}):
            report["valid"] = False
            report["errors"].append("Stunting must contain only binary 0/1 values")
        else:
            report["target_distribution"] = {
                "n_0": int((df[FEATURES.target] == 0).sum()),
                "n_1": int((df[FEATURES.target] == 1).sum()),
                "prevalence": float(df[FEATURES.target].mean()),
            }

    for col in (
        "Gender",
        "Exclusive_Milky",
        "Smoke_Habit",
        "PureWater_Access",
        "Healthy_Toilet",
        "Difficult_Acess",
    ):
        if col in df.columns:
            vals = set(df[col].dropna().unique().tolist())
            if not vals.issubset({0, 1}):
                report["warnings"].append(f"{col} contains non-binary values: {sorted(vals)}")

    if FEATURES.district in df.columns:
        report["n_districts"] = int(df[FEATURES.district].nunique(dropna=True))

    return report


def prepare_model_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare raw features for modeling; retain district as a categorical feature."""
    clean = df.copy()
    if FEATURES.target not in clean.columns:
        raise ValueError(f"Target column {FEATURES.target!r} not found")

    clean = clean.dropna(subset=[FEATURES.target]).copy()
    y = clean[FEATURES.target].astype(int)
    X = clean.drop(columns=[FEATURES.target, *FEATURES.excluded_features], errors="ignore")

    for col in FEATURES.categorical_features:
        if col in X.columns:
            X[col] = X[col].astype("string").fillna("<MISSING>")

    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = MODEL.test_size,
    seed: int = MODEL.seed,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    def _default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot JSON serialize {type(value)!r}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
