"""File-based visualizations for CI-friendly research artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from .config import FEATURES


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_district_prevalence(df: pd.DataFrame, path: str | Path) -> None:
    summary = (
        df.groupby(FEATURES.district)[FEATURES.target]
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    summary.plot(kind="barh", ax=ax)
    ax.set_title("Stunting prevalence by district")
    ax.set_xlabel("Prevalence")
    ax.set_ylabel("District")
    ax.set_xlim(0, 1)
    _save(fig, path)


def plot_confusion_matrix(y_true, y_pred, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Not stunting", "Stunting"], cmap="Blues", ax=ax
    )
    ax.set_title("XGBoost confusion matrix")
    _save(fig, path)


def plot_roc_pr(y_true, y_prob, roc_path: str | Path, pr_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("XGBoost ROC curve")
    _save(fig, roc_path)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("XGBoost precision-recall curve")
    _save(fig, pr_path)


def plot_feature_importance(importance: pd.DataFrame, path: str | Path, top_n: int = 20) -> None:
    data = importance.sort_values("gain", ascending=False).head(top_n).sort_values("gain")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(data["feature"], data["gain"])
    ax.set_title("XGBoost feature importance by gain")
    ax.set_xlabel("Gain")
    _save(fig, path)


def plot_binary_summary(df: pd.DataFrame, path: str | Path) -> None:
    columns = [
        c for c in [
            "Gender",
            "Exclusive_Milky",
            "Smoke_Habit",
            "PureWater_Access",
            "Healthy_Toilet",
            "Difficult_Acess",
        ]
        if c in df.columns
    ]
    means = df.groupby(FEATURES.target)[columns].mean().T
    fig, ax = plt.subplots(figsize=(10, 6))
    means.plot(kind="bar", ax=ax)
    ax.set_title("Binary covariates by stunting class")
    ax.set_ylabel("Proportion equal to 1")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path)
