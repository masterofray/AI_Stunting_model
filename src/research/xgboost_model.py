"""XGBoost baseline, Optuna tuning, evaluation, and model persistence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from .config import FEATURES, MODEL
from .data import save_json
from .visualization import plot_confusion_matrix, plot_feature_importance, plot_roc_pr


class XGBoostResearchModel:
    def __init__(self, seed: int = MODEL.seed, n_trials: int = MODEL.n_trials):
        self.seed = seed
        self.n_trials = n_trials
        self.pipeline: Pipeline | None = None
        self.study: optuna.Study | None = None
        self.feature_names_: list[str] = []
        self.transformed_feature_names_: list[str] = []

    def _preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        categorical = [c for c in FEATURES.categorical_features if c in X.columns]
        numeric = [c for c in X.columns if c not in categorical]
        return ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical,
                ),
                ("numeric", SimpleImputer(strategy="median"), numeric),
            ],
            remainder="drop",
        )

    def _classifier(self, params: Dict[str, Any]) -> XGBClassifier:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.seed,
            n_jobs=1,
            tree_method="hist",
            **params,
        )

    def _build_pipeline(self, X: pd.DataFrame, params: Dict[str, Any]) -> Pipeline:
        return Pipeline(
            [
                ("preprocessor", self._preprocessor(X)),
                ("classifier", self._classifier(params)),
            ]
        )

    def fit_baseline(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        self.pipeline = self._build_pipeline(X_train, MODEL.baseline_params)
        self.pipeline.fit(X_train, y_train)
        self._capture_feature_names()
        return self.pipeline

    def _cv_objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 1200, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        }
        model = self._build_pipeline(X, params)
        cv = StratifiedKFold(n_splits=MODEL.cv_splits, shuffle=True, random_state=self.seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring=MODEL.objective_metric, n_jobs=1)
        score = float(np.mean(scores))
        trial.set_user_attr("cv_std", float(np.std(scores)))
        return score

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)
        self.study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        self.study.optimize(
            lambda trial: self._cv_objective(trial, X_train, y_train),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )
        return self.study

    def fit_best(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        if self.study is None:
            raise RuntimeError("Run tune() before fit_best().")
        self.pipeline = self._build_pipeline(X_train, self.study.best_params)
        self.pipeline.fit(X_train, y_train)
        self._capture_feature_names()
        return self.pipeline

    def _capture_feature_names(self) -> None:
        if self.pipeline is None:
            return
        pre = self.pipeline.named_steps["preprocessor"]
        try:
            names = pre.get_feature_names_out()
            self.transformed_feature_names_ = [str(x) for x in names]
        except Exception:
            self.transformed_feature_names_ = []

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted")
        pred = self.pipeline.predict(X).astype(int)
        proba = self.pipeline.predict_proba(X)[:, 1]
        return pred, proba

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "brier_score_loss": float(brier_score_loss(y_true, y_prob)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true,
                y_pred,
                target_names=["Not stunting", "Stunting"],
                output_dict=True,
                zero_division=0,
            ),
        }

    def feature_importance(self) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted")
        classifier = self.pipeline.named_steps["classifier"]
        booster = classifier.get_booster()
        names = self.transformed_feature_names_ or list(booster.feature_names or [])
        gain = booster.get_score(importance_type="gain")
        weight = booster.get_score(importance_type="weight")
        cover = booster.get_score(importance_type="cover")
        rows = []
        for idx, feature in enumerate(names):
            key = f"f{idx}"
            rows.append(
                {
                    "feature": feature,
                    "gain": float(gain.get(key, 0.0)),
                    "weight": float(weight.get(key, 0.0)),
                    "cover": float(cover.get(key, 0.0)),
                }
            )
        out = pd.DataFrame(rows)
        if not out.empty and out["gain"].sum() > 0:
            out["gain_normalized"] = out["gain"] / out["gain"].sum()
        else:
            out["gain_normalized"] = 0.0
        return out.sort_values("gain", ascending=False).reset_index(drop=True)

    def save(self, model_path: str | Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted")
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, model_path)

    def save_report(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        graphics_dir: str | Path,
        json_dir: str | Path,
        data_dir: str | Path,
        prefix: str = "xgboost",
    ) -> Dict[str, Any]:
        graphics_dir = Path(graphics_dir)
        json_dir = Path(json_dir)
        data_dir = Path(data_dir)
        for directory in (graphics_dir, json_dir, data_dir):
            directory.mkdir(parents=True, exist_ok=True)
        metrics = self.evaluate(y_test, y_pred, y_prob)
        importance = self.feature_importance()
        importance.to_csv(data_dir / f"{prefix}_feature_importance.csv", index=False)
        save_json(metrics, json_dir / f"{prefix}_metrics.json")
        plot_confusion_matrix(y_test, y_pred, graphics_dir / f"{prefix}_confusion_matrix.png")
        plot_roc_pr(
            y_test,
            y_prob,
            graphics_dir / f"{prefix}_roc_curve.png",
            graphics_dir / f"{prefix}_precision_recall_curve.png",
        )
        plot_feature_importance(importance, graphics_dir / f"{prefix}_feature_importance.png")
        return metrics

    def save_tuning_report(self, output_path: str | Path) -> None:
        if self.study is None:
            raise RuntimeError("No Optuna study available")
        payload = {
            "best_value": float(self.study.best_value),
            "best_params": dict(self.study.best_params),
            "n_trials": len(self.study.trials),
            "objective_metric": MODEL.objective_metric,
            "seed": self.seed,
            "cv_splits": MODEL.cv_splits,
        }
        save_json(payload, output_path)
