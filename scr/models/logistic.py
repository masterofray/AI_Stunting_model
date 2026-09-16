"""Logistic Regression Model Module"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, classification_report)
from sklearn.impute import SimpleImputer
from config.settings import REQUIRED_COLUMNS, BINARY_VARS, TARGET_VAR, DISTRICT_COL, MALNUTRITION_COL, model_config

class LogisticModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        for col in REQUIRED_COLUMNS:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        self.model = None
        self.is_fitted = False

    def prepare_data(self, include_district: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        if include_district:
            X = self.df.drop(columns=[TARGET_VAR])
        else:
            X = self.df.drop(columns=[TARGET_VAR, DISTRICT_COL])
        y = self.df[TARGET_VAR]
        return X, y

    def _create_preprocessor(self) -> ColumnTransformer:
        all_cols = [col for col in self.df.columns if col != TARGET_VAR]
        categorical_cols = [DISTRICT_COL, MALNUTRITION_COL]
        numeric_cols = [col for col in all_cols if col not in categorical_cols + BINARY_VARS]
        binary_cols = [col for col in BINARY_VARS if col in all_cols]
        return ColumnTransformer([
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("bin", SimpleImputer(strategy="most_frequent"), binary_cols),
        ])

    def fit(self, include_district: bool = True, **kwargs) -> None:
        X, y = self.prepare_data(include_district=include_district)
        preprocessor = self._create_preprocessor()
        lr_params = {"max_iter": model_config.LOGISTIC_MAX_ITER, "solver": model_config.LOGISTIC_SOLVER}
        lr_params.update(kwargs)
        self.model = Pipeline([("preprocessor", preprocessor),
                             ("classifier", LogisticRegression(**lr_params))])
        self.model.fit(X, y)
        self.is_fitted = True

    def cross_validate(self, n_splits: int = None, include_district: bool = True) -> Dict[str, Any]:
        if n_splits is None:
            n_splits = model_config.CV_N_SPLITS
        X, y = self.prepare_data(include_district=include_district)
        preprocessor = self._create_preprocessor()
        lr_params = {"max_iter": model_config.LOGISTIC_MAX_ITER, "solver": model_config.LOGISTIC_SOLVER}
        pipe = Pipeline([("preprocessor", preprocessor),
                       ("classifier", LogisticRegression(**lr_params))])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=model_config.CV_RANDOM_STATE)
        y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred_class = cross_val_predict(pipe, X, y, cv=cv, method="predict")
        results = {
            "n_splits": n_splits, "n_samples": len(X),
            "metrics": {
                "accuracy": float(accuracy_score(y, y_pred_class)),
                "precision": float(precision_score(y, y_pred_class)),
                "recall": float(recall_score(y, y_pred_class)),
                "f1": float(f1_score(y, y_pred_class)),
                "roc_auc": float(roc_auc_score(y, y_pred)),
                "average_precision": float(average_precision_score(y, y_pred)),
                "brier_score_loss": float(brier_score_loss(y, y_pred)),
            },
            "confusion_matrix": confusion_matrix(y, y_pred_class).tolist(),
        }
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        results["cv_roc_auc_mean"] = float(np.mean(cv_scores))
        results["cv_roc_auc_std"] = float(np.std(cv_scores))
        self.cv_results = results
        return results

    def evaluate(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        if X is None or y is None:
            X, y = self.prepare_data()
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred_class = self.model.predict(X)
        return {
            "n_samples": len(X),
            "metrics": {
                "accuracy": float(accuracy_score(y, y_pred_class)),
                "precision": float(precision_score(y, y_pred_class)),
                "recall": float(recall_score(y, y_pred_class)),
                "f1": float(f1_score(y, y_pred_class)),
                "roc_auc": float(roc_auc_score(y, y_pred_proba)),
                "average_precision": float(average_precision_score(y, y_pred_proba)),
                "brier_score_loss": float(brier_score_loss(y, y_pred_proba)),
            },
            "confusion_matrix": confusion_matrix(y, y_pred_class).tolist(),
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        return self.model.predict(X), self.model.predict_proba(X)

    def get_coefficients(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        coefs = self.model.named_steps["classifier"].coef_[0]
        intercept = self.model.named_steps["classifier"].intercept_[0]
        rows = [{"feature": "Intercept", "coefficient": float(intercept), "odds_ratio": float(np.exp(intercept))}]
        if hasattr(self.model.named_steps["preprocessor"], "named_transformers_"):
            if "cat" in self.model.named_steps["preprocessor"].named_transformers_:
                cat_transformer = self.model.named_steps["preprocessor"].named_transformers_["cat"]
                cat_cols = [DISTRICT_COL, MALNUTRITION_COL]
                if hasattr(cat_transformer.named_steps["ohe"], "get_feature_names_out"):
                    cat_features = cat_transformer.named_steps["ohe"].get_feature_names_out(cat_cols)
                    for feature, coef in zip(cat_features, coefs[:len(cat_features)]):
                        rows.append({"feature": feature, "coefficient": float(coef), "odds_ratio": float(np.exp(coef))})
            num_start = len(cat_features) if 'cat_features' in locals() else 0
            for i, col in enumerate([c for c in self.df.columns if c not in [TARGET_VAR] + BINARY_VARS]):
                idx = num_start + i
                if idx < len(coefs):
                    rows.append({"feature": col, "coefficient": float(coefs[idx]), "odds_ratio": float(np.exp(coefs[idx]))})
        return pd.DataFrame(rows)

    def save_model(self, output_path: Path) -> None:
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        import joblib
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output_path)

    def compare_with_deduplication(self) -> Dict[str, Any]:
        raw_results = self.cross_validate()
        dedup_df = self.df.drop_duplicates()
        dedup_model = LogisticModel(dedup_df)
        dedup_results = dedup_model.cross_validate()
        return {"raw": raw_results, "deduplicated": dedup_results, "n_dedup": len(dedup_df)}