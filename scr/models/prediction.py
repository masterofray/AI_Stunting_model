"""Prediction Module"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any
from config.settings import TARGET_VAR

class PredictionEngine:
    def __init__(self, model: Any):
        self.model = model
        self.feature_names = None

    def predict(self, X: pd.DataFrame, include_proba: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if include_proba and hasattr(self.model, "predict_proba"):
            return self.model.predict(X), self.model.predict_proba(X)
        return self.model.predict(X), None

    def batch_predict(self, data: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
        result = data.copy()
        result["stunting_prediction"] = None
        result["stunting_probability"] = None
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]
            X_batch = batch.drop(columns=[TARGET_VAR]) if TARGET_VAR in batch.columns else batch
            class_pred, proba_pred = self.predict(X_batch, include_proba=True)
            result.iloc[i:i + batch_size, result.columns.get_loc("stunting_prediction")] = class_pred
            if proba_pred is not None:
                result.iloc[i:i + batch_size, result.columns.get_loc("stunting_probability")] = proba_pred[:, 1]
        return result

    def predict_with_uncertainty(self, X: pd.DataFrame, n_bootstraps: int = 100) -> Dict[str, Any]:
        predictions = []
        for _ in range(n_bootstraps):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X.iloc[indices]
            _, proba = self.predict(X_bootstrap, include_proba=True)
            predictions.append(proba[:, 1])
        predictions = np.array(predictions)
        return {
            "mean_predictions": np.mean(predictions, axis=0),
            "std_predictions": np.std(predictions, axis=0),
            "ci_95_low": np.percentile(predictions, 2.5, axis=0),
            "ci_95_high": np.percentile(predictions, 97.5, axis=0),
        }

    def interpret_prediction(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        class_pred, proba_pred = self.predict(X, include_proba=True)
        result = pd.DataFrame({"prediction": class_pred, "probability": proba_pred[:, 1]})
        result["risk_category"] = pd.cut(
            result["probability"], bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=["Very Low", "Low", "Medium", "High"]
        )
        result["risk_level"] = result["probability"].apply(lambda p: "High" if p >= threshold else "Low")
        return result

    def get_feature_importance(self) -> pd.DataFrame:
        if not hasattr(self.model, "coef_"):
            raise ValueError("Model does not have coef_ attribute")
        coefs = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        if hasattr(self.model, "feature_names_in_"):
            feature_names = self.model.feature_names_in_
        else:
            feature_names = [f"feature_{i}" for i in range(len(coefs))]
        importance = pd.DataFrame({
            "feature": ["Intercept"] + feature_names,
            "coefficient": [float(intercept)] + [float(c) for c in coefs],
            "abs_coefficient": [abs(float(intercept))] + [abs(float(c)) for c in coefs],
            "odds_ratio": [float(np.exp(intercept))] + [float(np.exp(c)) for c in coefs],
        }).sort_values("abs_coefficient", ascending=False)
        importance["rank"] = range(1, len(importance) + 1)
        return importance

    def save_predictions(self, X: pd.DataFrame, output_path: Path) -> None:
        class_pred, proba_pred = self.predict(X, include_proba=True)
        result = pd.DataFrame({
            "stunting_prediction": class_pred,
            "stunting_probability": proba_pred[:, 1]
        })
        result["risk_category"] = pd.cut(
            result["stunting_probability"], bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=["Very Low", "Low", "Medium", "High"]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)