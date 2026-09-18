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


"""Model Evaluation Module"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, classification_report, roc_curve,
                             precision_recall_curve, auc)
from config.settings import TARGET_VAR, model_config

class ModelEvaluator:
    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.metrics = None

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        self.y_true = y_true
        self.y_pred = y_pred
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }
        if y_pred_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
            metrics["average_precision"] = float(average_precision_score(y_true, y_pred_proba))
            metrics["brier_score_loss"] = float(brier_score_loss(y_true, y_pred_proba))
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            metrics["roc_curve"] = {"fpr": [float(f) for f in fpr], "tpr": [float(t) for t in tpr], "auc": float(auc(fpr, tpr))}
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics["precision_recall_curve"] = {"precision": [float(p) for p in precision], "recall": [float(r) for r in recall]}
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        metrics["n_samples"] = len(y_true)
        metrics["n_positive"] = int(np.sum(y_true))
        metrics["prevalence"] = float(np.mean(y_true))
        self.metrics = metrics
        return metrics

    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, n_splits: int = None) -> Dict[str, Any]:
        if n_splits is None:
            n_splits = model_config.CV_N_SPLITS
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=model_config.CV_RANDOM_STATE)
        y_pred = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred_class = cross_val_predict(model, X, y, cv=cv, method="predict")
        results = {"n_splits": n_splits, "n_samples": len(X), "metrics": self.calculate_metrics(y, y_pred_class, y_pred)}
        cv_scores_roc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        results["cv_roc_auc_mean"] = float(np.mean(cv_scores_roc))
        results["cv_roc_auc_std"] = float(np.std(cv_scores_roc))
        return results

    def generate_evaluation_report(self, model, X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
                                    model_name: str = "model") -> Dict[str, Any]:
        report = {"model_name": model_name, "training": {}, "test": {}}
        if hasattr(model, "predict"):
            y_train_pred = model.predict(X_train)
            y_train_pred_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None
            report["training"] = self.calculate_metrics(y_train, y_train_pred, y_train_pred_proba)
        if X_test is not None and y_test is not None:
            if hasattr(model, "predict"):
                y_test_pred = model.predict(X_test)
                y_test_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                report["test"] = self.calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
        return report

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       model_name: str = "Model", output_path: Optional[Path] = None):
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc="lower right")
        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig

    def save_evaluation_plots(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                            model_name: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_roc_curve(y_true, y_pred_proba, model_name,
                          output_path=output_dir / f"{model_name}_roc_curve.png")