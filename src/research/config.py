"""Configuration for the notebook-to-module XGBoost research pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class FeatureConfig:
    target: str = "Stunting"
    district: str = "District_Name"
    categorical_features: Tuple[str, ...] = ("District_Name",)
    excluded_features: Tuple[str, ...] = ()

    @property
    def required_columns(self) -> Tuple[str, ...]:
        return (
            "District_Name",
            "Stunting",
            "Gender",
            "Malnutrition_Level",
            "Exclusive_Milky",
            "Smoke_Habit",
            "PureWater_Access",
            "Healthy_Toilet",
            "Difficult_Acess",
        )


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    test_size: float = 0.20
    cv_splits: int = 5
    n_trials: int = 40
    objective_metric: str = "roc_auc"
    early_stopping_rounds: int = 50
    baseline_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "min_child_weight": 1,
        }
    )


@dataclass(frozen=True)
class OutputConfig:
    root: Path = Path("artefact")

    @property
    def graphics(self) -> Path:
        return self.root / "graphics"

    @property
    def json(self) -> Path:
        return self.root / "json"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def xai(self) -> Path:
        return self.root / "xai"

    @property
    def model(self) -> Path:
        return self.root / "model"

    @property
    def zip_path(self) -> Path:
        return self.root / "stunting_research_artifacts.zip"

    def create(self) -> None:
        for path in (
            self.root,
            self.graphics,
            self.json,
            self.data,
            self.xai,
            self.model,
        ):
            path.mkdir(parents=True, exist_ok=True)


FEATURES = FeatureConfig()
MODEL = ModelConfig()
OUTPUT = OutputConfig()
