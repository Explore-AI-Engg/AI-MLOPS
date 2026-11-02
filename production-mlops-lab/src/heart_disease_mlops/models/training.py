from __future__ import annotations

from typing import Final

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from heart_disease_mlops.features.pipeline import build_preprocessor


MODEL_NAME: Final[str] = "logistic-regression-baseline"


def build_model() -> Pipeline:
    """Build the complete preprocessing and model pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def configure_mlflow(experiment_name: str) -> None:
    """Configure a local MLflow tracking backend."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)