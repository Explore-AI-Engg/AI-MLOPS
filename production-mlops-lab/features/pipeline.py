from __future__ import annotations

from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.ingestion import (
    DataContractError,
    RAW_COLUMNS,
    RAW_TARGET_VALUES,
    validate_raw_dataset,
)


TARGET_COLUMN: Final[str] = "num"

NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
)

CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
)

MODEL_FEATURES: Final[tuple[str, ...]] = (
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
)


def prepare_labeled_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the raw frame and convert the original target to binary."""
    validate_raw_dataset(dataframe)

    if tuple(dataframe.columns) != RAW_COLUMNS:
        raise DataContractError(
            "Input columns do not match the expected raw data contract"
        )

    prepared = dataframe.copy()

    for column in RAW_COLUMNS:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    observed_target_values = set(
        prepared[TARGET_COLUMN].dropna().unique()
    )

    invalid_target_values = observed_target_values - RAW_TARGET_VALUES

    if invalid_target_values:
        raise DataContractError(
            "Unexpected target values after numeric conversion: "
            f"{sorted(invalid_target_values)}"
        )

    if prepared[TARGET_COLUMN].isna().any():
        raise DataContractError(
            "Target column contains missing values"
        )

    prepared[TARGET_COLUMN] = (
        prepared[TARGET_COLUMN] > 0
    ).astype("int64")

    if set(prepared[TARGET_COLUMN].unique()) != {0, 1}:
        raise DataContractError(
            "Binary target must contain both classes"
        )

    return prepared


def build_preprocessor() -> ColumnTransformer:
    """Build a leakage-safe preprocessing transformer."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )