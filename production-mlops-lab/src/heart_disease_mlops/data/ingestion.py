from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd


RAW_COLUMNS: Final[tuple[str, ...]] = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
)

RAW_TARGET_VALUES: Final[frozenset[int]] = frozenset({0, 1, 2, 3, 4})
MISSING_VALUE_MARKER: Final[str] = "?"


class DataContractError(ValueError):
    """Raised when the source dataset violates the expected contract."""


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the canonical Cleveland file without silently changing its schema."""
    source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {source_path}")

    try:
        dataframe = pd.read_csv(
            source_path,
            header=None,
            names=list(RAW_COLUMNS),
            sep=",",
            na_values=[MISSING_VALUE_MARKER],
            keep_default_na=False,
            skipinitialspace=True,
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise DataContractError(
            f"Unable to parse dataset file: {source_path}"
        ) from exc

    return dataframe


def validate_raw_dataset(dataframe: pd.DataFrame) -> dict[str, object]:
    """Validate structure, types, target values, and missing-value information."""
    actual_columns = tuple(str(column) for column in dataframe.columns)

    if actual_columns != RAW_COLUMNS:
        raise DataContractError(
            "Unexpected columns. "
            f"Expected {RAW_COLUMNS}, received {actual_columns}"
        )

    if dataframe.empty:
        raise DataContractError("Dataset is empty")

    numeric_columns = list(RAW_COLUMNS)

    converted = dataframe[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid_numeric_cells = int(converted.isna().sum().sum())
    missing_by_column = {
        str(column): int(count)
        for column, count in dataframe.isna().sum().items()
        if int(count) > 0
    }

    non_missing_target = converted["num"].dropna()
    observed_target_values = {
        int(value) for value in non_missing_target.unique()
    }
    invalid_target_values = observed_target_values - RAW_TARGET_VALUES

    if invalid_target_values:
        raise DataContractError(
            "Unexpected target values: "
            f"{sorted(invalid_target_values)}. "
            f"Expected values are {sorted(RAW_TARGET_VALUES)}."
        )

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "missing_by_column": missing_by_column,
        "missing_cell_count": int(dataframe.isna().sum().sum()),
        "invalid_numeric_cell_count": invalid_numeric_cells,
        "observed_target_values": sorted(observed_target_values),
        "valid": True,
    }


def prepare_binary_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a model-ready dataset after raw validation."""
    report = validate_raw_dataset(dataframe)

    if not bool(report["valid"]):
        raise DataContractError("Raw dataset validation failed")

    if int(report["missing_cell_count"]) > 0:
        missing_by_column = report["missing_by_column"]
        raise DataContractError(
            "Missing values must be handled explicitly before training: "
            f"{missing_by_column}"
        )

    prepared = dataframe.copy()

    for column in RAW_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")

    prepared["num"] = (prepared["num"] > 0).astype("int64")

    if set(prepared["num"].unique()) != {0, 1}:
        raise DataContractError(
            "Binary target transformation did not produce both classes"
        )

    return prepared


def ingest_and_prepare(
    source_path: str | Path,
    processed_path: str | Path,
) -> dict[str, object]:
    """Load, validate, prepare, and persist the processed dataset."""
    raw_dataframe = load_raw_dataset(source_path)
    validation_report = validate_raw_dataset(raw_dataframe)
    # Impute missing values with the most frequent value (mode)
 
    imputed_dataframe = raw_dataframe.copy()
 
    imputed_dataframe['ca'] = imputed_dataframe['ca'].fillna(imputed_dataframe['ca'].mode()[0])
 
    imputed_dataframe['thal'] = imputed_dataframe['thal'].fillna(imputed_dataframe['thal'].mode()[0])

    processed_dataframe = prepare_binary_dataset(imputed_dataframe)

    destination = Path(processed_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    processed_dataframe.to_csv(destination, index=False)

    return {
        "validation": validation_report,
        "processed_path": str(destination),
        "processed_row_count": int(len(processed_dataframe)),
        "processed_column_count": int(len(processed_dataframe.columns)),
        "positive_class_count": int(processed_dataframe["num"].sum()),
        "negative_class_count": int(
            (processed_dataframe["num"] == 0).sum()
        ),
    }