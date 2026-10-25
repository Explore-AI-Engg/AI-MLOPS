from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.ingestion import (
    DataContractError,
    RAW_COLUMNS,
    load_raw_dataset,
    prepare_binary_dataset,
    validate_raw_dataset,
)


def make_valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
            [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
            [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0, 3, 1],
        ],
        columns=RAW_COLUMNS,
    )


def test_valid_dataset_report() -> None:
    dataframe = make_valid_dataframe()

    report = validate_raw_dataset(dataframe)

    assert report["valid"] is True
    assert report["row_count"] == 3
    assert report["column_count"] == 14
    assert report["missing_cell_count"] == 0
    assert report["observed_target_values"] == [0, 1, 2]


def test_binary_target_transformation() -> None:
    dataframe = make_valid_dataframe()

    prepared = prepare_binary_dataset(dataframe)

    assert prepared["num"].tolist() == [0, 1, 1]
    assert set(prepared["num"].unique()) == {0, 1}


def test_missing_values_are_rejected_before_training() -> None:
    dataframe = make_valid_dataframe()
    dataframe.loc[0, "ca"] = float("nan")

    with pytest.raises(DataContractError, match="Missing values"):
        prepare_binary_dataset(dataframe)


def test_invalid_target_is_rejected() -> None:
    dataframe = make_valid_dataframe()
    dataframe.loc[0, "num"] = 9

    with pytest.raises(DataContractError, match="Unexpected target"):
        validate_raw_dataset(dataframe)


def test_raw_file_has_expected_schema() -> None:
    path = Path("data/raw/extracted/processed.cleveland.data")

    if not path.exists():
        pytest.skip("Raw dataset is not present locally")

    dataframe = load_raw_dataset(path)

    assert tuple(dataframe.columns) == RAW_COLUMNS
    assert len(dataframe.columns) == 14
    assert len(dataframe) == 303