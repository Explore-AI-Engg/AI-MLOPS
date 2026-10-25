from __future__ import annotations

import numpy as np
import pandas as pd

from data.ingestion import RAW_COLUMNS
from features.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    prepare_labeled_dataframe,
)
from features.split import split_dataset


def make_raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
            [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
            [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0, 3, 1],
            [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, 3, 0],
            [56, 1, 2, 130, 236, 0, 0, 178, 0, 0.8, 1, 0, 3, 0],
            [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, 2, 3, 3],
            [57, 0, 4, 120, 354, 0, 0, 163, 1, 0.6, 1, 0, 3, 0],
            [63, 1, 4, 150, 223, 0, 2, 115, 0, 2.0, 2, 1, 6, 1],
            [53, 1, 4, 140, 203, 1, 2, 155, 1, 3.1, 3, 0, 7, 2],
            [57, 1, 4, 140, 192, 0, 0, 148, 0, 0.4, 2, 0, 6, 0],
        ],
        columns=RAW_COLUMNS,
    )


def test_prepare_labeled_dataframe_creates_binary_target() -> None:
    prepared = prepare_labeled_dataframe(make_raw_dataframe())

    assert set(prepared["num"].unique()) == {0, 1}
    assert prepared["num"].dtype == "int64"


def test_preprocessor_handles_missing_values() -> None:
    dataframe = make_raw_dataframe()
    dataframe.loc[0, "chol"] = np.nan
    dataframe.loc[1, "thal"] = np.nan

    prepared = prepare_labeled_dataframe(dataframe)
    split = split_dataset(prepared, test_size=0.2)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(
        split.x_train,
        split.y_train,
    )

    assert transformed.shape[0] == len(split.x_train)
    assert np.isfinite(transformed).all()


def test_split_preserves_feature_groups() -> None:
    prepared = prepare_labeled_dataframe(make_raw_dataframe())
    split = split_dataset(prepared, test_size=0.2)

    assert tuple(split.x_train.columns) == (
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    )
    assert len(split.x_train) + len(split.x_test) == len(prepared)
    assert len(split.y_train) + len(split.y_test) == len(prepared)