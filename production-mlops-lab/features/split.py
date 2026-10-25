from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from features.pipeline import (
    MODEL_FEATURES,
    TARGET_COLUMN,
)


@dataclass(frozen=True)
class DatasetSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_dataset(
    dataframe: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Split raw prepared data before fitting transformations."""
    x = dataframe.loc[:, list(MODEL_FEATURES)]
    y = dataframe[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )