from __future__ import annotations

from pathlib import Path

from data.ingestion import load_raw_dataset
from features.pipeline import (
    build_preprocessor,
    prepare_labeled_dataframe,
)
from features.split import split_dataset


def main() -> int:
    source_path = Path(
        "data/raw/extracted/processed.cleveland.data"
    )

    raw = load_raw_dataset(source_path)
    prepared = prepare_labeled_dataframe(raw)
    split = split_dataset(prepared)

    preprocessor = build_preprocessor()
    transformed_train = preprocessor.fit_transform(
        split.x_train,
        split.y_train,
    )
    transformed_test = preprocessor.transform(split.x_test)

    print(f"Raw rows: {len(raw)}")
    print(f"Prepared rows: {len(prepared)}")
    print(f"Training rows: {len(split.x_train)}")
    print(f"Test rows: {len(split.x_test)}")
    print(f"Training feature shape: {transformed_train.shape}")
    print(f"Test feature shape: {transformed_test.shape}")
    print(
        "Training class distribution:",
        split.y_train.value_counts().sort_index().to_dict(),
    )
    print(
        "Test class distribution:",
        split.y_test.value_counts().sort_index().to_dict(),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())