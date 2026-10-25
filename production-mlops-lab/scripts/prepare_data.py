from __future__ import annotations

import argparse
from pathlib import Path

from data.ingestion import ingest_and_prepare
from data.validation import write_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and prepare the Cleveland heart disease dataset"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/extracted/processed.cleveland.data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/heart_disease.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/data-validation.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    result = ingest_and_prepare(
        source_path=args.input,
        processed_path=args.output,
    )

    write_validation_report(
        report=result,
        output_path=args.report,
    )

    print(f"Processed dataset: {result['processed_path']}")
    print(f"Rows: {result['processed_row_count']}")
    print(f"Columns: {result['processed_column_count']}")
    print(f"Negative class: {result['negative_class_count']}")
    print(f"Positive class: {result['positive_class_count']}")
    print(f"Validation report: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())