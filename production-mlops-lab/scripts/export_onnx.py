# python scripts/export_onnx.py \
#   --model-uri runs:/abc123.../model

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import mlflow
import mlflow.sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


FEATURE_COLUMNS = (
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the trained sklearn pipeline to ONNX"
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI, for example runs:/<run_id>/model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/heart_disease_model.onnx"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    mlflow.set_tracking_uri("file:./mlruns")
    model = mlflow.sklearn.load_model(args.model_uri)

    initial_types = [
        (
            column,
            FloatTensorType([None, 1]),
        )
        for column in FEATURE_COLUMNS
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=17,
    )

    onnx.checker.check_model(onnx_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(onnx_model.SerializeToString())

    print(f"ONNX model written to: {args.output}")
    print(f"ONNX model size: {args.output.stat().st_size} bytes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())