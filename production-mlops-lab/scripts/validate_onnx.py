from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import onnxruntime as ort
import pandas as pd


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


def make_validation_batch() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [63, 145, 233, 150, 2.3, 1, 1, 1, 2, 0, 3, 0, 6],
            [67, 160, 286, 108, 1.5, 1, 4, 0, 2, 1, 2, 3, 3],
            [37, 130, 250, 187, 3.5, 1, 3, 0, 0, 0, 3, 0, 3],
            [41, 130, 204, 172, 1.4, 0, 2, 0, 2, 0, 1, 0, 3],
        ],
        columns=FEATURE_COLUMNS,
    )


def extract_onnx_array(output: object) -> np.ndarray:
    if isinstance(output, list):
        if not output:
            raise ValueError("ONNX output list is empty")

        first_item = output[0]

        if isinstance(first_item, dict):
            probabilities = []
            for item in output:
                if 1 not in item:
                    raise ValueError(
                        "ONNX probability output does not contain class 1"
                    )
                probabilities.append(float(item[1]))

            return np.asarray(probabilities, dtype=np.float32)

        return np.asarray(output)

    return np.asarray(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ONNX predictions against the MLflow model"
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI, for example runs:/<run_id>/model",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/heart_disease_model.onnx"),
        help="Path to the exported ONNX model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    onnx_path = args.onnx

    if not onnx_path.is_file():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")

    mlflow.set_tracking_uri("file:./mlruns")
    native_model = mlflow.sklearn.load_model(args.model_uri)
    validation_batch = make_validation_batch()

    native_labels = native_model.predict(validation_batch)

    native_probabilities = native_model.predict_proba(
        validation_batch
    )[:, 1]

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_names = [
        input_value.name
        for input_value in session.get_inputs()
    ]

    feeds = {
        name: validation_batch[[name]].to_numpy(
            dtype=np.float32
        )
        for name in input_names
    }

    outputs = session.run(None, feeds)

    if len(outputs) < 2:
        raise ValueError(
            "Expected ONNX output to contain labels and probabilities"
        )

    onnx_labels = extract_onnx_array(outputs[0]).reshape(-1)
    onnx_probability_output = extract_onnx_array(outputs[1])

    if onnx_probability_output.ndim == 2:
        onnx_probabilities = onnx_probability_output[:, 1]
    else:
        onnx_probabilities = onnx_probability_output.reshape(-1)

    native_labels = np.asarray(native_labels).reshape(-1)
    native_probabilities = np.asarray(
        native_probabilities,
        dtype=np.float32,
    ).reshape(-1)

    if not np.array_equal(native_labels, onnx_labels):
        raise AssertionError(
            "Native and ONNX predicted labels are different: "
            f"native={native_labels}, onnx={onnx_labels}"
        )

    if not np.allclose(
        native_probabilities,
        onnx_probabilities,
        rtol=1e-4,
        atol=1e-5,
    ):
        raise AssertionError(
            "Native and ONNX probabilities are outside tolerance: "
            f"native={native_probabilities}, "
            f"onnx={onnx_probabilities}"
        )

    print("ONNX Runtime session created successfully")
    print("Native and ONNX labels match")
    print("Native and ONNX probabilities match within tolerance")
    print("Validation samples:", len(validation_batch))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())