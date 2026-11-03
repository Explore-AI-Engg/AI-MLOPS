from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest


ONNX_PATH = Path("artifacts/heart_disease_model.onnx")


def test_onnx_model_is_structurally_valid() -> None:
    if not ONNX_PATH.exists():
        pytest.skip("ONNX model has not been exported locally")

    model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model)

    assert len(model.graph.input) == 13
    assert len(model.graph.output) >= 1


def test_onnx_runtime_can_create_session() -> None:
    if not ONNX_PATH.exists():
        pytest.skip("ONNX model has not been exported locally")

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    assert session.get_inputs()
    assert session.get_outputs()


def test_onnx_outputs_are_finite() -> None:
    if not ONNX_PATH.exists():
        pytest.skip("ONNX model has not been exported locally")

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    feeds = {
        item.name: np.zeros(
            (2, 1),
            dtype=np.float32,
        )
        for item in session.get_inputs()
    }

    outputs = session.run(None, feeds)

    for output in outputs:
        if isinstance(output, list):
            continue

        array = np.asarray(output)

        if np.issubdtype(array.dtype, np.number):
            assert np.isfinite(array).all()