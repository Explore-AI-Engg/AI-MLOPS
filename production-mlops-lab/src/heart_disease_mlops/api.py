from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "artifacts/heart_disease_model.onnx",
    )
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "local-development",
)

SERVICE_VERSION = os.getenv(
    "SERVICE_VERSION",
    "0.1.0",
)

FEATURE_NAMES: tuple[str, ...] = (
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

REQUEST_COUNT = Counter(
    "heart_disease_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "heart_disease_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTION_COUNT = Counter(
    "heart_disease_predictions_total",
    "Total model predictions",
    ["prediction"],
)

PREDICTION_PROBABILITY = Histogram(
    "heart_disease_prediction_probability",
    "Distribution of positive-class prediction probabilities",
    buckets=(
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
)

INFERENCE_ERRORS = Counter(
    "heart_disease_inference_errors_total",
    "Total inference errors",
    ["error_type"],
)


class HeartDiseaseRequest(BaseModel):
    """Validated request schema for the ONNX model."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=0, le=120)
    trestbps: float = Field(ge=0, le=300)
    chol: float = Field(ge=0, le=1000)
    thalach: float = Field(ge=0, le=300)
    oldpeak: float = Field(ge=-20, le=50)

    sex: float = Field(ge=0, le=1)
    cp: float = Field(ge=1, le=4)
    fbs: float = Field(ge=0, le=1)
    restecg: float = Field(ge=0, le=2)
    exang: float = Field(ge=0, le=1)
    slope: float = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: float = Field(ge=0, le=7)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    service_version: str


class ModelRuntime:
    """Loads and executes the ONNX model."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"ONNX model does not exist: {model_path}"
            )

        self.model_path = model_path
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        input_names = tuple(
            value.name for value in self.session.get_inputs()
        )

        if set(input_names) != set(FEATURE_NAMES):
            raise RuntimeError(
                "ONNX input names do not match the API feature contract. "
                f"Expected={FEATURE_NAMES}, received={input_names}"
            )

        self.input_names = input_names

    @staticmethod
    def _extract_probability(output: Any) -> np.ndarray:
        if isinstance(output, list):
            if not output:
                raise RuntimeError(
                    "ONNX probability output is empty"
                )

            if isinstance(output[0], dict):
                try:
                    return np.asarray(
                        [float(item[1]) for item in output],
                        dtype=np.float32,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "ONNX probability map does not contain class 1"
                    ) from exc

            return np.asarray(output, dtype=np.float32)

        return np.asarray(output, dtype=np.float32)

    def predict(
        self,
        request: HeartDiseaseRequest,
    ) -> tuple[int, float]:
        values = request.model_dump()

        feeds = {
            name: np.asarray(
                [[values[name]]],
                dtype=np.float32,
            )
            for name in self.input_names
        }

        outputs = self.session.run(None, feeds)

        if len(outputs) < 2:
            raise RuntimeError(
                "ONNX model must return labels and probabilities"
            )

        labels = np.asarray(outputs[0]).reshape(-1)

        if not len(labels):
            raise RuntimeError("ONNX label output is empty")

        probability_output = self._extract_probability(outputs[1])

        if probability_output.ndim == 2:
            probability = float(probability_output[0, 1])
        else:
            probability = float(probability_output.reshape(-1)[0])

        prediction = int(labels[0])

        if prediction not in (0, 1):
            raise RuntimeError(
                f"ONNX returned invalid prediction: {prediction}"
            )

        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(
                f"ONNX returned invalid probability: {probability}"
            )

        return prediction, probability


runtime: ModelRuntime | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runtime
    if runtime is None:
        runtime = ModelRuntime(MODEL_PATH)
    yield
    runtime = None


app = FastAPI(
    title="Heart Disease Risk Inference API",
    version=SERVICE_VERSION,
    description=(
        "Educational inference API. This is not a clinical diagnostic "
        "system and must not be used for medical decisions."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(
    request: Request,
    call_next,
) -> Response:
    start = time.perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path

        REQUEST_COUNT.labels(
            method=request.method,
            path=path,
            status=str(status),
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            path=path,
        ).observe(elapsed)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail="Model runtime is not ready",
        )

    return {
        "status": "ready",
        "model_version": MODEL_VERSION,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/v1/model")
def model_metadata() -> dict[str, object]:
    return {
        "model_version": MODEL_VERSION,
        "service_version": SERVICE_VERSION,
        "runtime": "onnxruntime",
        "feature_count": len(FEATURE_NAMES),
        "features": list(FEATURE_NAMES),
    }


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
)
def predict(request: HeartDiseaseRequest) -> PredictionResponse:
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail="Model runtime is not ready",
        )

    try:
        prediction, probability = runtime.predict(request)
    except Exception as exc:
        INFERENCE_ERRORS.labels(
            error_type=type(exc).__name__,
        ).inc()

        raise HTTPException(
            status_code=500,
            detail="Inference failed",
        ) from exc

    PREDICTION_COUNT.labels(
        prediction=str(prediction),
    ).inc()

    PREDICTION_PROBABILITY.observe(probability)

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_version=MODEL_VERSION,
        service_version=SERVICE_VERSION,
    )