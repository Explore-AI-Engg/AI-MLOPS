FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/heart_disease_model.onnx \
    MODEL_VERSION=local-container \
    SERVICE_VERSION=0.1.0

WORKDIR /app

RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser

COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts/heart_disease_model.onnx ./artifacts/heart_disease_model.onnx

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()"

CMD ["uvicorn", "heart_disease_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]