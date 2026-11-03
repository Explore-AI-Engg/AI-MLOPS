#!/usr/bin/env bash
set -euo pipefail

LOCAL_PORT="${LOCAL_PORT:-8001}"

python -m compileall src scripts tests
pytest -q

./scripts/fetch_dataset.sh
python scripts/prepare_data.py
python scripts/train.py
MODEL_URI="$(<artifacts/model-uri.txt)"
python scripts/export_onnx.py --model-uri "${MODEL_URI}"
python scripts/validate_onnx.py --model-uri "${MODEL_URI}"
pytest -q

docker build \
  --tag heart-disease-api:local-ci \
  .

docker rm --force heart-disease-api-local-ci >/dev/null 2>&1 || true

docker run \
  --detach \
  --name heart-disease-api-local-ci \
  --publish "${LOCAL_PORT}:8000" \
  --env MODEL_VERSION=local-ci \
  --env SERVICE_VERSION=local-ci \
  heart-disease-api:local-ci

cleanup() {
  docker rm --force heart-disease-api-local-ci >/dev/null 2>&1 || true
}

trap cleanup EXIT

for attempt in $(seq 1 30); do
  if curl --fail --silent "http://127.0.0.1:${LOCAL_PORT}/readyz"; then
    echo
    echo "Container is ready"
    break
  fi

  if [[ "${attempt}" == "30" ]]; then
    docker logs heart-disease-api-local-ci
    exit 1
  fi

  sleep 2
done

BASE_URL="http://127.0.0.1:${LOCAL_PORT}" \
  ./scripts/container_smoke_test.sh

echo "Local CI-equivalent validation passed"