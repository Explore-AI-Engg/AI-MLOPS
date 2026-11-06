#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

curl --fail "${BASE_URL}/healthz"
curl --fail "${BASE_URL}/readyz"
curl --fail "${BASE_URL}/v1/model"

curl --fail \
  --request POST \
  --header "Content-Type: application/json" \
  --data '{
    "age": 63,
    "trestbps": 145,
    "chol": 233,
    "thalach": 150,
    "oldpeak": 2.3,
    "sex": 1,
    "cp": 1,
    "fbs": 1,
    "restecg": 2,
    "exang": 0,
    "slope": 3,
    "ca": 0,
    "thal": 6
  }' \
  "${BASE_URL}/v1/predict"

curl --fail "${BASE_URL}/metrics" | grep "heart_disease"

echo
echo "Container smoke test passed"