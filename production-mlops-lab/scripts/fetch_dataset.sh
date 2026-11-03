#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
ARCHIVE_PATH="${RAW_DIR}/uci-heart-disease.zip"
EXTRACTED_DIR="${RAW_DIR}/extracted"
DATASET_PATH="${EXTRACTED_DIR}/processed.cleveland.data"
SOURCE_URL="https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

mkdir -p "${EXTRACTED_DIR}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "Downloading UCI Heart Disease dataset..."
    curl --fail --location --retry 3 --output "${ARCHIVE_PATH}" "${SOURCE_URL}"
  fi

  echo "Extracting dataset..."
  unzip -o "${ARCHIVE_PATH}" -d "${EXTRACTED_DIR}" >/dev/null
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Expected dataset file was not found: ${DATASET_PATH}" >&2
  exit 1
fi

echo "Dataset ready: ${DATASET_PATH}"
