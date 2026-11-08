#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE_REFERENCE:?IMAGE_REFERENCE must be set}"

readonly TEMPLATE="deployment/kubernetes/deployment.yaml.tpl"
readonly OUTPUT="deployment/kubernetes/.generated-deployment.yaml"

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "Missing Kubernetes Deployment template: ${TEMPLATE}" >&2
  exit 1
fi

if [[ "${IMAGE_REFERENCE}" != *"@sha256:"* ]]; then
  echo "IMAGE_REFERENCE must use an immutable digest" >&2
  exit 1
fi

sed \
  "s|image: IMAGE_REFERENCE|image: ${IMAGE_REFERENCE}|" \
  "${TEMPLATE}" > "${OUTPUT}"

grep -Fq "image: ${IMAGE_REFERENCE}" "${OUTPUT}"

echo "${OUTPUT}"