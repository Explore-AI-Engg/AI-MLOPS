#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ID="production-mlops-06102026"
export REGION="us-central1"
export GITHUB_OWNER="Explore-AI-Engg"
export GITHUB_REPOSITORY="AI-MLOPS"

export WIF_POOL_ID="github-actions-pool"
export WIF_PROVIDER_ID="github-actions-provider"
export CI_SERVICE_ACCOUNT_ID="mlops-ci-publisher"
export CI_SERVICE_ACCOUNT_EMAIL="${CI_SERVICE_ACCOUNT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"

printf '%s\n' \
  "$PROJECT_ID" \
  "$REGION" \
  "$GITHUB_OWNER/$GITHUB_REPOSITORY" \
  "$WIF_POOL_ID" \
  "$WIF_PROVIDER_ID" \
  "$CI_SERVICE_ACCOUNT_EMAIL"