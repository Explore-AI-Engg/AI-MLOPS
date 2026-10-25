# Production MLOps Lab

A clean-slate production-style machine-learning system for Adult Income
classification.

## Planned architecture

Data ingestion -> validation -> EDA -> feature engineering -> model training
-> MLflow -> evaluation -> ONNX -> FastAPI -> Docker -> CI -> Artifact Registry
-> CD -> GKE -> Prometheus -> Grafana -> drift monitoring.

## Project principles

- Reproducible training
- Versioned models and containers
- CI must succeed before CD starts
- Immutable image tags
- ONNX parity validation
- Explicit health and readiness checks
- Operational and ML monitoring
- No secrets committed to Git
