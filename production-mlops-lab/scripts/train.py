from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn

from heart_disease_mlops.data.ingestion import load_raw_dataset
from heart_disease_mlops.features.pipeline import prepare_labeled_dataframe
from heart_disease_mlops.features.split import split_dataset
from heart_disease_mlops.models.evaluation import (
    evaluate_binary_classifier,
)
from heart_disease_mlops.models.quality import enforce_quality_gate
from heart_disease_mlops.models.training import (
    MODEL_NAME,
    build_model,
    configure_mlflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the heart disease classifier"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/raw/extracted/processed.cleveland.data"
        ),
    )
    parser.add_argument(
        "--experiment",
        default="heart-disease-classification",
    )
    parser.add_argument(
        "--minimum-roc-auc",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/evaluation.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    configure_mlflow(args.experiment)

    raw_dataframe = load_raw_dataset(args.input)
    prepared_dataframe = prepare_labeled_dataframe(raw_dataframe)
    dataset_split = split_dataset(prepared_dataframe)

    model = build_model()

    with mlflow.start_run(run_name=MODEL_NAME) as run:
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("feature_count", 13)
        mlflow.log_param("minimum_roc_auc", args.minimum_roc_auc)
        mlflow.log_param("minimum_recall", args.minimum_recall)

        model.fit(
            dataset_split.x_train,
            dataset_split.y_train,
        )

        metrics = evaluate_binary_classifier(
            model,
            dataset_split.x_test,
            dataset_split.y_test,
        )

        mlflow.log_metrics(metrics)

        metrics_payload = {
            "run_id": run.info.run_id,
            "model_name": MODEL_NAME,
            "metrics": metrics,
            "quality_gate": {
                "minimum_roc_auc": args.minimum_roc_auc,
                "minimum_recall": args.minimum_recall,
            },
        }

        args.metrics_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.metrics_output.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics_payload,
                file,
                indent=2,
                sort_keys=True,
            )
            file.write("\n")

        enforce_quality_gate(
            metrics,
            minimum_roc_auc=args.minimum_roc_auc,
            minimum_recall=args.minimum_recall,
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        print(f"MLflow run ID: {run.info.run_id}")
        print(f"MLflow model URI: runs:/{run.info.run_id}/model")
        print(f"Model: {MODEL_NAME}")
        print("Evaluation metrics:")

        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        print(f"Metrics report: {args.metrics_output}")
        print("Quality gate: PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())