from __future__ import annotations


class ModelQualityError(RuntimeError):
    """Raised when a model does not meet the configured quality gate."""


def enforce_quality_gate(
    metrics: dict[str, float],
    *,
    minimum_roc_auc: float,
    minimum_recall: float,
) -> None:
    """Fail explicitly when required evaluation metrics are insufficient."""
    roc_auc = metrics["roc_auc"]
    recall = metrics["recall"]

    failures: list[str] = []

    if roc_auc < minimum_roc_auc:
        failures.append(
            f"roc_auc={roc_auc:.4f} < minimum={minimum_roc_auc:.4f}"
        )

    if recall < minimum_recall:
        failures.append(
            f"recall={recall:.4f} < minimum={minimum_recall:.4f}"
        )

    if failures:
        raise ModelQualityError(
            "Model quality gate failed: " + "; ".join(failures)
        )