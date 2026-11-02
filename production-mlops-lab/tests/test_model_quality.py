from __future__ import annotations

import pytest

from heart_disease_mlops.models.quality import (
    ModelQualityError,
    enforce_quality_gate,
)


def test_quality_gate_passes() -> None:
    metrics = {
        "roc_auc": 0.80,
        "recall": 0.75,
    }

    enforce_quality_gate(
        metrics,
        minimum_roc_auc=0.70,
        minimum_recall=0.60,
    )


def test_quality_gate_fails_for_low_roc_auc() -> None:
    metrics = {
        "roc_auc": 0.60,
        "recall": 0.75,
    }

    with pytest.raises(ModelQualityError, match="roc_auc"):
        enforce_quality_gate(
            metrics,
            minimum_roc_auc=0.70,
            minimum_recall=0.60,
        )


def test_quality_gate_fails_for_low_recall() -> None:
    metrics = {
        "roc_auc": 0.80,
        "recall": 0.50,
    }

    with pytest.raises(ModelQualityError, match="recall"):
        enforce_quality_gate(
            metrics,
            minimum_roc_auc=0.70,
            minimum_recall=0.60,
        )