"""Tests for VM evaluation metrics."""
import numpy as np

from fabsentinel.vm.evaluate import metrics, pr_curve


def test_metrics_perfect_classifier():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.95, 0.99])
    m = metrics(y_true, y_proba, threshold=0.5)
    assert m["mcc"] == 1.0
    assert m["recall"] == 1.0
    assert m["pr_auc"] == 1.0


def test_metrics_all_negative_predictions_safe():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.45])
    m = metrics(y_true, y_proba, threshold=0.5)
    # All predicted negative; recall = 0
    assert m["recall"] == 0.0
    assert m["tp"] == 0
    assert m["fn"] == 2


def test_pr_curve_has_expected_columns(rng):
    y_true = rng.integers(0, 2, size=100)
    y_proba = rng.uniform(size=100)
    df = pr_curve(y_true, y_proba)
    assert {"threshold", "precision", "recall"}.issubset(df.columns)
    assert len(df) > 0
