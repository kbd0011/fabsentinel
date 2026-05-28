"""Evaluation helpers for the VM model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


def metrics(y_true: pd.Series | np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Compute the full FabSentinel metric set at a given threshold."""
    yt = np.asarray(y_true).astype(int)
    yhat = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yhat, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(yt, y_proba)),
        "roc_auc": float(roc_auc_score(yt, y_proba)),
        "mcc": float(matthews_corrcoef(yt, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yhat)),
        "recall": float(recall_score(yt, yhat, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def pr_curve(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    """Return precision / recall / threshold for plotting."""
    p, r, t = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(y_proba))
    t = np.append(t, 1.0)
    return pd.DataFrame({"threshold": t, "precision": p, "recall": r})
