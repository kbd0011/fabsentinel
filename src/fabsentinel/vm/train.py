"""Virtual metrology classifier: cost-sensitive LightGBM + PR-curve threshold tuning.

Why these choices:
- LightGBM handles tabular data well, runs fast on Apple Silicon, integrates with SHAP.
- Cost-sensitive learning (``scale_pos_weight = n_neg / n_pos``) handles SECOM's ~14:1
  imbalance WITHOUT resampling. This is the key correctness decision: SMOTE-balancing the
  training set inflates the model's output-probability scale to the resampled prior, so a
  threshold tuned on CV predictions does not transfer to the natural-prior (~6.6%) test set.
  Class weighting keeps probabilities on the natural-prior scale, so the out-of-fold (OOF)
  threshold transfers honestly to the held-out test set.
- We pick the threshold by maximizing F1 on the PR curve of the natural-prior OOF predictions.
- We report MCC, balanced accuracy, recall, PR-AUC. NEVER plain accuracy alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from fabsentinel.config import VMCfg


@dataclass
class VMArtifact:
    """Trained model + metadata; written to disk by train()."""
    model: lgb.LGBMClassifier
    threshold: float
    feature_names: list[str]
    metrics: dict
    cv_metrics: dict


def _find_threshold(y_true: np.ndarray, y_proba: np.ndarray, strategy: str = "pr_max_f1") -> tuple[float, float]:
    """Return (threshold, achieved_f1)."""
    p, r, t = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns thresholds of length n-1
    if strategy == "pr_max_f1":
        f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
        idx = int(np.nanargmax(f1[:-1])) if len(t) > 0 else 0
        thr = float(t[idx]) if len(t) > 0 else 0.5
        return thr, float(f1[idx])
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def train(X: pd.DataFrame, y: pd.Series, cfg: VMCfg, mlflow_tracking_uri: str | None = None) -> VMArtifact:
    """Train the VM classifier and save artifact to cfg.artifact_path."""
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("fabsentinel-vm")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )

    # Cost-sensitive learning instead of SMOTE: weight the minority class by n_neg / n_pos so
    # the model focuses on fails WITHOUT resampling. Crucially, this leaves the predicted
    # probabilities on the natural-prior scale, so a threshold tuned on the OOF (natural-prior)
    # validation folds transfers to the held-out test set. SMOTE would inflate the probability
    # scale to the resampled prior and break that transfer.
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    def _make_model() -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=cfg.model.n_estimators,
            learning_rate=cfg.model.learning_rate,
            num_leaves=cfg.model.num_leaves,
            min_child_samples=cfg.model.min_child_samples,
            scale_pos_weight=scale_pos_weight,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    # Stratified k-fold CV on the natural-prior training set (mean PR-AUC for model selection).
    # We pool the out-of-fold (OOF) predictions to derive the production threshold WITHOUT
    # touching the test set, then freeze that threshold and apply it to the held-out test set.
    # Because no fold is resampled, the OOF predictions share the test set's natural prior.
    cv_aps: list[float] = []
    cv_mcc: list[float] = []
    oof_y: list[np.ndarray] = []
    oof_proba: list[np.ndarray] = []
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    for tr_idx, va_idx in skf.split(X_train, y_train):
        model = _make_model()
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_train.iloc[tr_idx]
        y_va = y_train.iloc[va_idx]
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        cv_aps.append(average_precision_score(y_va, proba))
        thr, _ = _find_threshold(y_va.values, proba, cfg.threshold_strategy)
        cv_mcc.append(matthews_corrcoef(y_va, (proba >= thr).astype(int)))
        oof_y.append(y_va.values)
        oof_proba.append(proba)

    # Freeze the production threshold on the pooled OOF predictions (no test-set leakage).
    oof_y_all = np.concatenate(oof_y)
    oof_proba_all = np.concatenate(oof_proba)
    threshold, _ = _find_threshold(oof_y_all, oof_proba_all, cfg.threshold_strategy)
    oof_yhat = (oof_proba_all >= threshold).astype(int)
    cv_metrics = {
        "cv_pr_auc_mean": float(np.mean(cv_aps)),
        "cv_pr_auc_std": float(np.std(cv_aps)),
        "cv_mcc_mean": float(np.mean(cv_mcc)),
        "cv_mcc_std": float(np.std(cv_mcc)),
        "frozen_threshold": float(threshold),
        # OOF operating-point metrics at the frozen threshold (natural prior) — these should
        # roughly match the test-set numbers below, demonstrating the threshold transfers.
        "oof_mcc": float(matthews_corrcoef(oof_y_all, oof_yhat)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(oof_y_all, oof_yhat)),
        "oof_recall": float(recall_score(oof_y_all, oof_yhat, zero_division=0)),
        "oof_f1": float(f1_score(oof_y_all, oof_yhat, zero_division=0)),
        "scale_pos_weight": float(scale_pos_weight),
    }
    logger.info(f"CV metrics: {cv_metrics}")

    # Final model on the full natural-prior training set (cost-sensitive, no resampling).
    final = _make_model()
    final.fit(X_train, y_train)
    proba_test = final.predict_proba(X_test)[:, 1]
    # Apply the FROZEN threshold derived from OOF CV predictions; never re-tune on the test set.
    yhat = (proba_test >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
    metrics = {
        "test_pr_auc": float(average_precision_score(y_test, proba_test)),
        "test_roc_auc": float(roc_auc_score(y_test, proba_test)),
        "test_mcc": float(matthews_corrcoef(y_test, yhat)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, yhat)),
        "test_recall": float(recall_score(y_test, yhat)),
        "test_f1": float(f1_score(y_test, yhat)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
        "n_test": len(y_test),
    }
    logger.info(f"Test metrics: {metrics}")

    # MLflow logging
    with mlflow.start_run(run_name="vm_lightgbm"):
        mlflow.log_params({
            "n_estimators": cfg.model.n_estimators,
            "learning_rate": cfg.model.learning_rate,
            "num_leaves": cfg.model.num_leaves,
            "imbalance_strategy": "scale_pos_weight",
            "scale_pos_weight": scale_pos_weight,
            "cv_folds": cfg.cv_folds,
            "threshold_strategy": cfg.threshold_strategy,
        })
        mlflow.log_metrics({**metrics, **cv_metrics})

    artifact = VMArtifact(
        model=final,
        threshold=threshold,
        feature_names=list(X.columns),
        metrics=metrics,
        cv_metrics=cv_metrics,
    )

    cfg.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, cfg.artifact_path)
    logger.info(f"Saved VM artifact to {cfg.artifact_path}")
    return artifact


def load(artifact_path: Path) -> VMArtifact:
    """Load a previously trained VM artifact."""
    if not artifact_path.exists():
        raise FileNotFoundError(f"VM artifact not found: {artifact_path}")
    return joblib.load(artifact_path)
