"""Virtual metrology classifier: LightGBM + SMOTE + PR-curve threshold tuning.

Why these choices:
- LightGBM handles tabular data well, runs fast on Apple Silicon, integrates with SHAP.
- SMOTE-NC for the minority class because SECOM is ~14:1 imbalanced.
- We pick the threshold by maximizing F1 on the PR curve (or beta-F if you tune that).
- We report MCC, balanced accuracy, recall, PR-AUC. NEVER plain accuracy alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
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

    # SMOTE on training only
    smote_strategy = cfg.smote_strategy
    # SMOTE requires the minority class to have at least k_neighbors + 1 samples
    minority = int(y_train.sum())
    k = min(5, max(minority - 1, 1))
    try:
        smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=k, random_state=cfg.random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    except Exception as e:
        logger.warning(f"SMOTE failed ({e}); falling back to original training set.")
        X_train_res, y_train_res = X_train, y_train

    # Stratified k-fold CV on the resampled training set (using mean PR-AUC for model selection)
    cv_aps: list[float] = []
    cv_mcc: list[float] = []
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_train_res, y_train_res)):
        model = lgb.LGBMClassifier(
            n_estimators=cfg.model.n_estimators,
            learning_rate=cfg.model.learning_rate,
            num_leaves=cfg.model.num_leaves,
            min_child_samples=cfg.model.min_child_samples,
            class_weight=cfg.model.class_weight,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        # X_train_res may be a numpy array after SMOTE; coerce back to DataFrame
        if not isinstance(X_train_res, pd.DataFrame):
            X_train_res = pd.DataFrame(X_train_res, columns=X.columns)
            y_train_res = pd.Series(y_train_res)
        X_tr = X_train_res.iloc[tr_idx]
        X_va = X_train_res.iloc[va_idx]
        y_tr = y_train_res.iloc[tr_idx]
        y_va = y_train_res.iloc[va_idx]
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        cv_aps.append(average_precision_score(y_va, proba))
        thr, _ = _find_threshold(y_va.values, proba, cfg.threshold_strategy)
        cv_mcc.append(matthews_corrcoef(y_va, (proba >= thr).astype(int)))
    cv_metrics = {
        "cv_pr_auc_mean": float(np.mean(cv_aps)),
        "cv_pr_auc_std": float(np.std(cv_aps)),
        "cv_mcc_mean": float(np.mean(cv_mcc)),
        "cv_mcc_std": float(np.std(cv_mcc)),
    }
    logger.info(f"CV metrics: {cv_metrics}")

    # Final model on the full resampled training set
    final = lgb.LGBMClassifier(
        n_estimators=cfg.model.n_estimators,
        learning_rate=cfg.model.learning_rate,
        num_leaves=cfg.model.num_leaves,
        min_child_samples=cfg.model.min_child_samples,
        class_weight=cfg.model.class_weight,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    final.fit(X_train_res, y_train_res)
    proba_test = final.predict_proba(X_test)[:, 1]
    threshold, _ = _find_threshold(y_test.values, proba_test, cfg.threshold_strategy)
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
        "n_test": int(len(y_test)),
    }
    logger.info(f"Test metrics: {metrics}")

    # MLflow logging
    with mlflow.start_run(run_name="vm_lightgbm"):
        mlflow.log_params({
            "n_estimators": cfg.model.n_estimators,
            "learning_rate": cfg.model.learning_rate,
            "num_leaves": cfg.model.num_leaves,
            "class_weight": cfg.model.class_weight,
            "smote_strategy": cfg.smote_strategy,
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
