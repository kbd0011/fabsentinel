"""Tests for VM training (small-data, fast)."""
import numpy as np
import pandas as pd
import pytest

from fabsentinel.config import VMCfg, VMModelCfg
from fabsentinel.vm.train import train, load


def _cfg(tmp_path):
    return VMCfg(
        test_size=0.3,
        random_state=0,
        cv_folds=2,
        smote_strategy=0.5,
        model=VMModelCfg(n_estimators=30, learning_rate=0.1, num_leaves=15),
        threshold_strategy="pr_max_f1",
        artifact_path=tmp_path / "vm.joblib",
    )


def test_train_small_synthetic(rng, tmp_path):
    n = 300
    p = 10
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"s{i}" for i in range(p)])
    y = pd.Series(np.zeros(n, dtype=int))
    fails = rng.choice(n, size=30, replace=False)
    y.loc[fails] = 1
    # Plant a strong signal in s0 and s1
    X.loc[fails, "s0"] += 3.0
    X.loc[fails, "s1"] -= 3.0

    cfg = _cfg(tmp_path)
    artifact = train(X, y, cfg)
    assert artifact.metrics["test_pr_auc"] > 0.30  # learnable signal
    assert artifact.metrics["test_mcc"] > 0.0
    assert cfg.artifact_path.exists()

    # Round-trip via load
    loaded = load(cfg.artifact_path)
    assert loaded.threshold == artifact.threshold


def test_train_extreme_imbalance_does_not_crash(rng, tmp_path):
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = pd.Series(np.zeros(n, dtype=int))
    y.iloc[:3] = 1  # only 3 positives
    cfg = _cfg(tmp_path)
    artifact = train(X, y, cfg)
    assert artifact.metrics is not None
