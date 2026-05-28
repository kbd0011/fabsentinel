"""Tests for SHAP attribution."""
import numpy as np
import pandas as pd

from fabsentinel.config import VMCfg, VMModelCfg
from fabsentinel.vm.train import train
from fabsentinel.rca.shap_attribution import fit, top_features_for_lot


def _vm_cfg(tmp_path):
    return VMCfg(
        test_size=0.3,
        random_state=0,
        cv_folds=2,
        smote_strategy=0.5,
        model=VMModelCfg(n_estimators=30, learning_rate=0.1, num_leaves=15),
        threshold_strategy="pr_max_f1",
        artifact_path=tmp_path / "vm.joblib",
    )


def test_shap_global_top_includes_signal_features(rng, tmp_path):
    """A model that depends on s0 and s1 should rank them at the top."""
    n = 300
    p = 10
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"s{i}" for i in range(p)])
    y = pd.Series(np.zeros(n, dtype=int))
    fails = rng.choice(n, size=30, replace=False)
    y.loc[fails] = 1
    X.loc[fails, "s0"] += 3.0
    X.loc[fails, "s1"] -= 3.0

    artifact = train(X, y, _vm_cfg(tmp_path))
    report = fit(artifact, X, n_background=50)
    top = report.global_importance.head(3).index.tolist()
    # At least one of the true signal features should be in the top 3
    assert "s0" in top or "s1" in top


def test_top_features_for_lot_shape(rng, tmp_path):
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=list("abcdefgh"))
    y = pd.Series(np.zeros(n, dtype=int))
    fails = rng.choice(n, size=20, replace=False)
    y.loc[fails] = 1
    X.loc[fails, "a"] += 3.0

    artifact = train(X, y, _vm_cfg(tmp_path))
    report = fit(artifact, X, n_background=30)
    local = top_features_for_lot(report, X.iloc[0], top_k=5)
    assert len(local) == 5
    assert {"feature", "value", "shap_value", "abs_shap"}.issubset(local.columns)
