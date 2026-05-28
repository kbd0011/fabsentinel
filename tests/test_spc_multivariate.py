"""Tests for PCA-based multivariate SPC."""
import numpy as np
import pandas as pd

from fabsentinel.spc.multivariate import fit, compute_statistics, contribution_plot


def test_fit_returns_valid_model(rng):
    X = pd.DataFrame(rng.normal(size=(300, 20)),
                     columns=[f"s{i}" for i in range(20)])
    m = fit(X, explained_variance=0.90, alpha=0.01)
    assert m.n_components >= 1
    assert m.n_components <= 20
    assert m.t2_ucl > 0
    assert m.spe_ucl > 0
    assert m.explained_variance >= 0.90


def test_t2_detects_outliers(rng):
    X_train = pd.DataFrame(rng.normal(size=(500, 10)),
                           columns=[f"s{i}" for i in range(10)])
    m = fit(X_train, alpha=0.01)
    # Inject a clear outlier
    X_test = X_train.copy()
    X_test.iloc[0, :5] = 8.0  # huge multivariate outlier
    stats = compute_statistics(m, X_test)
    assert stats["t2_ooc"].iloc[0]


def test_contribution_plot_returns_top_k(rng):
    X = pd.DataFrame(rng.normal(size=(200, 15)),
                     columns=[f"s{i}" for i in range(15)])
    m = fit(X, alpha=0.01)
    x_row = X.iloc[0]
    contrib = contribution_plot(m, x_row, top_k=5)
    assert len(contrib) == 5
    # Values are absolute contributions, so non-negative
    assert (contrib >= 0).all()


def test_t2_false_alarm_rate_bounded(rng):
    """For alpha=0.01 we expect roughly 1% T^2 OOC on held-out IC data."""
    X_train = pd.DataFrame(rng.normal(size=(1000, 10)),
                           columns=[f"s{i}" for i in range(10)])
    X_test = pd.DataFrame(rng.normal(size=(1000, 10)),
                          columns=[f"s{i}" for i in range(10)])
    m = fit(X_train, alpha=0.01)
    stats = compute_statistics(m, X_test)
    rate = stats["t2_ooc"].mean()
    # Be generous: 0.1% - 5% (finite-sample effects)
    assert 0.001 <= rate <= 0.05
