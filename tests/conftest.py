"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def in_control_series(rng):
    """200 IID N(10, 1) samples - process in control."""
    return rng.normal(loc=10.0, scale=1.0, size=200)


@pytest.fixture
def shifted_series(rng):
    """100 N(10,1) followed by 100 N(13,1) - clear mean shift."""
    a = rng.normal(loc=10.0, scale=1.0, size=100)
    b = rng.normal(loc=13.0, scale=1.0, size=100)
    return np.concatenate([a, b])


@pytest.fixture
def secom_like_df(rng):
    """Small fake SECOM-shaped DataFrame: 200 lots x 30 signals + label + timestamp."""
    n = 200
    p = 30
    X = rng.normal(size=(n, p))
    df = pd.DataFrame(X, columns=[f"signal_{i:03d}" for i in range(p)])
    # 5% fails to mirror imbalance
    y = np.ones(n, dtype=int) * -1
    fail_idx = rng.choice(n, size=10, replace=False)
    y[fail_idx] = 1
    # Move fails away from mean to make them learnable
    df.loc[fail_idx, "signal_000"] += 3.0
    df.loc[fail_idx, "signal_001"] -= 3.0
    df.insert(0, "label", y)
    df.insert(0, "timestamp", pd.date_range("2024-01-01", periods=n, freq="h"))
    return df
