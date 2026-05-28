"""Tests for data quality profiling."""
import numpy as np
import pandas as pd

from fabsentinel.config import QualityCfg
from fabsentinel.data.quality import clean, profile


def test_profile_basic_shape(secom_like_df):
    cfg = QualityCfg()
    rep = profile(secom_like_df, cfg)
    assert rep.n_lots == 200
    assert rep.n_signals == 30
    assert rep.n_fails == 10


def test_profile_drops_high_missing():
    df = pd.DataFrame({
        "label": np.ones(50, dtype=int) * -1,
        "signal_000": np.random.randn(50),
        "signal_001": [np.nan] * 45 + [1.0] * 5,  # 90% missing
    })
    cfg = QualityCfg(missing_threshold=0.5)
    rep = profile(df, cfg)
    assert "signal_001" in rep.signals_dropped_missing
    assert "signal_000" not in rep.signals_dropped_missing


def test_profile_drops_lowvar():
    df = pd.DataFrame({
        "label": np.ones(50, dtype=int) * -1,
        "signal_000": np.random.randn(50),
        "signal_001": np.zeros(50),  # zero variance
    })
    cfg = QualityCfg(variance_threshold=1e-9)
    rep = profile(df, cfg)
    assert "signal_001" in rep.signals_dropped_lowvar


def test_clean_no_nan_in_output(secom_like_df):
    cfg = QualityCfg()
    rep = profile(secom_like_df, cfg)
    cleaned = clean(secom_like_df, rep)
    sigs = rep.cleaned_signal_columns
    assert cleaned[sigs].isna().sum().sum() == 0


def test_imbalance_warning_triggers():
    df = pd.DataFrame({
        "label": [-1] * 95 + [1] * 5,
        "signal_000": np.random.randn(100),
    })
    cfg = QualityCfg(imbalance_warn_ratio=5.0)
    rep = profile(df, cfg)
    assert rep.pass_fail_imbalance_warning is True
