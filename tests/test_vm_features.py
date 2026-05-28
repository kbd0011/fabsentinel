"""Tests for VM feature prep."""
import numpy as np

from fabsentinel.vm.features import prepare


def test_prepare_remaps_labels(secom_like_df):
    signal_cols = [c for c in secom_like_df.columns if c.startswith("signal_")]
    X, y = prepare(secom_like_df, signal_cols)
    assert X.shape == (200, 30)
    # Labels are remapped: -1 -> 0, +1 -> 1
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 10  # 10 original +1s become 10 ones


def test_prepare_handles_nan(secom_like_df):
    df = secom_like_df.copy()
    df.loc[5, "signal_010"] = np.nan
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    X, y = prepare(df, signal_cols)
    assert not X.isna().any().any()


def test_prepare_requires_label_column():
    import pandas as pd
    df = pd.DataFrame({"signal_000": [1, 2, 3]})
    import pytest
    with pytest.raises(ValueError):
        prepare(df, ["signal_000"])
