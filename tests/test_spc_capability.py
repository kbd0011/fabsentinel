"""Tests for Cp / Cpk capability calculation."""
import numpy as np
import pandas as pd

from fabsentinel.spc.capability import estimate, simulate_specs_and_estimate


def test_cp_matches_textbook_value():
    """If USL-LSL = 8 sigma, then Cp = 8/6 = 1.333..."""
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=100.0, scale=2.0, size=2000)
    sigma = arr.std(ddof=1)
    usl = 100 + 4 * sigma
    lsl = 100 - 4 * sigma
    r = estimate(arr, usl=usl, lsl=lsl)
    assert abs(r.cp - 4 * 2 / 6) < 0.01  # 8/6 = 1.333
    assert r.cpk <= r.cp  # always


def test_cpk_smaller_than_cp_when_offcenter():
    """A shifted process has Cpk < Cp."""
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=102.0, scale=2.0, size=2000)
    r = estimate(arr, usl=110.0, lsl=90.0)
    assert r.cp > r.cpk
    assert r.cpk < r.cp


def test_simulate_specs_returns_expected_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"signal_{i:03d}": rng.normal(size=200) for i in range(5)})
    out = simulate_specs_and_estimate(df, list(df.columns), sigma_margin=4.0)
    assert {"signal", "cp", "cpk", "mean", "sigma"}.issubset(out.columns)
    # mean Cp for sigma_margin=4 is exactly 8/6 = 1.333...
    assert abs(out["cp"].mean() - 4 * 2 / 6) < 0.05
