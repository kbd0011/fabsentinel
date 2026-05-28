"""Tests for EWMA chart."""
import numpy as np

from fabsentinel.spc.ewma import fit_and_detect


def test_ewma_detects_small_shift_faster_than_shewhart(rng):
    # Construct a small (1-sigma) shift in the middle of the series
    a = rng.normal(loc=0, scale=1, size=200)
    b = rng.normal(loc=1.0, scale=1, size=200)
    series = np.concatenate([a, b])
    _, mask = fit_and_detect(series, lam=0.2, L=3.0,
                             target_mean=0.0, target_sigma=1.0)
    # EWMA should fire somewhere in the shifted region
    assert mask[200:].sum() > 0


def test_ewma_in_control_few_alarms(in_control_series):
    _, mask = fit_and_detect(in_control_series, lam=0.2, L=3.0)
    # Should have very few false alarms on stable data
    assert mask.sum() < len(in_control_series) * 0.05


def test_ewma_limits_widen_with_time():
    """Time-varying limits should asymptotically approach steady-state."""
    arr = np.zeros(500)
    state, _ = fit_and_detect(arr, lam=0.2, L=3.0, target_mean=0.0, target_sigma=1.0)
    ucl_early = state.ucl[5]
    ucl_late = state.ucl[400]
    assert ucl_late > ucl_early
