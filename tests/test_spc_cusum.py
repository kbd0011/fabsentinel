"""Tests for CUSUM chart."""
import numpy as np

from fabsentinel.spc.cusum import fit_and_detect


def test_cusum_detects_sustained_shift(rng):
    a = rng.normal(loc=0, scale=1, size=100)
    b = rng.normal(loc=1.5, scale=1, size=100)
    series = np.concatenate([a, b])
    _, mask = fit_and_detect(series, k=0.5, h=5.0,
                             target_mean=0.0, target_sigma=1.0)
    assert mask[100:].sum() > 5


def test_cusum_in_control_low_false_alarm(in_control_series):
    _, mask = fit_and_detect(in_control_series, k=0.5, h=5.0)
    # ARL_0 ~ 465 for k=0.5, h=5; on 200 points expect almost no alarms
    assert mask.sum() <= 5


def test_cusum_negative_shift_caught_by_c_minus(rng):
    a = rng.normal(loc=0, scale=1, size=100)
    b = rng.normal(loc=-1.5, scale=1, size=100)
    series = np.concatenate([a, b])
    state, _ = fit_and_detect(series, k=0.5, h=5.0,
                              target_mean=0.0, target_sigma=1.0)
    assert state.c_minus.min() < -5.0  # in sigma units after scaling
