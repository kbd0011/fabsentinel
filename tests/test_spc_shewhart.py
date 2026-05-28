"""Tests for Shewhart individuals chart."""
import numpy as np
import pytest

from fabsentinel.spc.shewhart import fit_individuals, detect


def test_in_control_no_ooc(in_control_series):
    limits = fit_individuals(in_control_series)
    mask = detect(in_control_series, limits)
    # On 200 IID N(0,1) points at 3-sigma, expect very few OOC
    assert mask.sum() <= 3


def test_shift_detected(shifted_series):
    # Fit limits on the in-control portion only
    limits = fit_individuals(shifted_series[:100])
    mask = detect(shifted_series, limits)
    # The shifted second half should contain many OOC points
    assert mask[100:].sum() > 20


def test_estimated_sigma_close_to_truth(rng):
    arr = rng.normal(loc=5.0, scale=2.0, size=2000)
    limits = fit_individuals(arr)
    # d2-based estimator should land within ~10% of true sigma
    assert abs(limits.sigma - 2.0) / 2.0 < 0.10
    # Sample mean ~ N(5, sigma/sqrt(2000)). Three SEs is 3 * 2 / sqrt(2000) ≈ 0.134.
    assert abs(limits.mean - 5.0) < 0.2


def test_requires_minimum_observations():
    with pytest.raises(ValueError):
        fit_individuals(np.array([1.0]))
