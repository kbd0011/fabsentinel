"""Tests for Western Electric + Nelson rules."""
import numpy as np

from fabsentinel.spc.rules import western_electric, nelson


def test_rule_1_single_point_beyond_3sigma():
    arr = np.zeros(20)
    arr[10] = 4.0
    mask = western_electric(arr, mean=0.0, sigma=1.0, rules=[1])
    assert mask[10] is np.bool_(True) or mask[10] == True


def test_rule_4_eight_in_a_row_same_side():
    arr = np.array([0.5] * 10 + [-0.5] * 5)
    mask = western_electric(arr, mean=0.0, sigma=1.0, rules=[4])
    # Rule 4: 8 in a row same side - should fire at index 7 onwards in the first 10
    assert mask[7:10].any()


def test_nelson_rule_2_nine_in_a_row():
    arr = np.array([0.3] * 12 + [-0.5] * 3)
    mask = nelson(arr, mean=0.0, sigma=1.0, rules=[2])
    # Rule 2 needs 9 same-side; should fire after index 8
    assert mask[8:12].any()


def test_nelson_rule_3_six_trending():
    arr = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] + [0] * 5, dtype=float)
    mask = nelson(arr, mean=0.0, sigma=10.0, rules=[3])
    # Should fire at the end of the monotonic run
    assert mask[5] or mask[6]


def test_rules_robust_to_zero_sigma():
    arr = np.zeros(20)
    mask = western_electric(arr, mean=0.0, sigma=0.0, rules=[1, 2, 3, 4])
    assert not mask.any()  # No alarms when sigma is degenerate
