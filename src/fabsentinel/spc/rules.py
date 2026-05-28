"""Western Electric and Nelson run rules for Shewhart charts.

References:
- AT&T Statistical Quality Control Handbook (Western Electric), 1956.
- Nelson, L.S., "The Shewhart Control Chart - Tests for Special Causes," J. Quality Tech., 1984.
"""
from __future__ import annotations

import numpy as np


def western_electric(x: np.ndarray, mean: float, sigma: float, rules: list[int] | None = None) -> np.ndarray:
    """Western Electric rules 1-4.

    Returns a boolean OOC mask the same length as x.
    1: One point > 3 sigma from mean.
    2: Two of three consecutive points > 2 sigma (same side).
    3: Four of five consecutive points > 1 sigma (same side).
    4: Eight consecutive points on same side of mean.
    """
    if rules is None:
        rules = [1, 2, 3, 4]
    n = len(x)
    out = np.zeros(n, dtype=bool)
    if sigma <= 0:
        return out

    z = (np.asarray(x, dtype=float) - mean) / sigma

    if 1 in rules:
        out |= np.abs(z) > 3.0
    if 2 in rules:
        big = z > 2.0
        small = z < -2.0
        for i in range(2, n):
            if big[i - 2 : i + 1].sum() >= 2 or small[i - 2 : i + 1].sum() >= 2:
                out[i] = True
    if 3 in rules:
        big = z > 1.0
        small = z < -1.0
        for i in range(4, n):
            if big[i - 4 : i + 1].sum() >= 4 or small[i - 4 : i + 1].sum() >= 4:
                out[i] = True
    if 4 in rules:
        sign = np.sign(z)
        for i in range(7, n):
            window = sign[i - 7 : i + 1]
            if (window > 0).all() or (window < 0).all():
                out[i] = True
    return out


def nelson(x: np.ndarray, mean: float, sigma: float, rules: list[int] | None = None) -> np.ndarray:
    """Nelson rules 1-8 (Nelson 1984).

    1: One point > 3 sigma from mean.
    2: Nine points in a row on same side of mean.
    3: Six points in a row, all increasing or all decreasing.
    4: Fourteen points in a row alternating up and down.
    5: Two of three consecutive points > 2 sigma (same side).
    6: Four of five consecutive points > 1 sigma (same side).
    7: Fifteen points in a row within 1 sigma of mean (stratification).
    8: Eight points in a row outside 1 sigma of mean on either side.
    """
    if rules is None:
        rules = [1, 2, 3, 4, 5, 6, 7, 8]
    n = len(x)
    out = np.zeros(n, dtype=bool)
    if sigma <= 0:
        return out

    z = (np.asarray(x, dtype=float) - mean) / sigma
    sign = np.sign(z)

    if 1 in rules:
        out |= np.abs(z) > 3.0
    if 2 in rules:
        for i in range(8, n):
            w = sign[i - 8 : i + 1]
            if (w > 0).all() or (w < 0).all():
                out[i] = True
    if 3 in rules:
        diffs = np.diff(x)
        for i in range(5, n - 1):
            seg = diffs[i - 5 : i]
            if (seg > 0).all() or (seg < 0).all():
                out[i] = True
    if 4 in rules:
        diffs = np.diff(x)
        sign_d = np.sign(diffs)
        for i in range(13, n - 1):
            seg = sign_d[i - 13 : i]
            if len(seg) < 13:
                continue
            alternates = np.all(seg[:-1] * seg[1:] < 0)
            if alternates:
                out[i] = True
    if 5 in rules:
        big = z > 2.0
        small = z < -2.0
        for i in range(2, n):
            if big[i - 2 : i + 1].sum() >= 2 or small[i - 2 : i + 1].sum() >= 2:
                out[i] = True
    if 6 in rules:
        big = z > 1.0
        small = z < -1.0
        for i in range(4, n):
            if big[i - 4 : i + 1].sum() >= 4 or small[i - 4 : i + 1].sum() >= 4:
                out[i] = True
    if 7 in rules:
        within = np.abs(z) < 1.0
        for i in range(14, n):
            if within[i - 14 : i + 1].all():
                out[i] = True
    if 8 in rules:
        outside = np.abs(z) > 1.0
        for i in range(7, n):
            if outside[i - 7 : i + 1].all():
                out[i] = True
    return out
