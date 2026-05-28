"""Tabular CUSUM (two-sided)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CUSUMState:
    c_plus: np.ndarray
    c_minus: np.ndarray
    mean: float
    sigma: float
    h: float
    k: float


def fit_and_detect(
    x: np.ndarray | pd.Series,
    k: float = 0.5,
    h: float = 5.0,
    target_mean: float | None = None,
    target_sigma: float | None = None,
) -> tuple[CUSUMState, np.ndarray]:
    """Standard tabular CUSUM with reference value k*sigma and decision interval h*sigma.

    Returns the state and an OOC mask: True where C+ > h*sigma or C- < -h*sigma.
    """
    arr = np.asarray(x, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size < 2:
        raise ValueError("Need at least 2 observations for CUSUM")
    mu = float(target_mean) if target_mean is not None else float(valid.mean())
    if target_sigma is not None:
        sigma = float(target_sigma)
    else:
        moving_range = np.abs(np.diff(valid))
        sigma = float(moving_range.mean()) / 1.128
    if sigma <= 0:
        raise ValueError("Estimated sigma is non-positive; cannot run CUSUM")

    K = k * sigma
    H = h * sigma
    n = arr.shape[0]
    c_plus = np.zeros(n)
    c_minus = np.zeros(n)
    for i in range(n):
        xi = arr[i] if not np.isnan(arr[i]) else mu
        prev_plus = c_plus[i - 1] if i > 0 else 0.0
        prev_minus = c_minus[i - 1] if i > 0 else 0.0
        c_plus[i] = max(0.0, prev_plus + (xi - mu) - K)
        c_minus[i] = min(0.0, prev_minus + (xi - mu) + K)
    ooc = (c_plus > H) | (c_minus < -H)
    return CUSUMState(
        c_plus=c_plus, c_minus=c_minus, mean=mu, sigma=sigma, h=H, k=K
    ), ooc
