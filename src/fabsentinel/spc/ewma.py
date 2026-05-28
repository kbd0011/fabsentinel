"""Exponentially-Weighted Moving Average (EWMA) control chart."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EWMAState:
    z: np.ndarray
    mean: float
    sigma: float
    ucl: np.ndarray
    lcl: np.ndarray


def fit_and_detect(
    x: np.ndarray | pd.Series,
    lam: float = 0.2,
    L: float = 3.0,
    target_mean: float | None = None,
    target_sigma: float | None = None,
) -> tuple[EWMAState, np.ndarray]:
    """Compute the EWMA statistic, control limits, and OOC mask.

    Limits are time-varying:
        sigma_z(t) = sigma * sqrt( lam / (2 - lam) * (1 - (1 - lam)^(2t)) )
        UCL(t) = mu + L * sigma_z(t)
        LCL(t) = mu - L * sigma_z(t)
    """
    arr = np.asarray(x, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size < 2:
        raise ValueError("Need at least 2 observations for EWMA")
    mu = float(target_mean) if target_mean is not None else float(valid.mean())
    if target_sigma is not None:
        sigma = float(target_sigma)
    else:
        moving_range = np.abs(np.diff(valid))
        sigma = float(moving_range.mean()) / 1.128

    n = arr.shape[0]
    z = np.empty(n)
    z_prev = mu
    for i, xi in enumerate(arr):
        if np.isnan(xi):
            z[i] = z_prev  # carry previous
        else:
            z[i] = lam * xi + (1 - lam) * z_prev
            z_prev = z[i]

    t = np.arange(1, n + 1)
    factor = np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
    sigma_z = sigma * factor
    ucl = mu + L * sigma_z
    lcl = mu - L * sigma_z
    ooc = (z > ucl) | (z < lcl)
    return EWMAState(z=z, mean=mu, sigma=sigma, ucl=ucl, lcl=lcl), ooc
