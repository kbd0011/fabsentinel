"""Shewhart individuals control chart (subgroup size = 1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ShewhartLimits:
    mean: float
    sigma: float
    ucl: float
    lcl: float


def fit_individuals(x: np.ndarray | pd.Series, sigma_multiplier: float = 3.0) -> ShewhartLimits:
    """Fit Shewhart individuals limits using moving range estimate of sigma.

    Uses d2 = 1.128 for moving range of consecutive observations (per Montgomery).
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        raise ValueError("Need at least 2 observations to estimate moving range")
    mean = float(arr.mean())
    moving_range = np.abs(np.diff(arr))
    mr_bar = float(moving_range.mean())
    # d2 for n=2 is 1.128
    sigma = mr_bar / 1.128
    ucl = mean + sigma_multiplier * sigma
    lcl = mean - sigma_multiplier * sigma
    return ShewhartLimits(mean=mean, sigma=sigma, ucl=ucl, lcl=lcl)


def detect(x: np.ndarray | pd.Series, limits: ShewhartLimits) -> np.ndarray:
    """Boolean mask: True where observation is outside the control limits."""
    arr = np.asarray(x, dtype=float)
    return (arr > limits.ucl) | (arr < limits.lcl)
