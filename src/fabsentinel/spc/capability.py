"""Cp / Cpk capability indices on signals with simulated specification limits.

IMPORTANT: SECOM is anonymized.  Real specification limits do not exist in the dataset.
This module demonstrates capability analysis on a subset where USL and LSL are
SIMULATED from the in-control mean and sigma. The methodology is correct; the
specific Cp / Cpk values are illustrative, not engineering truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CapabilityResult:
    signal: str
    mean: float
    sigma: float
    usl: float
    lsl: float
    cp: float
    cpk: float
    cpm: float | None
    notes: str


def estimate(values: np.ndarray | pd.Series, usl: float, lsl: float, target: float | None = None) -> CapabilityResult:
    """Estimate Cp, Cpk, and (optionally) Cpm for a single signal.

    Cp  = (USL - LSL) / (6 sigma)
    Cpk = min((USL - mu) / (3 sigma), (mu - LSL) / (3 sigma))
    Cpm = (USL - LSL) / (6 sqrt(sigma^2 + (mu - T)^2))
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 5:
        raise ValueError("Need at least 5 observations for capability estimation")
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    if sigma <= 0:
        return CapabilityResult(
            signal="<unnamed>", mean=mu, sigma=sigma, usl=usl, lsl=lsl,
            cp=float("inf"), cpk=float("inf"), cpm=None,
            notes="degenerate: sigma <= 0",
        )
    cp = (usl - lsl) / (6 * sigma)
    cpk = min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma))
    cpm = None
    if target is not None:
        denom = 6 * np.sqrt(sigma ** 2 + (mu - target) ** 2)
        cpm = float((usl - lsl) / denom)
    return CapabilityResult(
        signal="<unnamed>", mean=mu, sigma=sigma, usl=usl, lsl=lsl,
        cp=float(cp), cpk=float(cpk), cpm=cpm, notes="",
    )


def simulate_specs_and_estimate(
    df: pd.DataFrame,
    signal_columns: list[str],
    sigma_margin: float = 4.0,
) -> pd.DataFrame:
    """For each given signal, derive simulated USL / LSL and compute Cp / Cpk.

    The simulated spec margin is sigma_margin * in-sample sigma on each side of the mean.
    A Cp of 1.33 is the typical "capable" threshold; for sigma_margin=4 we expect Cp ~= 1.33.
    """
    rows: list[dict] = []
    for col in signal_columns:
        if col not in df.columns:
            continue
        arr = df[col].dropna().values
        if arr.size < 5:
            continue
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=1))
        usl = mu + sigma_margin * sigma
        lsl = mu - sigma_margin * sigma
        result = estimate(arr, usl=usl, lsl=lsl, target=mu)
        rows.append(
            {
                "signal": col,
                "mean": result.mean,
                "sigma": result.sigma,
                "usl": result.usl,
                "lsl": result.lsl,
                "cp": result.cp,
                "cpk": result.cpk,
                "cpm": result.cpm,
                "notes": "simulated specs; SECOM has no real engineering specs",
            }
        )
    return pd.DataFrame(rows)
