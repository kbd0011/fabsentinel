"""Data quality profiling for SECOM."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from fabsentinel.config import QualityCfg


@dataclass
class QualityReport:
    n_lots: int
    n_signals: int
    n_fails: int
    pass_fail_ratio: float
    missingness_by_signal: pd.Series             # fraction missing per signal
    signals_dropped_missing: list[str]
    signals_dropped_lowvar: list[str]
    outlier_counts: pd.Series                    # |z| > threshold per signal
    pass_fail_imbalance_warning: bool
    cleaned_signal_columns: list[str]            # signals retained after dropping


def profile(df: pd.DataFrame, cfg: QualityCfg) -> QualityReport:
    """Profile data quality of a SECOM-shaped DataFrame.

    Assumes columns: timestamp, label, signal_000, signal_001, ...
    """
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    if not signal_cols:
        raise ValueError("No signal_xxx columns found in DataFrame")

    n_lots = len(df)
    n_signals = len(signal_cols)
    if "label" in df.columns:
        n_fails = int((df["label"] == 1).sum())
    else:
        n_fails = 0
    ratio = (n_lots - n_fails) / max(n_fails, 1)

    missing_frac = df[signal_cols].isna().mean()
    missing_frac.name = "missing_fraction"
    dropped_missing = missing_frac[missing_frac > cfg.missing_threshold].index.tolist()

    sub = df[signal_cols].drop(columns=dropped_missing)
    variances = sub.var(numeric_only=True)
    dropped_lowvar = variances[variances < cfg.variance_threshold].index.tolist()

    retained = [c for c in signal_cols if c not in dropped_missing and c not in dropped_lowvar]
    z = (df[retained] - df[retained].mean()) / df[retained].std(ddof=0).replace(0, np.nan)
    outlier_counts = (z.abs() > cfg.outlier_zscore).sum().sort_values(ascending=False)
    outlier_counts.name = "outlier_count"

    imbalance_warning = ratio > cfg.imbalance_warn_ratio
    if imbalance_warning:
        logger.warning(
            f"Severe class imbalance: pass:fail = {ratio:.1f}:1 > "
            f"warning threshold {cfg.imbalance_warn_ratio:.1f}:1. "
            "Always report MCC / PR-AUC, never accuracy."
        )

    logger.info(
        f"Quality profile: n_lots={n_lots} n_signals={n_signals} "
        f"dropped_missing={len(dropped_missing)} dropped_lowvar={len(dropped_lowvar)} "
        f"retained={len(retained)}"
    )

    return QualityReport(
        n_lots=n_lots,
        n_signals=n_signals,
        n_fails=n_fails,
        pass_fail_ratio=ratio,
        missingness_by_signal=missing_frac,
        signals_dropped_missing=dropped_missing,
        signals_dropped_lowvar=dropped_lowvar,
        outlier_counts=outlier_counts,
        pass_fail_imbalance_warning=imbalance_warning,
        cleaned_signal_columns=retained,
    )


def clean(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    """Return a cleaned DataFrame: drop low-quality signals, median-impute missing."""
    keep_cols = [c for c in df.columns if not c.startswith("signal_")] + report.cleaned_signal_columns
    out = df[keep_cols].copy()
    signal_cols = report.cleaned_signal_columns
    medians = out[signal_cols].median()
    out[signal_cols] = out[signal_cols].fillna(medians)
    return out
