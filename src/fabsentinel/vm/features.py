"""Feature preparation for the virtual metrology classifier."""
from __future__ import annotations

import pandas as pd


def prepare(df: pd.DataFrame, signal_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Return X, y with SECOM label remapped to 0/1 (1 = fail).

    Assumes df already had medians imputed by data.quality.clean.
    """
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the input DataFrame")
    X = df[signal_columns].copy()
    # Belt-and-braces: any remaining NaNs get the median
    X = X.fillna(X.median())
    y_raw = df["label"]
    # SECOM uses -1 = pass, +1 = fail. We want 1 = fail (positive class).
    y = (y_raw == 1).astype(int)
    return X, y
