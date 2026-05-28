"""SECOM dataset ingest.

SECOM has 1567 lots x 590 sensor signals + a labels file with timestamp and pass/fail.
Source: McCann and Johnston, UCI ML Repository, CC BY 4.0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SecomData:
    """Ingested SECOM dataset."""
    X: pd.DataFrame             # (n_lots, n_signals) float, NaNs preserved
    y: pd.Series                # (n_lots,) int -1/+1 (-1 = pass, +1 = fail)
    timestamps: pd.Series       # (n_lots,) datetime64
    n_lots: int
    n_signals: int
    n_fails: int


def load_secom(data_path: Path, labels_path: Path) -> SecomData:
    """Read raw SECOM .data files and return a typed dataset.

    Parameters
    ----------
    data_path:
        Path to secom.data - space-separated, NaN markers as 'NaN'.
    labels_path:
        Path to secom_labels.data - label and "DD/MM/YYYY HH:MM:SS".
    """
    if not data_path.exists():
        raise FileNotFoundError(f"SECOM data not found: {data_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"SECOM labels not found: {labels_path}")

    logger.info(f"Reading SECOM features from {data_path}")
    X = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        engine="python",
        na_values=["NaN"],
    )
    X.columns = [f"signal_{i:03d}" for i in range(X.shape[1])]

    logger.info(f"Reading SECOM labels from {labels_path}")
    # Format is: '<label> "DD/MM/YYYY HH:MM:SS"' on each line.
    # The timestamp contains an internal space, so do not split on whitespace -
    # parse manually instead.
    labels: list[int] = []
    timestamp_strings: list[str] = []
    with labels_path.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            # First token is the label; the remainder (after the first whitespace) is the quoted timestamp.
            label_str, _, ts_str = line.partition(" ")
            labels.append(int(label_str))
            timestamp_strings.append(ts_str.strip().strip('"').strip("'"))
    y = pd.Series(labels, name="label", dtype=int)
    timestamps = pd.to_datetime(
        pd.Series(timestamp_strings, name="timestamp"),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    n_missing_ts = timestamps.isna().sum()
    if n_missing_ts:
        logger.warning(f"{n_missing_ts} timestamps failed to parse; will fall back to row order")

    if len(X) != len(y):
        raise ValueError(f"Row count mismatch: X={len(X)} vs labels={len(y)}")

    n_fails = int((y == 1).sum())
    logger.info(
        f"Loaded SECOM: {len(X)} lots x {X.shape[1]} signals, {n_fails} fails "
        f"(ratio {(len(X) - n_fails) / max(n_fails, 1):.1f}:1)"
    )
    return SecomData(
        X=X,
        y=y,
        timestamps=timestamps,
        n_lots=len(X),
        n_signals=X.shape[1],
        n_fails=n_fails,
    )


def to_parquet(secom: SecomData, processed_path: Path, metadata_path: Path) -> None:
    """Persist the joined dataset as parquet plus a small metadata JSON."""
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df = secom.X.copy()
    df.insert(0, "timestamp", secom.timestamps.values)
    df.insert(1, "label", secom.y.values)
    df.to_parquet(processed_path, index=False)
    metadata = {
        "n_lots": secom.n_lots,
        "n_signals": secom.n_signals,
        "n_fails": secom.n_fails,
        "pass_fail_ratio": (secom.n_lots - secom.n_fails) / max(secom.n_fails, 1),
        "source": "UCI ML Repository - McCann and Johnston 2008",
        "license": "CC BY 4.0",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Wrote processed dataset to {processed_path}")


def load_processed(processed_path: Path) -> pd.DataFrame:
    """Load the parquet that to_parquet produced."""
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed parquet not found: {processed_path}. Run 'fabsentinel ingest' first."
        )
    return pd.read_parquet(processed_path)
