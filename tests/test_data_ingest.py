"""Ingest tests - run against the bundled SECOM (small, ~5MB)."""

import pytest

from fabsentinel.config import load_config
from fabsentinel.data.ingest import load_processed, load_secom, to_parquet


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/config.yaml")


def test_load_secom_shape(cfg):
    sec = load_secom(cfg.data.secom_data_path, cfg.data.secom_labels_path)
    assert sec.n_lots == 1567
    assert sec.n_signals == 590
    assert sec.n_fails == 104
    # Pass/fail counts are documented in the SECOM dataset card
    assert (sec.y == -1).sum() == 1463
    assert (sec.y == 1).sum() == 104


def test_load_secom_timestamps_parse(cfg):
    sec = load_secom(cfg.data.secom_data_path, cfg.data.secom_labels_path)
    # All timestamps in SECOM should parse cleanly
    assert sec.timestamps.isna().sum() == 0
    # Span the known SECOM range (2008)
    assert sec.timestamps.dt.year.min() >= 2008
    assert sec.timestamps.dt.year.max() <= 2008


def test_to_parquet_and_reload(cfg, tmp_path):
    sec = load_secom(cfg.data.secom_data_path, cfg.data.secom_labels_path)
    out = tmp_path / "p.parquet"
    meta = tmp_path / "m.json"
    to_parquet(sec, out, meta)
    assert out.exists()
    assert meta.exists()
    df = load_processed(out)
    assert df.shape[0] == sec.n_lots
    assert "label" in df.columns
    assert "timestamp" in df.columns


def test_load_processed_missing_raises(tmp_path):
    from fabsentinel.data.ingest import load_processed
    with pytest.raises(FileNotFoundError):
        load_processed(tmp_path / "missing.parquet")
