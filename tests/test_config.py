"""Tests for the YAML config loader."""
from pathlib import Path

import pytest
import yaml

from fabsentinel.config import (
    FabSentinelConfig,
    load_config,
)


def test_load_default_config_succeeds():
    cfg = load_config("configs/config.yaml")
    assert isinstance(cfg, FabSentinelConfig)
    assert cfg.data.secom_data_path.name == "secom.data"
    assert cfg.spc.shewhart.sigma == 3.0
    assert cfg.spc.multivariate.pca_explained_variance > 0
    assert cfg.llm.provider == "openai"


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_config_rejects_extra_fields(tmp_path):
    bad = yaml.safe_load(Path("configs/config.yaml").read_text())
    bad["bogus"] = True
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(Exception):
        load_config(p)
