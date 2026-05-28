"""Pydantic configuration for FabSentinel."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    secom_data_path: Path
    secom_labels_path: Path
    processed_path: Path
    metadata_path: Path


class QualityCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    missing_threshold: float = Field(default=0.55, ge=0, le=1)
    variance_threshold: float = Field(default=1e-6, ge=0)
    outlier_zscore: float = Field(default=4.0, gt=0)
    imbalance_warn_ratio: float = Field(default=10.0, gt=0)


class ShewhartCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sigma: float = 3.0
    subgroup_size: int = 1


class EWMACfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lambda_: float = Field(default=0.2, alias="lambda", ge=0, le=1)
    L: float = 3.0


class CUSUMCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    k: float = 0.5
    h: float = 5.0


class RulesCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    western_electric: list[int]
    nelson: list[int]


class MultivariateCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pca_explained_variance: float = Field(default=0.90, gt=0, lt=1)
    alpha: float = Field(default=0.01, gt=0, lt=1)


class SPCCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shewhart: ShewhartCfg
    ewma: EWMACfg
    cusum: CUSUMCfg
    rules: RulesCfg
    multivariate: MultivariateCfg


class VMModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    class_weight: str = "balanced"


class VMCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    smote_strategy: float = 0.30
    model: VMModelCfg
    threshold_strategy: str = "pr_max_f1"
    artifact_path: Path


class RCACfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shap_top_k: int = 10
    background_samples: int = 100


class CapabilitySubsetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signal_columns: list[int]
    sigma_margin: float = 4.0


class CapabilityCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    simulated_subset: CapabilitySubsetCfg


class ReportsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scorecard_dir: Path
    memo_dir: Path
    template_dir: Path
    top_n_signals: int = 10


class LLMCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 600
    temperature: float = 0.2
    api_key_env: str = "OPENAI_API_KEY"


class FabSentinelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    data: DataCfg
    quality: QualityCfg
    spc: SPCCfg
    vm: VMCfg
    rca: RCACfg
    capability: CapabilityCfg
    reports: ReportsCfg
    llm: LLMCfg


def load_config(path: str | Path = "configs/config.yaml") -> FabSentinelConfig:
    """Load and validate the YAML config."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open() as fh:
        raw = yaml.safe_load(fh)
    return FabSentinelConfig.model_validate(raw)
