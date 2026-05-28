"""Tests for LLM narrator (provider boundary, no real API call)."""

from fabsentinel.config import LLMCfg
from fabsentinel.llm.narrator import NarratorPayload, OpenAINarrator, get_narrator


def _payload():
    return NarratorPayload(
        lot_idx=42,
        timestamp="2024-01-15T08:00:00",
        t2=210.0,
        spe=12.3,
        vm_fail_proba=0.87,
        actual_label=1,
        top_signals_t2=[("signal_033", 5.0), ("signal_059", 4.2)],
        top_signals_shap=[("signal_033", 3.1, 0.42), ("signal_059", -2.4, 0.31)],
    )


def test_narrator_disabled_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = LLMCfg(api_key_env="OPENAI_API_KEY")
    nar = OpenAINarrator(cfg)
    assert not nar.available()
    text = nar.draft(_payload())
    assert "disabled" in text.lower() or "not set" in text.lower()


def test_get_narrator_returns_none_when_disabled():
    cfg = LLMCfg(enabled=False)
    assert get_narrator(cfg) is None


def test_get_narrator_returns_openai_when_enabled(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = LLMCfg(enabled=True, provider="openai")
    nar = get_narrator(cfg)
    assert isinstance(nar, OpenAINarrator)


def test_unknown_provider_raises():
    import pytest
    cfg = LLMCfg(enabled=True, provider="unknown")
    with pytest.raises(ValueError):
        get_narrator(cfg)
