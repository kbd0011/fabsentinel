"""LLM-drafted excursion narrative (OpenAI default, swap-able).

Usage:
    nar = OpenAINarrator(model="gpt-4o-mini")
    text = nar.draft(payload)

The narrator NEVER classifies. It rewrites deterministic SPC/SHAP output
into engineer-readable prose. The UI labels the output as a suggestion.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from fabsentinel.config import LLMCfg


@dataclass
class NarratorPayload:
    lot_idx: int
    timestamp: str
    t2: float
    spe: float
    vm_fail_proba: float
    actual_label: int | None
    top_signals_t2: list[tuple[str, float]]      # (signal, contribution)
    top_signals_shap: list[tuple[str, float, float]]  # (signal, value, shap)


class Narrator(Protocol):
    def draft(self, payload: NarratorPayload) -> str: ...


_SYSTEM_PROMPT = """You are an experienced semiconductor manufacturing engineer drafting an
8D-style excursion memo. Given the SPC and SHAP attribution output below, write a brief,
factual narrative (3-5 short paragraphs) suitable for a fab quality review meeting.

Hard rules:
- Cite specific signal IDs and numeric values from the input.
- Do not invent process steps, tool names, or root causes that are not implied by the data.
- Use cautious language: "candidate root cause", "consistent with", "warrants investigation".
- Do not state a final conclusion. The human engineer will own that.
- Use plain English at the level of a process engineer reading the memo at the start of a shift.
"""


def _build_user_prompt(p: NarratorPayload) -> str:
    lines = [
        f"Lot index: {p.lot_idx}",
        f"Timestamp: {p.timestamp}",
        f"Hotelling T2: {p.t2:.3f}",
        f"SPE residual: {p.spe:.3f}",
        f"Virtual metrology fail probability: {p.vm_fail_proba:.4f}",
        f"Actual label (if known): {p.actual_label}",
        "",
        "Top T2 contributors:",
    ]
    for sig, c in p.top_signals_t2:
        lines.append(f"  - {sig}: contribution {c:+.3f}")
    lines.append("")
    lines.append("Top SHAP-attributed features (sign indicates push toward fail):")
    for sig, val, shap in p.top_signals_shap:
        lines.append(f"  - {sig}: value {val:+.3f}, SHAP {shap:+.4f}")
    lines.append("")
    lines.append(
        "Draft the narrative sections D2 (Problem Description) and D4 (Candidate Root Causes) only. "
        "Keep total length under 300 words."
    )
    return "\n".join(lines)


class OpenAINarrator:
    """OpenAI Chat Completions narrator."""

    def __init__(self, cfg: LLMCfg) -> None:
        self.cfg = cfg
        self.api_key = os.environ.get(cfg.api_key_env)
        if not self.api_key:
            logger.warning(
                f"{cfg.api_key_env} not set. LLM narrator is disabled. "
                "Memo will use the deterministic SHAP/contribution text only."
            )
            self._client = None
        else:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"OpenAI client init failed: {e}")
                self._client = None

    def available(self) -> bool:
        return self._client is not None

    def draft(self, payload: NarratorPayload) -> str:
        if not self._client:
            return (
                "[LLM narrator disabled — OPENAI_API_KEY not set. "
                "The deterministic SHAP attribution above is the authoritative output.]"
            )
        user_prompt = _build_user_prompt(payload)
        logger.info(f"Requesting LLM narrative for lot {payload.lot_idx} via {self.cfg.model}")
        try:
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            )
            return resp.choices[0].message.content or "[empty LLM response]"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[LLM narrative failed: {e!s}]"


def get_narrator(cfg: LLMCfg) -> Narrator | None:
    """Factory returning a narrator implementation, or None if disabled."""
    if not cfg.enabled:
        return None
    if cfg.provider == "openai":
        return OpenAINarrator(cfg)
    raise ValueError(f"Unknown LLM provider: {cfg.provider}")
