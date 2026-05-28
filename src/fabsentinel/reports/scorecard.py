"""Generate the weekly SPC scorecard PDF (or HTML if Quarto is unavailable)."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger


@dataclass
class ScorecardPayload:
    week_label: str
    n_lots: int
    n_fails: int
    pass_fail_ratio: float
    top_t2_lots: pd.DataFrame                    # columns: timestamp, lot_idx, t2, spe
    top_ooc_signals: pd.DataFrame                # columns: signal, ooc_count
    capability_table: pd.DataFrame               # columns: signal, cp, cpk
    vm_metrics: dict
    shap_top_global: pd.Series
    notes: str = ""


def quarto_available() -> bool:
    return shutil.which("quarto") is not None


def render(payload: ScorecardPayload, template_dir: Path, output_dir: Path) -> Path:
    """Render the scorecard.  Tries PDF via Quarto first, falls back to HTML.

    Returns the path to the rendered artifact.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"scorecard_{timestamp}"

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )

    # Always render HTML (cheap, no system deps).
    html_tmpl = env.get_template("scorecard.html.j2")
    html_content = html_tmpl.render(
        payload=payload,
        top_t2=payload.top_t2_lots.to_html(index=False, classes="table"),
        top_ooc=payload.top_ooc_signals.to_html(index=False, classes="table"),
        capability=payload.capability_table.to_html(index=False, classes="table"),
        shap_top=payload.shap_top_global.to_frame("mean_abs_shap").to_html(classes="table"),
        generated=datetime.now().isoformat(timespec="seconds"),
    )
    html_path = base.with_suffix(".html")
    html_path.write_text(html_content)
    logger.info(f"Wrote scorecard HTML to {html_path}")

    # Try Quarto for PDF
    if quarto_available():
        qmd_tmpl = env.get_template("scorecard.qmd.j2")
        qmd_content = qmd_tmpl.render(payload=payload, generated=html_path.name)
        qmd_path = base.with_suffix(".qmd")
        qmd_path.write_text(qmd_content)
        try:
            subprocess.run(
                ["quarto", "render", str(qmd_path), "--to", "pdf"],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            pdf_path = base.with_suffix(".pdf")
            if pdf_path.exists():
                logger.info(f"Wrote scorecard PDF to {pdf_path}")
                return pdf_path
        except subprocess.CalledProcessError as e:
            logger.warning(f"Quarto PDF render failed; returning HTML only. stderr: {e.stderr[:300]}")
        except subprocess.TimeoutExpired:
            logger.warning("Quarto PDF render timed out; returning HTML only.")
    else:
        logger.info("Quarto not installed; HTML only. Install with: brew install --cask quarto")

    return html_path
