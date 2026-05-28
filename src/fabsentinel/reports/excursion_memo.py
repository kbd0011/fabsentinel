"""Generate an 8D-style excursion memo for a single OOC event."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger


@dataclass
class ExcursionEvent:
    """A single OOC excursion."""
    lot_idx: int
    timestamp: pd.Timestamp | None
    t2: float
    spe: float
    predicted_fail_proba: float
    actual_label: int | None
    top_t2_contributors: pd.Series              # signal -> contribution
    shap_top: pd.DataFrame                       # feature, value, shap_value
    d1_team: list[str]                            # 8D D1: members
    d2_problem: str                               # 8D D2: problem narrative
    d3_containment: str                           # 8D D3: containment
    d4_root_cause_candidates: list[str]           # 8D D4: candidate root causes
    d5_corrective_action: str
    d6_implement: str
    d7_prevent: str
    d8_recognize: str
    llm_narrative: str | None = None              # optional drafted narrative


def render(
    event: ExcursionEvent,
    template_dir: Path,
    output_dir: Path,
) -> Path:
    """Render an 8D memo.  Returns the path to the rendered artifact (PDF if Quarto is available, else HTML)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"memo_lot_{event.lot_idx}_{timestamp}"

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )

    html_tmpl = env.get_template("memo.html.j2")
    html_content = html_tmpl.render(
        event=event,
        top_contrib=event.top_t2_contributors.to_frame("contribution").to_html(classes="table"),
        shap_table=event.shap_top.to_html(index=False, classes="table"),
        generated=datetime.now().isoformat(timespec="seconds"),
    )
    html_path = base.with_suffix(".html")
    html_path.write_text(html_content)
    logger.info(f"Wrote excursion memo HTML to {html_path}")

    if shutil.which("quarto"):
        qmd_tmpl = env.get_template("memo.qmd.j2")
        qmd_content = qmd_tmpl.render(event=event)
        qmd_path = base.with_suffix(".qmd")
        qmd_path.write_text(qmd_content)
        try:
            subprocess.run(
                ["quarto", "render", str(qmd_path), "--to", "pdf"],
                check=True, capture_output=True, text=True, timeout=120,
            )
            pdf_path = base.with_suffix(".pdf")
            if pdf_path.exists():
                logger.info(f"Wrote excursion memo PDF to {pdf_path}")
                return pdf_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Quarto PDF render failed; returning HTML. {e!s}")
    return html_path
