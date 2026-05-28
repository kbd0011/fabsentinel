"""Light tests for report rendering (HTML path; Quarto PDF tested manually)."""
from pathlib import Path

import pandas as pd

from fabsentinel.reports.excursion_memo import ExcursionEvent
from fabsentinel.reports.excursion_memo import render as render_memo
from fabsentinel.reports.scorecard import ScorecardPayload, quarto_available
from fabsentinel.reports.scorecard import render as render_scorecard

TEMPLATE_DIR = Path("src/fabsentinel/reports/templates")


def test_scorecard_html_renders(tmp_path):
    payload = ScorecardPayload(
        week_label="Week of 2024-01-01",
        n_lots=1567,
        n_fails=104,
        pass_fail_ratio=14.1,
        top_t2_lots=pd.DataFrame({"lot_idx": [1, 2], "t2": [200, 180], "spe": [10, 9]}),
        top_ooc_signals=pd.DataFrame({"signal": ["signal_001"], "shewhart_ooc": [5]}),
        capability_table=pd.DataFrame({"signal": ["signal_000"], "cp": [1.33], "cpk": [1.25], "mean": [10.0], "sigma": [1.0]}),
        vm_metrics={"test_pr_auc": 0.18, "test_mcc": 0.32},
        shap_top_global=pd.Series([0.6, 0.5], index=["signal_033", "signal_059"]),
    )
    path = render_scorecard(payload, TEMPLATE_DIR, tmp_path)
    assert path.exists()
    assert path.suffix in {".html", ".pdf"}
    text = path.read_text(errors="ignore") if path.suffix == ".html" else ""
    if text:
        assert "FabSentinel" in text
        assert "1567" in text


def test_memo_html_renders(tmp_path):
    event = ExcursionEvent(
        lot_idx=42,
        timestamp=None,
        t2=210.0,
        spe=12.3,
        predicted_fail_proba=0.87,
        actual_label=1,
        top_t2_contributors=pd.Series({"signal_033": 5.0, "signal_059": 4.2}),
        shap_top=pd.DataFrame({
            "feature": ["signal_033", "signal_059"],
            "value": [3.1, -2.4],
            "shap_value": [0.42, 0.31],
        }),
        d1_team=["a", "b"],
        d2_problem="T^2 exceeded UCL.",
        d3_containment="Hold lot",
        d4_root_cause_candidates=["s33"],
        d5_corrective_action="TBD",
        d6_implement="TBD",
        d7_prevent="TBD",
        d8_recognize="TBD",
    )
    path = render_memo(event, TEMPLATE_DIR, tmp_path)
    assert path.exists()


def test_quarto_available_is_bool():
    assert isinstance(quarto_available(), bool)
