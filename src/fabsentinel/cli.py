"""Typer-based command line interface for FabSentinel."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger

from fabsentinel.config import load_config
from fabsentinel.data.ingest import load_secom, to_parquet, load_processed
from fabsentinel.data.quality import profile, clean
from fabsentinel.spc.shewhart import fit_individuals, detect as shew_detect
from fabsentinel.spc.ewma import fit_and_detect as ewma_run
from fabsentinel.spc.cusum import fit_and_detect as cusum_run
from fabsentinel.spc.rules import western_electric, nelson
from fabsentinel.spc.multivariate import fit as mspc_fit, compute_statistics, contribution_plot
from fabsentinel.spc.capability import simulate_specs_and_estimate
from fabsentinel.vm.features import prepare
from fabsentinel.vm.train import train as vm_train, load as vm_load
from fabsentinel.rca.shap_attribution import fit as shap_fit, top_features_for_lot
from fabsentinel.reports.scorecard import ScorecardPayload, render as render_scorecard
from fabsentinel.reports.excursion_memo import ExcursionEvent, render as render_memo
from fabsentinel.llm.narrator import OpenAINarrator, NarratorPayload, get_narrator

app = typer.Typer(help="FabSentinel: SPC, virtual metrology, and excursion RCA for semiconductor manufacturing.")

CONFIG_PATH = Path("configs/config.yaml")


def _cfg() -> tuple:
    cfg = load_config(CONFIG_PATH)
    return cfg


@app.command()
def ingest() -> None:
    """Read SECOM raw files and write a processed parquet + metadata JSON."""
    cfg = _cfg()
    sec = load_secom(cfg.data.secom_data_path, cfg.data.secom_labels_path)
    to_parquet(sec, cfg.data.processed_path, cfg.data.metadata_path)
    typer.echo(f"Ingested {sec.n_lots} lots x {sec.n_signals} signals -> {cfg.data.processed_path}")


@app.command()
def quality() -> None:
    """Profile data quality and persist the cleaned signal list."""
    cfg = _cfg()
    df = load_processed(cfg.data.processed_path)
    report = profile(df, cfg.quality)
    cleaned = clean(df, report)
    cleaned_path = cfg.data.processed_path.with_name("cleaned.parquet")
    cleaned.to_parquet(cleaned_path, index=False)
    Path("artifacts/quality_report.json").parent.mkdir(parents=True, exist_ok=True)
    import json
    Path("artifacts/quality_report.json").write_text(json.dumps({
        "n_lots": report.n_lots,
        "n_signals": report.n_signals,
        "n_fails": report.n_fails,
        "pass_fail_ratio": report.pass_fail_ratio,
        "dropped_missing": report.signals_dropped_missing,
        "dropped_lowvar": report.signals_dropped_lowvar,
        "retained_count": len(report.cleaned_signal_columns),
        "imbalance_warning": report.pass_fail_imbalance_warning,
    }, indent=2))
    typer.echo(f"Quality OK; cleaned parquet -> {cleaned_path} ({cleaned.shape})")


@app.command()
def univariate(top_n: int = 10) -> None:
    """Run univariate SPC on the top-N highest-variance retained signals."""
    cfg = _cfg()
    cleaned_path = cfg.data.processed_path.with_name("cleaned.parquet")
    df = pd.read_parquet(cleaned_path)
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    top = df[signal_cols].var().sort_values(ascending=False).head(top_n).index.tolist()

    rows = []
    for sig in top:
        arr = df[sig].values
        sh = fit_individuals(arr, cfg.spc.shewhart.sigma)
        sh_mask = shew_detect(arr, sh)
        _, ewma_mask = ewma_run(arr, lam=cfg.spc.ewma.lambda_, L=cfg.spc.ewma.L)
        _, cusum_mask = cusum_run(arr, k=cfg.spc.cusum.k, h=cfg.spc.cusum.h)
        we_mask = western_electric(arr, sh.mean, sh.sigma, cfg.spc.rules.western_electric)
        nl_mask = nelson(arr, sh.mean, sh.sigma, cfg.spc.rules.nelson)
        rows.append({
            "signal": sig,
            "mean": sh.mean,
            "sigma": sh.sigma,
            "shewhart_ooc": int(sh_mask.sum()),
            "ewma_ooc": int(ewma_mask.sum()),
            "cusum_ooc": int(cusum_mask.sum()),
            "we_ooc": int(we_mask.sum()),
            "nelson_ooc": int(nl_mask.sum()),
        })
    out_path = Path("artifacts/univariate_ooc.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    typer.echo(f"Univariate SPC -> {out_path}")


@app.command()
def multivariate() -> None:
    """Fit PCA + Hotelling T^2 + SPE and persist the model + statistics."""
    cfg = _cfg()
    cleaned_path = cfg.data.processed_path.with_name("cleaned.parquet")
    df = pd.read_parquet(cleaned_path)
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    X = df[signal_cols]
    model = mspc_fit(X, cfg.spc.multivariate.pca_explained_variance, cfg.spc.multivariate.alpha)
    stats = compute_statistics(model, X)
    Path("artifacts/multivariate_stats.csv").parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv("artifacts/multivariate_stats.csv", index=True)
    joblib.dump(model, "artifacts/mspc_model.joblib")
    typer.echo(
        f"Multivariate SPC: PCA k={model.n_components}, T^2 UCL={model.t2_ucl:.2f}, "
        f"SPE UCL={model.spe_ucl:.2f}, T^2 OOC={stats['t2_ooc'].sum()}, SPE OOC={stats['spe_ooc'].sum()}"
    )


@app.command("train-vm")
def train_vm() -> None:
    """Train the LightGBM virtual metrology classifier."""
    cfg = _cfg()
    cleaned_path = cfg.data.processed_path.with_name("cleaned.parquet")
    df = pd.read_parquet(cleaned_path)
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    X, y = prepare(df, signal_cols)
    artifact = vm_train(X, y, cfg.vm)
    typer.echo(f"VM trained. Test PR-AUC={artifact.metrics['test_pr_auc']:.3f}, MCC={artifact.metrics['test_mcc']:.3f}")


@app.command()
def rca() -> None:
    """Compute SHAP attribution and persist the global ranking."""
    cfg = _cfg()
    cleaned_path = cfg.data.processed_path.with_name("cleaned.parquet")
    df = pd.read_parquet(cleaned_path)
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    X, _ = prepare(df, signal_cols)
    artifact = vm_load(cfg.vm.artifact_path)
    report = shap_fit(artifact, X, n_background=cfg.rca.background_samples)
    out_path = Path("artifacts/shap_global.csv")
    report.global_importance.head(50).to_csv(out_path)
    typer.echo(f"Wrote SHAP global ranking (top-50) -> {out_path}")


@app.command()
def scorecard(week: str | None = None) -> None:
    """Render the weekly SPC scorecard (HTML always, PDF if Quarto is installed)."""
    cfg = _cfg()
    cleaned_path = cfg.data.processed_path.with_name("cleaned.parquet")
    df = pd.read_parquet(cleaned_path)
    signal_cols = [c for c in df.columns if c.startswith("signal_")]

    # Pull artifacts produced by earlier stages
    mstats = pd.read_csv("artifacts/multivariate_stats.csv", index_col=0)
    univ = pd.read_csv("artifacts/univariate_ooc.csv")
    artifact = vm_load(cfg.vm.artifact_path)

    # Recompute SHAP global (cheap on 50 background samples)
    X, _ = prepare(df, signal_cols)
    shap_report = shap_fit(artifact, X, n_background=cfg.rca.background_samples)

    # Top T^2 lots
    top_t2 = mstats.nlargest(10, "t2").copy()
    top_t2["lot_idx"] = top_t2.index
    # Attach timestamp if available
    if "timestamp" in df.columns:
        top_t2 = top_t2.merge(
            df[["timestamp"]].reset_index().rename(columns={"index": "lot_idx"}),
            on="lot_idx",
            how="left",
        )
    top_t2 = top_t2[["lot_idx", "timestamp" if "timestamp" in top_t2.columns else "lot_idx", "t2", "spe"]].head(10)

    # Top OOC signals
    top_ooc = univ.sort_values("nelson_ooc", ascending=False).head(10)[
        ["signal", "shewhart_ooc", "ewma_ooc", "cusum_ooc", "we_ooc", "nelson_ooc"]
    ]

    # Capability
    cap_signals = [f"signal_{i:03d}" for i in cfg.capability.simulated_subset.signal_columns]
    cap_signals = [s for s in cap_signals if s in df.columns]
    cap = simulate_specs_and_estimate(df, cap_signals, cfg.capability.simulated_subset.sigma_margin)

    payload = ScorecardPayload(
        week_label=week or datetime.now().strftime("Week of %Y-%m-%d"),
        n_lots=len(df),
        n_fails=int((df["label"] == 1).sum()),
        pass_fail_ratio=(len(df) - int((df["label"] == 1).sum())) / max(int((df["label"] == 1).sum()), 1),
        top_t2_lots=top_t2,
        top_ooc_signals=top_ooc,
        capability_table=cap[["signal", "cp", "cpk", "mean", "sigma"]],
        vm_metrics=artifact.metrics,
        shap_top_global=shap_report.global_importance.head(cfg.reports.top_n_signals),
        notes=(
            "Cp/Cpk demonstrated on signals with simulated specification limits. "
            "SECOM is anonymized and does not include real engineering specs."
        ),
    )
    path = render_scorecard(payload, cfg.reports.template_dir, cfg.reports.scorecard_dir)
    typer.echo(f"Scorecard -> {path}")


@app.command()
def memo(lot_idx: int | None = None, week: str = "current", use_llm: bool = True) -> None:
    """Generate an 8D excursion memo for the worst T^2 lot (or a specific lot)."""
    cfg = _cfg()
    cleaned_path = cfg.data.processed_path.with_name("cleaned.parquet")
    df = pd.read_parquet(cleaned_path)
    signal_cols = [c for c in df.columns if c.startswith("signal_")]

    mstats = pd.read_csv("artifacts/multivariate_stats.csv", index_col=0)
    artifact = vm_load(cfg.vm.artifact_path)
    mspc_model = joblib.load("artifacts/mspc_model.joblib")
    X, _ = prepare(df, signal_cols)
    shap_report = shap_fit(artifact, X, n_background=cfg.rca.background_samples)

    if lot_idx is None:
        lot_idx = int(mstats["t2"].idxmax())
    lot_stats = mstats.loc[lot_idx]
    x_row = X.iloc[lot_idx] if lot_idx < len(X) else X.loc[lot_idx]

    # Contributions
    contrib = contribution_plot(mspc_model, x_row, top_k=cfg.reports.top_n_signals)

    # SHAP local
    local = top_features_for_lot(shap_report, x_row, top_k=cfg.reports.top_n_signals)

    # VM prediction
    proba = float(artifact.model.predict_proba(x_row.to_frame().T)[0, 1])

    # Optional LLM narrative
    narrative = None
    if use_llm and cfg.llm.enabled:
        nar = OpenAINarrator(cfg.llm)
        payload = NarratorPayload(
            lot_idx=lot_idx,
            timestamp=str(df["timestamp"].iloc[lot_idx]) if "timestamp" in df.columns else "n/a",
            t2=float(lot_stats["t2"]),
            spe=float(lot_stats["spe"]),
            vm_fail_proba=proba,
            actual_label=int(df["label"].iloc[lot_idx]) if "label" in df.columns else None,
            top_signals_t2=[(s, float(v)) for s, v in contrib.items()],
            top_signals_shap=[(r.feature, float(r.value), float(r.shap_value)) for r in local.itertuples()],
        )
        narrative = nar.draft(payload)

    event = ExcursionEvent(
        lot_idx=lot_idx,
        timestamp=df["timestamp"].iloc[lot_idx] if "timestamp" in df.columns else None,
        t2=float(lot_stats["t2"]),
        spe=float(lot_stats["spe"]),
        predicted_fail_proba=proba,
        actual_label=int(df["label"].iloc[lot_idx]) if "label" in df.columns else None,
        top_t2_contributors=contrib,
        shap_top=local[["feature", "value", "shap_value"]],
        d1_team=["Process Engineer (lead)", "Quality Engineer", "Tool Owner"],
        d2_problem=(
            f"Hotelling T^2 = {lot_stats['t2']:.2f} (UCL {mspc_model.t2_ucl:.2f}). "
            f"VM fail probability = {proba:.3f}. "
            "Multivariate excursion flagged by FabSentinel pipeline."
        ),
        d3_containment="Hold lot pending review. Quarantine downstream lots from this tool family.",
        d4_root_cause_candidates=[
            f"Signal {row.feature}: SHAP {row.shap_value:+.4f} (value {row.value:+.3f})"
            for row in local.head(5).itertuples()
        ],
        d5_corrective_action="To be filled by process engineer after review.",
        d6_implement="To be filled.",
        d7_prevent="To be filled.",
        d8_recognize="To be filled.",
        llm_narrative=narrative,
    )
    path = render_memo(event, cfg.reports.template_dir, cfg.reports.memo_dir)
    typer.echo(f"Excursion memo -> {path}")


if __name__ == "__main__":
    app()
