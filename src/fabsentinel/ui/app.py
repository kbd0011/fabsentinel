"""Streamlit UI for FabSentinel.

Run with: streamlit run src/fabsentinel/ui/app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fabsentinel.config import load_config
from fabsentinel.data.ingest import load_processed
from fabsentinel.llm.narrator import NarratorPayload, OpenAINarrator
from fabsentinel.rca.shap_attribution import fit as shap_fit
from fabsentinel.rca.shap_attribution import top_features_for_lot
from fabsentinel.spc.ewma import fit_and_detect as ewma_run
from fabsentinel.spc.multivariate import contribution_plot
from fabsentinel.spc.shewhart import fit_individuals
from fabsentinel.vm.features import prepare
from fabsentinel.vm.train import load as vm_load

st.set_page_config(page_title="FabSentinel", layout="wide", page_icon="📊")
st.title("FabSentinel — SPC + Virtual Metrology + Excursion RCA")
st.caption("Built on UCI SECOM. CC BY 4.0.")


@st.cache_resource
def get_cfg():
    return load_config("configs/config.yaml")


@st.cache_data
def get_data():
    cfg = get_cfg()
    cleaned_path = cfg.data.processed_path.with_name("cleaned.parquet")
    if cleaned_path.exists():
        df = pd.read_parquet(cleaned_path)
    else:
        df = load_processed(cfg.data.processed_path)
    return df


@st.cache_resource
def get_models():
    cfg = get_cfg()
    artifact = vm_load(cfg.vm.artifact_path) if cfg.vm.artifact_path.exists() else None
    mspc = joblib.load("artifacts/mspc_model.joblib") if Path("artifacts/mspc_model.joblib").exists() else None
    mstats = pd.read_csv("artifacts/multivariate_stats.csv", index_col=0) if Path("artifacts/multivariate_stats.csv").exists() else None
    return artifact, mspc, mstats


cfg = get_cfg()

# --- Sidebar ---
with st.sidebar:
    st.header("Pipeline status")
    pipeline_ok = True
    if not Path(cfg.data.processed_path).exists():
        st.error("Run `make ingest`")
        pipeline_ok = False
    else:
        st.success("Ingested")
    if not Path("artifacts/multivariate_stats.csv").exists():
        st.warning("Run `make multivariate`")
        pipeline_ok = False
    else:
        st.success("Multivariate SPC fitted")
    if not Path(cfg.vm.artifact_path).exists():
        st.warning("Run `make vm`")
        pipeline_ok = False
    else:
        st.success("VM trained")
    st.markdown("---")
    st.markdown("**Tip:** `make demo` runs all stages end-to-end.")

if not pipeline_ok:
    st.info("Some pipeline stages are not yet complete. Run `make demo` from the repo root.")
    st.stop()

df = get_data()
artifact, mspc, mstats = get_models()

# --- Tabs ---
tab_kpis, tab_univ, tab_mult, tab_lot = st.tabs(["KPIs", "Univariate SPC", "Multivariate SPC", "Lot Inspector"])

with tab_kpis:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Lots", len(df))
    n_fails = int((df["label"] == 1).sum())
    col2.metric("Fails", n_fails)
    col3.metric("Pass:Fail", f"{(len(df) - n_fails) / max(n_fails, 1):.1f}:1")
    col4.metric("VM PR-AUC", f"{artifact.metrics['test_pr_auc']:.3f}")
    col5.metric("VM MCC", f"{artifact.metrics['test_mcc']:.3f}")

    st.subheader("Multivariate excursion timeline")
    timeline = mstats.copy()
    if "timestamp" in df.columns:
        timeline["timestamp"] = df["timestamp"].values[: len(timeline)]
        fig = px.scatter(
            timeline.reset_index(),
            x="timestamp", y="t2",
            color="t2_ooc",
            color_discrete_map={True: "#c53030", False: "#4299e1"},
            labels={"t2": "Hotelling T²", "timestamp": "Date"},
        )
    else:
        fig = px.scatter(timeline.reset_index(), x="index", y="t2", color="t2_ooc")
    fig.add_hline(y=mspc.t2_ucl, line_dash="dash", line_color="red", annotation_text="T² UCL")
    st.plotly_chart(fig, use_container_width=True)

with tab_univ:
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    top_cols = df[signal_cols].var().sort_values(ascending=False).head(20).index.tolist()
    sig = st.selectbox("Signal", top_cols, index=0)
    arr = df[sig].values
    sh = fit_individuals(arr)
    ewma_state, ewma_mask = ewma_run(arr)
    fig = go.Figure()
    fig.add_scatter(y=arr, mode="lines+markers", name=sig, line={"color": "#4299e1"})
    fig.add_hline(y=sh.mean, line_dash="dash", line_color="green", annotation_text="mean")
    fig.add_hline(y=sh.ucl, line_dash="dash", line_color="red", annotation_text="UCL")
    fig.add_hline(y=sh.lcl, line_dash="dash", line_color="red", annotation_text="LCL")
    fig.add_scatter(y=ewma_state.z, mode="lines", name="EWMA Z", line={"color": "#dd6b20"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Mean={sh.mean:.3f}, sigma={sh.sigma:.3f}, OOC count (Shewhart) = {int((arr > sh.ucl).sum() + (arr < sh.lcl).sum())}")

with tab_mult:
    st.subheader("T² vs SPE")
    fig = px.scatter(
        mstats.reset_index(),
        x="spe", y="t2",
        color="t2_ooc",
        color_discrete_map={True: "#c53030", False: "#4299e1"},
        labels={"t2": "Hotelling T²", "spe": "SPE residual"},
    )
    fig.add_hline(y=mspc.t2_ucl, line_dash="dash", line_color="red")
    fig.add_vline(x=mspc.spe_ucl, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    top10 = mstats.nlargest(10, "t2").reset_index().rename(columns={"index": "lot_idx"})
    st.subheader("Top 10 T² lots")
    st.dataframe(top10, use_container_width=True)

with tab_lot:
    st.subheader("Inspect a single lot")
    default_idx = int(mstats["t2"].idxmax())
    lot_idx = int(st.number_input("Lot index", min_value=0, max_value=len(df) - 1, value=default_idx))
    row = mstats.loc[lot_idx]
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    X_full, _ = prepare(df, signal_cols)
    x_row = X_full.iloc[lot_idx]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("T²", f"{row['t2']:.2f}", delta=f"UCL {mspc.t2_ucl:.2f}")
    col2.metric("SPE", f"{row['spe']:.2f}", delta=f"UCL {mspc.spe_ucl:.2f}")
    proba = float(artifact.model.predict_proba(x_row.to_frame().T)[0, 1])
    col3.metric("VM fail prob", f"{proba:.3f}")
    label = int(df["label"].iloc[lot_idx])
    col4.metric("Actual", "FAIL" if label == 1 else "PASS")

    st.subheader("Top T² contributors")
    contrib = contribution_plot(mspc, x_row, top_k=10)
    fig = px.bar(contrib.reset_index(), x="contribution", y="index", orientation="h", labels={"index": "signal"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("SHAP local attribution")
    if "_shap_cache" not in st.session_state:
        with st.spinner("Building SHAP explainer..."):
            st.session_state["_shap_cache"] = shap_fit(artifact, X_full, n_background=cfg.rca.background_samples)
    shap_report = st.session_state["_shap_cache"]
    local = top_features_for_lot(shap_report, x_row, top_k=10)
    st.dataframe(local, use_container_width=True)

    if st.button("Draft AI excursion narrative (uses your OPENAI_API_KEY)"):
        nar = OpenAINarrator(cfg.llm)
        if not nar.available():
            st.warning("OPENAI_API_KEY is not set. Set it and refresh.")
        else:
            payload = NarratorPayload(
                lot_idx=lot_idx,
                timestamp=str(df["timestamp"].iloc[lot_idx]) if "timestamp" in df.columns else "n/a",
                t2=float(row["t2"]),
                spe=float(row["spe"]),
                vm_fail_proba=proba,
                actual_label=label,
                top_signals_t2=[(s, float(v)) for s, v in contrib.items()],
                top_signals_shap=[(r.feature, float(r.value), float(r.shap_value)) for r in local.itertuples()],
            )
            with st.spinner("Drafting narrative via OpenAI..."):
                text = nar.draft(payload)
            st.info("AI-DRAFTED — review before publishing.")
            st.write(text)
