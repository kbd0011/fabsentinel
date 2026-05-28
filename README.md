# FabSentinel

**Statistical Process Control + Virtual Metrology + 8D Excursion RCA for semiconductor manufacturing.**
Production-grade reference pipeline on the UCI SECOM dataset (1,567 wafer lots × 590 sensor signals, 104 fails, 14:1 imbalance).

[![CI](https://github.com/kbd0011/fabsentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/kbd0011/fabsentinel/actions)
![python](https://img.shields.io/badge/python-3.12-blue.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)

---

## What this is

FabSentinel is a small, opinionated end-to-end SPC stack of the kind a semiconductor process engineer would actually use on a production line. It takes raw fab sensor traces, profiles them for data quality, runs both **univariate** and **multivariate** statistical process control, trains a **virtual metrology** classifier to predict end-of-line failure, attributes excursions to specific signals with **SHAP**, and emits a **weekly SPC scorecard** plus an **8D-style excursion memo** for each out-of-control lot.

What sets it apart from a notebook:

- **Pluggable provider for the LLM excursion narrator** — uses OpenAI by default, configured via `OPENAI_API_KEY`. The LLM is an editorial layer that rewrites the deterministic SHAP/contribution output into prose; it never classifies, and every output is clearly labeled as a draft.
- **Honest about anonymized data.** SECOM has no engineering specification limits, so Cp / Cpk are computed only on a clearly-labeled simulated-spec subset. The methodology is correct; the specific Cp values are illustrative.
- **Imbalance-correct metrics.** Reports PR-AUC, MCC, balanced accuracy, recall. Never plain accuracy. SMOTE on training only. PR-curve max-F1 threshold tuning, not 0.5.
- **Reproducible.** uv-managed environment, locked dependencies, deterministic seeds, MLflow tracking, single `make demo` rebuilds everything from scratch.

## Quickstart (macOS, Apple Silicon)

```bash
# One-time setup
brew install --cask quarto      # optional but enables PDF rendering of the scorecard / memo

# Install + run the full pipeline + generate the scorecard
make install
make demo

# Launch the Streamlit UI on http://localhost:8501
make ui
```

`make demo` ingests SECOM, profiles data quality, fits univariate and multivariate SPC, trains the LightGBM virtual metrology classifier, computes SHAP attribution, and writes the scorecard HTML to `artifacts/scorecards/`. If `OPENAI_API_KEY` is set in your environment, `make memo` and the Streamlit "Lot Inspector" tab will also produce an LLM-drafted 8D narrative (clearly labeled as a draft).

## Pipeline at a glance

```
┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  SECOM     │──▶│ Data quality │──▶│ Univariate   │──▶│ Multivariate │
│  (bundled) │   │ + cleaning   │   │ SPC          │   │ SPC (PCA-T²) │
└────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                              │
                              ┌───────────────────────────────┘
                              ▼
                       ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
                       │ Virtual     │──▶│ SHAP        │──▶│ Scorecard +  │
                       │ Metrology   │   │ attribution │   │ 8D memo +    │
                       │ (LightGBM)  │   │ (global +   │   │ Streamlit UI │
                       └─────────────┘   │  per-lot)   │   └──────────────┘
                                         └─────────────┘
```

## What each stage produces

| Stage | Command | Output |
|---|---|---|
| Ingest | `make ingest` | `data/processed.parquet`, `data/metadata.json` |
| Quality | `make quality` | `data/cleaned.parquet`, `artifacts/quality_report.json` |
| Univariate SPC | `make univariate` | `artifacts/univariate_ooc.csv` |
| Multivariate SPC | `make multivariate` | `artifacts/multivariate_stats.csv`, `artifacts/mspc_model.joblib` |
| Virtual metrology | `make vm` | `artifacts/vm_model.joblib`, `mlruns/` (MLflow) |
| RCA | `make rca` | `artifacts/shap_global.csv` |
| Scorecard | `make scorecard` | `artifacts/scorecards/scorecard_*.html` (+ `.pdf` if Quarto) |
| Memo | `make memo` | `artifacts/memos/memo_lot_*.html` (+ `.pdf` if Quarto) |
| UI | `make ui` | Streamlit app on `http://localhost:8501` |

## Methods

| Method | Where | Reference |
|---|---|---|
| Shewhart individuals (moving range, d₂=1.128) | `spc/shewhart.py` | Montgomery, *Introduction to Statistical Quality Control*, 7e |
| EWMA with time-varying limits | `spc/ewma.py` | Roberts (1959); Lucas & Saccucci (1990) |
| Tabular CUSUM | `spc/cusum.py` | Page (1954); Montgomery, ch. 9 |
| Western Electric rules 1–4 | `spc/rules.py` | AT&T Statistical Quality Control Handbook (1956) |
| Nelson rules 1–8 | `spc/rules.py` | Nelson, *J. Quality Tech.* 16 (1984) |
| PCA + Hotelling T² UCL (F-distribution) | `spc/multivariate.py` | Jackson, *A User's Guide to PCA* (1991) |
| SPE / Q residual + Jackson–Mudholkar UCL | `spc/multivariate.py` | Jackson & Mudholkar (1979) |
| T² contribution decomposition | `spc/multivariate.py` | Westerhuis et al., *J. Chemometrics* (2000) |
| Cp / Cpk / Cpm | `spc/capability.py` | Kane (1986); Chan, Cheng & Spiring (1988) |
| LightGBM gradient boosting | `vm/train.py` | Ke et al., NeurIPS (2017) |
| SMOTE oversampling | `vm/train.py` | Chawla et al., *JAIR* (2002) |
| SHAP TreeExplainer | `rca/shap_attribution.py` | Lundberg et al., *Nat. Mach. Intell.* (2020) |

## Benchmarks (held-out test set, 314 lots, 21 fails)

| Metric | Value |
|---|---|
| PR-AUC | 0.184 |
| ROC-AUC | 0.717 |
| MCC | 0.318 |
| Balanced accuracy | 0.648 |
| Recall | 0.333 |
| F1 (PR-max threshold) | 0.359 |

These numbers reflect SECOM's deliberate difficulty: 21 minority-class lots in a 314-lot test set make this a hard classification problem. The published SECOM literature is fairly consistent in the PR-AUC 0.15–0.25 range; PR-AUC and MCC are the metrics to compare on, not accuracy.

## Deployment

- **Local** — `make ui` runs Streamlit on `http://localhost:8501`.
- **Docker** — `docker build -t fabsentinel . && docker run -p 8501:8501 fabsentinel`.
- **Hugging Face Spaces** — `huggingface/spaces/app.py` and `huggingface/spaces/README.md` are configured for one-click deploy. Push to a Space and add `OPENAI_API_KEY` as a Space secret.

## Configuration

All knobs live in `configs/config.yaml`. Edit the YAML, restart the pipeline, done. The Pydantic schema in `src/fabsentinel/config.py` enforces strict validation — extra fields are rejected.

| Knob | Default | Where |
|---|---|---|
| Shewhart sigma multiplier | 3.0 | `spc.shewhart.sigma` |
| EWMA λ | 0.2 | `spc.ewma.lambda` |
| CUSUM k, h | 0.5, 5.0 | `spc.cusum.{k,h}` |
| PCA explained variance | 0.90 | `spc.multivariate.pca_explained_variance` |
| T² / SPE α | 0.01 | `spc.multivariate.alpha` |
| LightGBM trees | 400 | `vm.model.n_estimators` |
| SMOTE minority ratio | 0.30 | `vm.smote_strategy` |
| LLM model | gpt-4o-mini | `llm.model` |

## What this is not

- This is not a real fab MES. SECOM is anonymized historical data from 2008.
- It is not a substitute for a real engineering investigation. The LLM narrator drafts text; humans own the diagnosis.
- It is not benchmarking research. We do not claim novel methods or SOTA performance.

## License & attribution

Code: MIT.  
SECOM dataset: CC BY 4.0, McCann & Johnston, UCI ML Repository, redistributed in `data/`.

## References

See [`docs/references.md`](docs/references.md) for the full bibliography.
