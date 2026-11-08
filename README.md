# FabSentinel

**Statistical process control, virtual metrology and 8D-style excursion reports for semiconductor manufacturing.**
End-to-end pipeline on the public UCI SECOM dataset: anonymized sensor data from a real fab, 1,567 production runs × 590 signals, 104 failures (about 14:1 pass:fail).

[![CI](https://github.com/kbd0011/fabsentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/kbd0011/fabsentinel/actions)
![python](https://img.shields.io/badge/python-3.12-blue.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)
[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-Hugging%20Face%20Space-ffce1c.svg)](https://huggingface.co/spaces/kbdev0011/fabsentinel)

▶ **Live demo:** https://huggingface.co/spaces/kbdev0011/fabsentinel

---

## What this is

FabSentinel is a compact end-to-end SPC pipeline on real fab sensor data. It screens and cleans the signals, runs **univariate** and **multivariate** statistical process control, trains a **virtual metrology** classifier to predict end-of-line failure, explains its predictions with **SHAP**, and renders an **SPC scorecard** plus an **8D-style excursion memo** for a flagged production run. (The code calls each SECOM row a "lot".)

Design choices:

- **Imbalance-aware metrics.** Reports PR-AUC, ROC-AUC, MCC, balanced accuracy and recall, never plain accuracy: a model that always predicts "pass" scores 93% on SECOM and catches no failures.
- **Threshold frozen before testing.** Class weighting (`scale_pos_weight = n_neg / n_pos`, about 14) handles the imbalance without resampling. The decision threshold maximizes F1 on pooled out-of-fold predictions from 5-fold CV and is applied unchanged to the held-out test set.
- **Capability on simulated specs only.** SECOM has no engineering specification limits, so Cp / Cpk are computed on a small set of signals with simulated limits (mean ± 4σ) and labeled that way in the scorecard.
- **LLM as editor, not classifier.** An optional OpenAI narrator (`gpt-4o-mini`, behind a small `Narrator` protocol) rewrites the deterministic T² and SHAP output into a draft D2/D4 narrative. It never classifies, the memo and UI label its text as an AI draft, and without `OPENAI_API_KEY` the memo uses the deterministic output only.
- **Reproducible.** uv-managed Python 3.12 environment, fixed seeds (`random_state=42`), Pydantic-validated config, MLflow run tracking, and one `make demo` that runs every stage through the scorecard.

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

`make demo` ingests SECOM, profiles data quality, fits univariate and multivariate SPC, trains the LightGBM virtual metrology classifier, computes SHAP attribution, and writes the scorecard HTML to `artifacts/scorecards/`. `make memo` writes an 8D memo for the run with the highest T²; with `OPENAI_API_KEY` set, the memo and the Streamlit "Lot Inspector" tab also include an LLM-drafted narrative, labeled as a draft.

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

## Pipeline details

- **Data quality.** Drops 24 signals with more than 55% missing values and 126 near-constant signals, keeping 440, then median-imputes the remaining gaps (`artifacts/quality_report.json`). Cleaning runs once on all 1,567 runs, before the train/test split, and uses no labels.
- **Univariate SPC.** Shewhart individuals, EWMA and CUSUM charts, with Western Electric and Nelson run rules, on the 20 highest-variance retained signals (`artifacts/univariate_ooc.csv`; `fabsentinel univariate --top-n` sets the count).
- **Multivariate SPC.** PCA keeping 90% of variance, with Hotelling T² and SPE limits at α = 0.01, across all 440 retained signals (`artifacts/multivariate_stats.csv`).
- **Virtual metrology.** LightGBM on the 440 signals; stratified 80/20 split (1,253 training runs, 314 test runs) with 5-fold CV inside the training set.
- **Root-cause support.** SHAP TreeExplainer explains the trained model, globally and per run. The 8D memo lists the run's top Hotelling T² contributors, and ranks its candidate root causes by that run's SHAP values only.

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
| Cost-sensitive learning (`scale_pos_weight`) | `vm/train.py` | King & Zeng, *Political Analysis* (2001); He & Garcia, *IEEE TKDE* (2009) |
| SHAP TreeExplainer | `rca/shap_attribution.py` | Lundberg et al., *Nat. Mach. Intell.* (2020) |

## Results (held-out test set: 314 production runs, 21 failures, 6.7% base rate)

| Metric | Value |
|---|---|
| PR-AUC | 0.169 |
| ROC-AUC | 0.730 |
| MCC | 0.183 |
| Balanced accuracy | 0.592 |
| Recall | 0.238 |
| Precision | 0.238 |
| F1 (frozen threshold) | 0.238 |

Test confusion matrix (314 runs, 21 failures): **TN 277, FP 16, FN 16, TP 5**.

A random ranking's expected PR-AUC equals the base rate, 0.067; the model's 0.169 is about 2.5 times that. With only 21 failures in the test set, PR-AUC and MCC are the numbers to compare on, not accuracy.

**Threshold frozen on out-of-fold predictions.** The classification threshold is chosen on pooled out-of-fold (OOF) cross-validation predictions and applied unchanged to the held-out test set; it is never tuned on test labels. The OOF operating point carries over to the test set with a modest drop:

| Metric | OOF (validation folds) | Test (held-out) |
|---|---|---|
| MCC | 0.212 | 0.183 |
| Balanced accuracy | 0.634 | 0.592 |
| Recall | 0.361 | 0.238 |
| F1 | 0.270 | 0.238 |

Class weighting trains the model as if passes and failures were equally common, so its scores rank runs by risk but are not calibrated failure probabilities. The pipeline uses them for ranking (PR-AUC, ROC-AUC) and for comparison against the frozen threshold.

## Deployment

- **Local** — `make ui` runs Streamlit on `http://localhost:8501`.
- **Docker** — `docker build -t fabsentinel . && docker run -p 8501:8501 fabsentinel`.
- **Hugging Face Spaces** — the live demo runs from `huggingface/spaces/` on the Docker SDK; `make deploy-hf` assembles and pushes it. Add `OPENAI_API_KEY` as a Space secret to enable the narrator.

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
| Imbalance handling | cost-sensitive (`scale_pos_weight`, auto) | `vm/train.py` |
| LLM model | gpt-4o-mini | `llm.model` |

## Tests

55 pytest tests cover ingest, data quality, every SPC module, virtual metrology, SHAP, reports, the narrator and the CLI, including Hypothesis property tests for the Western Electric and EWMA limit invariants (`tests/`). A GitHub Actions workflow runs ruff and pytest on each push to `main`.

## Scope and limitations

- SECOM rows carry timestamps but no tool, chamber or lot-genealogy metadata, so SPC runs on the full sensor vector in time order and cannot stratify by tool.
- Control limits are estimated in-sample on data that is not a verified in-control (Phase I) reference set, so more runs exceed the T² limit (156 of 1,567) than α = 0.01 implies.
- Signal screening and median imputation are fit on all 1,567 runs before the split. No labels are used, but test-set values do inform the imputation medians.
- At the frozen threshold the classifier catches 5 of 21 test failures with 16 false alarms: useful for ranking runs for engineering review, not for automatic disposition. The memo's "VM fail probability" is the model score, not a calibrated probability.
- Capability figures (Cp, Cpk) use simulated specification limits.

## License & attribution

Code: MIT.  
SECOM dataset: CC BY 4.0, McCann & Johnston, UCI ML Repository, redistributed in `data/`.

## References

See [`docs/references.md`](docs/references.md) for the full bibliography.
