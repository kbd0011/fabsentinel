# Blog post outline — "FabSentinel: shipping an end-to-end SPC + virtual metrology pipeline on SECOM"

## Hook (1 paragraph)
Show the scorecard screenshot. Lead with: "104 failures in 1,567 production runs, 14:1 imbalance, no spec limits. Here is an end-to-end SPC stack on this dataset, with the rough edges shown."

## Why SECOM (1 short section)
- It's public, real fab sensor data.
- Imbalance + missingness + anonymization make it honest about real-world friction.
- Free under CC BY 4.0.

## The pipeline (3–4 sections; aim for screenshots of each)
1. **Data quality first** — drop high-missing and low-variance signals. Imbalance warning. Median impute.
2. **Univariate SPC** — Shewhart / EWMA / CUSUM + Western Electric + Nelson rules on the 20 highest-variance signals.
3. **Multivariate SPC** — PCA, Hotelling T², SPE residual on all 440 retained signals, contribution plots. Show the T² vs SPE scatter.
4. **Virtual metrology** — class-weighted LightGBM, max-F1 threshold frozen on out-of-fold predictions, 5-fold CV.
5. **SHAP attribution** — global ranking and per-lot. The signals that consistently move up/down on fails.

## The numbers, in context
Held-out test (314 runs, 21 failures): PR-AUC 0.169 against a 6.7% base rate, MCC 0.183, recall 0.238 at a threshold frozen on out-of-fold predictions. A model that always predicts "pass" scores 93% accuracy and catches no failures. Show why accuracy is left out.

## The opinionated design choices
- Cp / Cpk only on a simulated-spec subset, with a disclaimer in the scorecard. Anonymization means no real specs exist.
- LLM narrator as an *editorial layer*, never a classifier. Every output banner-flagged as a draft.
- Reports are PDF via Quarto if available, HTML always.
- Reproducible: single `make demo`, MLflow tracking, deterministic seeds.

## What I'd add next
- Bayesian adaptive EWMA for the small-shift regime.
- VAE-LSTM autoencoder for unsupervised excursion detection (alternative to Hotelling T²).
- Routing-aware multivariate SPC that stratifies T² by tool/chamber when metadata is available.
- Real spec limits if any fab partner is willing to contribute them under NDA.

## Code + license
GitHub link. MIT. SECOM under CC BY 4.0.

## Closer
"FabSentinel is a small reference implementation. It will not run your fab. But it is the kind of pipeline I would propose on Day 60 in role."
