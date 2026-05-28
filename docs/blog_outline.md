# Blog post outline — "FabSentinel: shipping an end-to-end SPC + virtual metrology pipeline on SECOM"

## Hook (1 paragraph)
Show the scorecard screenshot. Lead with: "104 fails in 1,567 lots, 14:1 imbalance, no spec limits. Here is what a credible production-ready SPC stack looks like on this dataset, end-to-end, with the rough edges shown."

## Why SECOM (1 short section)
- It's the dataset everyone in fab ML benchmarks on.
- Imbalance + missingness + anonymization make it honest about real-world friction.
- Free under CC BY 4.0.

## The pipeline (3–4 sections; aim for screenshots of each)
1. **Data quality first** — drop high-missing and low-variance signals. Imbalance warning. Median impute.
2. **Univariate SPC** — Shewhart / EWMA / CUSUM + Western Electric + Nelson rules on the top-variance signals.
3. **Multivariate SPC** — PCA, Hotelling T², SPE residual, contribution plots. Show the T² vs SPE scatter.
4. **Virtual metrology** — LightGBM, SMOTE on training only, PR-curve max-F1 threshold, 5-fold CV.
5. **SHAP attribution** — global ranking and per-lot. The signals that consistently move up/down on fails.

## Honest about the numbers
PR-AUC 0.18, MCC 0.32. These are typical for SECOM in the literature. Anyone claiming 95%+ accuracy on SECOM has either leaked data or is reporting the wrong metric. Show why.

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
"FabSentinel is a small reference implementation. It will not run your fab. But it is the kind of pipeline I would propose on Day 60 in role." Then list the 8 Micron job IDs this maps to in a small table at the end (link to the role / company).
