# Methods

A compact reference for the algorithms in each module, with the choices a process engineer cares about.

## Univariate SPC

The CLI charts the 20 highest-variance retained signals in the committed run (`artifacts/univariate_ooc.csv`; `fabsentinel univariate --top-n` sets the count). Multivariate SPC below covers all 440 retained signals.

### Shewhart individuals (`spc/shewhart.py`)

SECOM has one row per production run (a "lot" in the code), so we use the **individuals** chart (subgroup size = 1). The within-subgroup variance estimator is the average of the moving range divided by Hartley's d₂:

```
σ_hat = MR_bar / d₂      where  d₂ = 1.128  (n=2)
UCL   = μ + 3 σ_hat
LCL   = μ - 3 σ_hat
```

This is the standard choice in fab SPC for sensor traces with no natural subgrouping.

### EWMA (`spc/ewma.py`)

```
z_t = λ x_t + (1 - λ) z_{t-1}            with z_0 = μ_0
σ_z(t) = σ · sqrt( λ/(2-λ) · (1 - (1-λ)^{2t}) )
UCL(t) = μ_0 + L · σ_z(t)
```

Defaults λ=0.2, L=3 are a standard setting for small sustained shifts that a Shewhart chart is slow to catch; Montgomery (ch. 9) tabulates the ARL trade-offs.

### CUSUM (`spc/cusum.py`)

Tabular two-sided form:

```
C⁺_t = max(0, C⁺_{t-1} + (x_t - μ) - k σ)
C⁻_t = min(0, C⁻_{t-1} + (x_t - μ) + k σ)
OOC  if C⁺_t > h σ  or  C⁻_t < -h σ
```

Defaults k=0.5σ, h=5σ → ARL₀ ≈ 465, ARL₁(1σ) ≈ 10.

### Run rules (`spc/rules.py`)

Both Western Electric (rules 1–4) and Nelson (rules 1–8) are implemented as vectorized scans. Activate per-rule via `configs/config.yaml`.

## Multivariate SPC

PCA + Hotelling T² + SPE/Q residual (`spc/multivariate.py`) — the canonical Kourti & MacGregor (1995) approach.

1. Standardize each signal.
2. Run PCA on all cleaned production runs (in-sample; see the caveat below); keep `k` PCs whose cumulative explained variance ≥ 0.90.
3. **Hotelling T²** on the principal subspace:

   ```
   T² = sum_{a=1..k} t_a² / λ_a
   UCL = k(n-1)(n+1) / (n(n-k)) · F_{α,k,n-k}
   ```

4. **SPE / Q residual** in the orthogonal complement; UCL via Jackson–Mudholkar χ² moment matching:

   ```
   h = 2 μ² / σ²
   g = σ² / (2 μ)
   UCL = g · χ²_{1-α, df=h}
   ```

5. **T² contributions** (Westerhuis et al., 2000) attribute a T²-detected excursion to the signals driving the loadings.

> **Caveat:** the T²/SPE limits here are computed in-sample on SECOM data that is not a verified in-control (Phase I) reference set, so the empirical out-of-control rate exceeds the nominal α used to set the UCLs: 156 of 1,567 runs exceed the T² UCL at α = 0.01 (`artifacts/multivariate_stats.csv`).

## Capability (Cp / Cpk / Cpm)

```
Cp  = (USL - LSL) / (6σ)
Cpk = min( (USL - μ)/(3σ), (μ - LSL)/(3σ) )
Cpm = (USL - LSL) / (6 sqrt(σ² + (μ - T)²))
```

**SECOM has no engineering spec limits.** We demonstrate the methodology on a simulated-spec subset where USL = μ + 4σ and LSL = μ − 4σ (so Cp = 4/3 = 1.333 by construction). The clearly-labeled disclaimer is in the scorecard and in `docs/dataset_card_secom.md`.

## Virtual metrology

### LightGBM (`vm/train.py`)

Gradient-boosted trees, 400 estimators, learning rate 0.05, `scale_pos_weight = n_neg / n_pos` (about 14.1 on the training split). Why LightGBM:

- Handles the 440 retained features without manual selection.
- Native SHAP integration via TreeExplainer.

### Class imbalance

SECOM is 14:1. We:

- Weight the failure class by `scale_pos_weight = n_neg / n_pos` instead of resampling. This fits the model as if the classes were balanced, so its scores are not calibrated failure probabilities; they are used for ranking and thresholding.
- Choose the operating threshold by maximizing F1 on the PR curve of pooled out-of-fold predictions (stratified 5-fold CV on the training split), then apply it unchanged to the held-out test set (314 runs).

Signal screening and median imputation (`data/quality.py`) run on all 1,567 runs before the train/test split; they use no labels.

### Metrics

We report PR-AUC, ROC-AUC, MCC, balanced accuracy, recall, F1, and a confusion-matrix breakdown. **Plain accuracy is intentionally absent** — a model that predicts "pass" for everything would score 93% accuracy on SECOM and catch zero fails.

## SHAP attribution

`shap.TreeExplainer` on the LightGBM model:

- **Global ranking** — mean(|SHAP|) over `n_background=100` randomly sampled production runs.
- **Local attribution** — per-lot Shapley values, signed (positive ⇒ pushes toward `fail`).

The 8D memo's D4 "Candidate Root Causes" section lists the five signals with the largest |SHAP| for the run in question, ranked by SHAP only. The memo shows the run's top Hotelling T² contributors in a separate table.

## LLM narrator

`llm/narrator.py` wraps the OpenAI Chat Completions API behind a tiny `Narrator` protocol. The system prompt enforces:

- Cite signal IDs and numeric values from the input.
- Cautious language: "candidate root cause", "warrants investigation".
- Never state a final conclusion.

Output goes into the 8D memo under an "AI-Drafted Narrative (suggestion only)" heading with a review note, and the Streamlit UI shows it with an "AI-DRAFTED — review before publishing" label. To swap providers, implement the `Narrator` protocol and edit `get_narrator()`.
