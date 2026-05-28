# Methods

A compact reference for the algorithms in each module, with the choices a process engineer cares about.

## Univariate SPC

### Shewhart individuals (`spc/shewhart.py`)

SECOM is delivered as a row-per-lot dataset, so we use the **individuals** chart (subgroup size = 1). The within-subgroup variance estimator is the average of the moving range divided by Hartley's d₂:

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

Defaults λ=0.2, L=3 give an ARL₀ ≈ 200 — comparable to a 3σ Shewhart for the in-control state, but ~5× faster at detecting a 1σ shift.

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
2. Run PCA on the training set; keep `k` PCs whose cumulative explained variance ≥ 0.90.
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

> **Caveat:** the T²/SPE limits here are computed in-sample on SECOM data that is not a verified in-control (Phase I) reference set, so the empirical out-of-control rate exceeds the nominal α used to set the UCLs.

## Capability (Cp / Cpk / Cpm)

```
Cp  = (USL - LSL) / (6σ)
Cpk = min( (USL - μ)/(3σ), (μ - LSL)/(3σ) )
Cpm = (USL - LSL) / (6 sqrt(σ² + (μ - T)²))
```

**SECOM has no engineering spec limits.** We demonstrate the methodology on a simulated-spec subset where USL = μ + 4σ and LSL = μ − 4σ (so Cp = 4/3 = 1.333 by construction). The clearly-labeled disclaimer is in the scorecard and in `docs/dataset_card_secom.md`.

## Virtual metrology

### LightGBM (`vm/train.py`)

Gradient-boosted trees, 400 estimators, learning rate 0.05, `class_weight=balanced`. Why LightGBM:

- Handles ~440 retained features without manual selection.
- Fast on Apple Silicon — full 5-fold CV in under 2 minutes for SECOM.
- Native SHAP integration via TreeExplainer.

### Class imbalance

SECOM is 14:1. We:

- Apply SMOTE (sampling_strategy=0.30) **only on the training set**, never on test.
- Use `class_weight="balanced"` in the boosting itself.
- Choose the operating threshold by maximizing F1 on the PR curve (not 0.5).

### Metrics

We report PR-AUC, ROC-AUC, MCC, balanced accuracy, recall, F1, and a confusion-matrix breakdown. **Plain accuracy is intentionally absent** — a model that predicts "pass" for everything would score 93% accuracy on SECOM and catch zero fails.

## SHAP attribution

`shap.TreeExplainer` on the LightGBM model:

- **Global ranking** — mean(|SHAP|) over `n_background=100` randomly-sampled lots.
- **Local attribution** — per-lot Shapley values, signed (positive ⇒ pushes toward `fail`).

The 8D memo's D4 "Candidate Root Causes" section is generated directly from the top-k SHAP-attributed signals for the lot in question.

## LLM narrator

`llm/narrator.py` wraps the OpenAI Chat Completions API behind a tiny `Narrator` protocol. The system prompt enforces:

- Cite signal IDs and numeric values from the input.
- Cautious language: "candidate root cause", "warrants investigation".
- Never state a final conclusion.

Output is stitched into the 8D memo template behind a prominent "AI-DRAFTED — review before publishing" banner. To swap providers, implement the `Narrator` protocol and edit `get_narrator()`.
