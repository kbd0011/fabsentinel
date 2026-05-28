# Architecture

```
fabsentinel/
├── src/fabsentinel/
│   ├── config.py              # Pydantic settings, single source of truth
│   ├── cli.py                 # Typer CLI (ingest, quality, …, scorecard, memo)
│   ├── data/
│   │   ├── ingest.py          # SECOM raw → typed DataFrame → parquet
│   │   └── quality.py         # Missingness, low-variance, outlier, imbalance
│   ├── spc/
│   │   ├── shewhart.py        # Individuals chart, d₂-based sigma
│   │   ├── ewma.py            # EWMA with time-varying limits
│   │   ├── cusum.py           # Tabular two-sided CUSUM
│   │   ├── rules.py           # Western Electric + Nelson rules
│   │   ├── multivariate.py    # PCA + Hotelling T² + SPE
│   │   └── capability.py      # Cp / Cpk / Cpm on simulated specs
│   ├── vm/
│   │   ├── features.py        # Label remap, missing imputation
│   │   ├── train.py           # LightGBM + SMOTE + PR threshold + MLflow
│   │   └── evaluate.py        # PR-AUC, MCC, balanced acc, recall
│   ├── rca/
│   │   └── shap_attribution.py  # TreeExplainer, global + per-lot
│   ├── reports/
│   │   ├── scorecard.py       # Weekly SPC scorecard, HTML+PDF
│   │   ├── excursion_memo.py  # 8D-style memo, HTML+PDF
│   │   └── templates/         # Jinja2 / Quarto templates
│   ├── llm/
│   │   └── narrator.py        # OpenAI-backed editorial narrator
│   └── ui/
│       └── app.py             # Streamlit dashboard
├── configs/config.yaml         # All tunable knobs
├── tests/                      # 53 pytest tests across all modules
├── data/                       # SECOM bundled (CC BY 4.0)
├── docs/                       # this directory
├── notebooks/                  # 5 exploratory notebooks
├── artifacts/                  # generated outputs (gitignored)
├── Makefile                    # `make install`, `make demo`, …
├── Dockerfile
└── pyproject.toml              # uv + hatchling
```

## Data contracts

- **Raw SECOM** — `data/secom.data` is space-separated, `NaN` markers; `data/secom_labels.data` is `<label> "DD/MM/YYYY HH:MM:SS"`. Labels are `-1` (pass) / `+1` (fail).
- **Processed parquet** — Columns: `timestamp` (datetime64), `label` (int -1/+1), then `signal_000` … `signal_589` (float, NaNs preserved).
- **Cleaned parquet** — Same shape minus signals dropped for missingness or low variance; NaNs are median-imputed.
- **VM features** — Label remapped to `0` (pass) / `1` (fail). Anything else is identical to cleaned.

## Configuration philosophy

All tunables live in `configs/config.yaml`. The Pydantic schema in `config.py` enforces:

- Types (no string-as-float bugs).
- Bounds (e.g., `pca_explained_variance` ∈ (0, 1)).
- `extra="forbid"` — typos in YAML keys are rejected at load time, not at runtime.

## Resumability

Every CLI subcommand is idempotent and resumable. `make demo` is the canonical path, but you can re-run any single stage and it will rebuild only its own artifacts. The cleaned parquet, model joblib, and MLflow runs persist across invocations.

## Reproducibility

- Python pinned to 3.12 via `.python-version`.
- All dependency versions pinned in `pyproject.toml`.
- `random_state=42` in every stochastic step (train/test split, SMOTE, LightGBM, SHAP background sample).
- MLflow logs every training run with parameters, metrics, and model artifact for full audit.
- Quarto rendering is optional; HTML is always emitted. PDFs are byte-stable across runs (modulo the timestamp).
