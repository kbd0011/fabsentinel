# Dataset Card — UCI SECOM

## Provenance

- **Authors**: Michael McCann and Adrian Johnston, 2008.
- **Source**: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/179/secom
- **License**: Creative Commons Attribution 4.0 (CC BY 4.0). Redistribution permitted with attribution.
- **Provenance note**: The dataset is anonymized historical data from a real semiconductor fab. The original tool / process / step identities are intentionally not disclosed.

## Schema

| File | Rows | Columns | Description |
|---|---|---|---|
| `secom.data` | 1,567 | 590 | Sensor / process measurement signals, space-separated, `NaN` markers. |
| `secom_labels.data` | 1,567 | 2 | `<label> "DD/MM/YYYY HH:MM:SS"`. Label is `-1` (pass) or `+1` (fail). |

After ingest, FabSentinel produces:

| File | Rows | Columns |
|---|---|---|
| `data/processed.parquet` | 1,567 | 592 (`timestamp`, `label`, `signal_000`…`signal_589`) |
| `data/cleaned.parquet` | 1,567 | ≈442 (dropped 24 high-missing + 126 low-variance signals at default thresholds) |

## Class distribution

| Class | Count | Fraction |
|---|---|---|
| Pass (`-1`) | 1,463 | 93.4% |
| Fail (`+1`) | 104 | 6.6% |
| **Pass:fail ratio** | **14.1:1** | |

This is a deliberately imbalanced classification problem; reporting plain accuracy is misleading.

## Known limitations

- **Anonymization** means there are no engineering specification limits. Cp / Cpk in FabSentinel are demonstrated on a clearly-labeled simulated-spec subset.
- **Missing values** are common — ~24 signals have >55% missing data at the default threshold.
- **Low-variance / sensor-broken columns** — many signals are near-constant for long stretches; FabSentinel drops these by default.
- **Time span** — about 4 months of production data in 2008. Concept drift over longer horizons is not exercised.
- **No tool / chamber metadata** — multivariate SPC is on the full sensor vector; we cannot stratify by chamber.

## Why we use it

- It is the only freely available, peer-cited fab dataset of meaningful size (1.5k+ lots, 500+ signals).
- The class imbalance and missingness pattern are realistic.
- The community has converged on PR-AUC and MCC as the right metrics, which keeps benchmarking honest.

## How we redistribute it

Both raw files are committed to `data/` under their original license. Attribution is in `README.md` and `docs/references.md`. No modifications to the source files.
