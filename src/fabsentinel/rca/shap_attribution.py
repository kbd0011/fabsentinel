"""SHAP attribution for the virtual metrology classifier."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from fabsentinel.vm.train import VMArtifact


@dataclass
class SHAPReport:
    global_importance: pd.Series        # mean(|SHAP|) per feature, descending
    explainer: shap.TreeExplainer
    background_sample: pd.DataFrame


def fit(artifact: VMArtifact, X_background: pd.DataFrame, n_background: int = 100) -> SHAPReport:
    """Build a SHAP TreeExplainer and compute the global feature ranking."""
    n = min(n_background, len(X_background))
    bg = X_background.sample(n=n, random_state=42) if len(X_background) > n else X_background.copy()
    explainer = shap.TreeExplainer(artifact.model)
    sv = explainer.shap_values(bg)
    # LightGBM binary returns a single array of shape (n, p)
    sv_arr = sv if isinstance(sv, np.ndarray) else np.asarray(sv)
    if sv_arr.ndim == 3:
        # (n, p, 2) for some versions; pick the positive class slice
        sv_arr = sv_arr[..., 1]
    importance = pd.Series(np.abs(sv_arr).mean(axis=0), index=bg.columns, name="mean_abs_shap")
    importance = importance.sort_values(ascending=False)
    return SHAPReport(global_importance=importance, explainer=explainer, background_sample=bg)


def top_features_for_lot(report: SHAPReport, x_row: pd.Series, top_k: int = 10) -> pd.DataFrame:
    """For a single lot, return the top_k features ranked by |SHAP|.

    Each row contains feature, value, shap_value, abs_shap.
    """
    sv = report.explainer.shap_values(x_row.to_frame().T)
    arr = sv if isinstance(sv, np.ndarray) else np.asarray(sv)
    if arr.ndim == 3:
        arr = arr[..., 1]
    arr = arr[0]
    df = pd.DataFrame(
        {
            "feature": x_row.index,
            "value": x_row.values,
            "shap_value": arr,
            "abs_shap": np.abs(arr),
        }
    ).sort_values("abs_shap", ascending=False).head(top_k).reset_index(drop=True)
    return df
