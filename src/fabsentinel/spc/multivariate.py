"""Multivariate SPC: PCA, Hotelling T^2 in principal subspace, SPE / Q residual.

References:
- Jackson, J.E. (1991). A User's Guide to Principal Components.
- Kourti, T. & MacGregor, J.F. (1995). "Process analysis, monitoring and diagnosis,
  using multivariate projection methods." Chemometrics and Intel. Lab. Sys.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class MultivariateSPCModel:
    """Fitted multivariate SPC model: standardizer + PCA + control limits."""
    scaler: StandardScaler
    pca: PCA
    n_components: int
    explained_variance: float
    t2_ucl: float
    spe_ucl: float
    alpha: float
    feature_names: list[str]


def fit(
    X: pd.DataFrame | np.ndarray,
    explained_variance: float = 0.90,
    alpha: float = 0.01,
) -> MultivariateSPCModel:
    """Fit PCA-based multivariate SPC model on training (presumed in-control) data.

    Parameters
    ----------
    X:
        Cleaned signal matrix (no NaNs).  Rows are lots, columns are signals.
    explained_variance:
        Choose the smallest number of PCs whose cumulative explained variance is >= this.
    alpha:
        Type-I error rate for control limits (0.01 => 99% UCL).

    Returns
    -------
    Fitted MultivariateSPCModel.
    """
    if hasattr(X, "values"):
        feature_names = list(X.columns)
        arr = X.values
    else:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
        arr = np.asarray(X, dtype=float)

    scaler = StandardScaler().fit(arr)
    Xs = scaler.transform(arr)
    pca_full = PCA().fit(Xs)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cum, explained_variance) + 1)
    k = max(1, min(k, Xs.shape[1] - 1))

    pca = PCA(n_components=k).fit(Xs)
    T = pca.transform(Xs)

    # Hotelling T^2 UCL via F distribution
    n = Xs.shape[0]
    f_crit = stats.f.ppf(1 - alpha, dfn=k, dfd=n - k)
    t2_ucl = float(k * (n - 1) * (n + 1) / (n * (n - k)) * f_crit)

    # SPE / Q residual UCL via Jackson-Mudholkar (Jackson 1991)
    residuals = Xs - pca.inverse_transform(T)
    spe = np.sum(residuals ** 2, axis=1)
    # Robust UCL via chi-squared moment-matching on SPE samples
    spe_mean = float(spe.mean())
    spe_var = float(spe.var(ddof=1))
    if spe_var <= 0 or spe_mean <= 0:
        spe_ucl = float("inf")
    else:
        h = (2 * spe_mean ** 2) / spe_var
        g = spe_var / (2 * spe_mean)
        spe_ucl = float(g * stats.chi2.ppf(1 - alpha, df=h))

    return MultivariateSPCModel(
        scaler=scaler,
        pca=pca,
        n_components=k,
        explained_variance=float(cum[k - 1]),
        t2_ucl=t2_ucl,
        spe_ucl=spe_ucl,
        alpha=alpha,
        feature_names=feature_names,
    )


def compute_statistics(model: MultivariateSPCModel, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Compute T^2 and SPE for new observations.

    Returns a DataFrame with columns [t2, spe, t2_ooc, spe_ooc].
    """
    if hasattr(X, "values"):
        arr = X.values
        index = X.index
    else:
        arr = np.asarray(X, dtype=float)
        index = pd.RangeIndex(arr.shape[0])

    Xs = model.scaler.transform(arr)
    T = model.pca.transform(Xs)
    lambdas = model.pca.explained_variance_
    # Hotelling T^2 = sum_i t_i^2 / lambda_i
    t2 = (T ** 2 / lambdas).sum(axis=1)

    residuals = Xs - model.pca.inverse_transform(T)
    spe = (residuals ** 2).sum(axis=1)

    out = pd.DataFrame(
        {
            "t2": t2,
            "spe": spe,
            "t2_ooc": t2 > model.t2_ucl,
            "spe_ooc": spe > model.spe_ucl,
        },
        index=index,
    )
    return out


def contribution_plot(model: MultivariateSPCModel, x_row: np.ndarray | pd.Series, top_k: int = 10) -> pd.Series:
    """Per-signal contribution to T^2 for a single observation.

    Uses the Westerhuis et al. (2000) formulation:
        contrib_i = sum_a (t_a / lambda_a) * P[i, a] * Xs[i]
    """
    arr = np.asarray(x_row, dtype=float).reshape(1, -1)
    Xs = model.scaler.transform(arr)[0]
    P = model.pca.components_.T            # (n_features, k)
    t = (P.T @ Xs)                          # (k,)
    lambdas = model.pca.explained_variance_  # (k,)
    weights = t / lambdas
    contrib = np.zeros_like(Xs)
    for a in range(model.n_components):
        contrib += weights[a] * P[:, a] * Xs
    s = pd.Series(contrib, index=model.feature_names, name="t2_contribution")
    return s.abs().sort_values(ascending=False).head(top_k)
