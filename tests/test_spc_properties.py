"""Property-based tests for SPC invariants (Hypothesis).

These assert structural guarantees that must hold for *any* admissible input,
not just hand-picked fixtures:

1. Western-Electric rule 1 never fires on a series whose points all lie strictly
   within +/-3 sigma of the mean (rule 1 is, by definition, "one point beyond 3 sigma").
2. The EWMA control-limit half-width is monotonically non-decreasing in time t and
   converges to its known steady-state value L * sigma * sqrt(lam / (2 - lam)).
"""
from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from fabsentinel.spc.ewma import fit_and_detect as ewma_fit
from fabsentinel.spc.rules import western_electric


@settings(max_examples=200, deadline=None)
@given(
    mean=st.floats(min_value=-100.0, max_value=100.0),
    sigma=st.floats(min_value=1e-3, max_value=50.0),
    # z-scores strictly inside +/-3 sigma => rule 1 (|z| > 3) must never trigger
    zscores=st.lists(
        st.floats(min_value=-2.999, max_value=2.999),
        min_size=1,
        max_size=200,
    ),
)
def test_rule1_never_fires_within_3sigma(mean: float, sigma: float, zscores: list[float]) -> None:
    """If every observation is within +/-3 sigma, WE rule 1 produces an all-False mask."""
    x = mean + sigma * np.asarray(zscores, dtype=float)
    mask = western_electric(x, mean=mean, sigma=sigma, rules=[1])
    assert mask.shape == x.shape
    assert not mask.any()


@settings(max_examples=200, deadline=None)
@given(
    lam=st.floats(min_value=0.01, max_value=0.99),
    L=st.floats(min_value=1.0, max_value=4.0),
    sigma=st.floats(min_value=1e-3, max_value=50.0),
    n=st.integers(min_value=2, max_value=300),
)
def test_ewma_limit_width_monotone_and_converges(lam: float, L: float, sigma: float, n: int) -> None:
    """EWMA half-width is non-decreasing in t and converges to L*sigma*sqrt(lam/(2-lam))."""
    # Constant series => mu == the constant, control limits depend only on t, lam, L, sigma.
    x = np.full(n, 7.0)
    state, _ = ewma_fit(x, lam=lam, L=L, target_mean=7.0, target_sigma=sigma)

    half_width = state.ucl - state.mean  # = L * sigma_z(t)
    assert np.all(half_width >= 0.0)

    # Monotonic non-decreasing in t (allow tiny float slack).
    diffs = np.diff(half_width)
    assert np.all(diffs >= -1e-9)

    # Bounded above by, and monotonically approaching, the steady-state half-width.
    steady = L * sigma * np.sqrt(lam / (2 - lam))
    assert np.all(half_width <= steady + 1e-6)

    # The half-width follows the exact analytic form
    #   L * sigma * sqrt(lam/(2-lam) * (1 - (1-lam)^(2t))),
    # which equals steady * sqrt(1 - (1-lam)^(2t)) and -> steady as t -> inf.
    t = np.arange(1, n + 1)
    expected = steady * np.sqrt(1 - (1 - lam) ** (2 * t))
    assert np.allclose(half_width, expected, atol=1e-9, rtol=1e-9)
    # Remaining gap to steady state shrinks strictly toward zero with t.
    assert (steady - half_width[-1]) <= (steady - half_width[0]) + 1e-12
