"""SABR Monte Carlo against closed forms and the density PDE."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from spxa.market.black import bachelier_price
from spxa.rates.sabr import SABR
from spxa.rates.sabr_mc import sabr_paths
from spxa.rates.sabr_pde import ArbitrageFreeSABR


def test_absorbed_normal_model_with_bridge() -> None:
    f, a, T = 0.01, 0.008, 2.0
    K = np.array([0.005, 0.01, 0.02])
    exact = bachelier_price(f, K, T, a) - bachelier_price(-f, K, T, a)
    p_abs = 2 * norm.cdf(-f / (a * np.sqrt(T)))
    se_abs = np.sqrt(p_abs * (1 - p_abs) / 200_000)
    p = sabr_paths(SABR(a, 0.0, 0.0, 0.0), f, T, 50, 200_000, seed=2)
    c, se = p.call(K)
    assert np.all(np.abs(c - exact) < 4 * se)
    assert abs(p.absorbed.mean() - p_abs) < 4 * se_abs
    # without the bridge, crossings inside a step are missed
    q = sabr_paths(SABR(a, 0.0, 0.0, 0.0), f, T, 50, 200_000, seed=2, bridge=False)
    assert p_abs - q.absorbed.mean() > 20 * se_abs


def test_matches_the_pde_for_moderate_parameters() -> None:
    m, F, T = SABR(0.035, 0.5, -0.3, 0.4), 0.03, 1.0
    K = F * np.exp(np.linspace(-0.6, 0.6, 7))
    p = sabr_paths(m, F, T, 200, 400_000, seed=1)
    c, se = p.call(K)
    d = ArbitrageFreeSABR(m, F, T, n=500, n_steps=100).solve()
    assert np.all(np.abs(c - d.call(K)) < 4 * se)
    assert np.all(np.abs(c - np.asarray(m.price(F, K, T))) < 4 * se)
    assert abs(p.F.mean() - F) < 4 * p.F.std() / np.sqrt(p.F.size)
    put, se_p = p.put(K)
    assert put - c == pytest.approx(K - p.F.mean(), abs=1e-15)


def test_the_effective_pde_is_an_expansion() -> None:
    # nu^2 T = 2.5: the one-dimensional PDE departs from the two-factor model
    m, F, T = SABR(0.02, 0.5, -0.3, 0.5), 0.01, 10.0
    p = sabr_paths(m, F, T, 400, 200_000, seed=3)
    d = ArbitrageFreeSABR(m, F, T).solve()
    c, se = p.call(F)
    assert abs(c[0] - d.call(F)) > 10 * se[0]
    assert p.absorbed.mean() > d.P_L
