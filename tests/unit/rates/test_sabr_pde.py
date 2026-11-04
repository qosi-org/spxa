"""Arbitrage-free SABR density PDE."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from spxa.market.black import bachelier_price
from spxa.rates.sabr import SABR
from spxa.rates.sabr_pde import ArbitrageFreeSABR


def test_absorbed_bachelier_is_exact_in_the_limit() -> None:
    # beta = 0, nu = 0: Brownian motion absorbed at zero, priced by reflection
    f, a, T = 0.01, 0.008, 2.0
    K = np.array([0.0, 0.005, 0.01, 0.02, 0.03])
    exact = bachelier_price(f, K, T, a) - bachelier_price(-f, K, T, a)
    errs = []
    for n, steps in ((200, 50), (400, 100), (800, 200)):
        d = ArbitrageFreeSABR(SABR(a, 0.0, 0.0, 0.0), f, T, n=n, n_steps=steps).solve()
        errs.append(np.max(np.abs(d.call(K) - exact)))
    assert errs[2] < 5e-9
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.1)
    assert errs[1] / errs[2] == pytest.approx(4.0, rel=0.1)
    assert d.P_L == pytest.approx(2 * norm.cdf(-f / (a * np.sqrt(T))), abs=1e-5)


@pytest.mark.parametrize(
    "model, F, T",
    [
        (SABR(0.35, 0.25, -0.1, 1.0), 1.0, 1.0),
        (SABR(0.02, 0.5, -0.3, 0.5), 0.01, 10.0),
        (SABR(0.01, 0.3, 0.2, 0.6, shift=0.02), -0.004, 5.0),
    ],
)
def test_conservation_and_positivity(model: SABR, F: float, T: float) -> None:
    d = ArbitrageFreeSABR(model, F, T, n=400, n_steps=80).solve()
    assert d.mass() == pytest.approx(1.0, abs=1e-12)
    assert d.mean() == pytest.approx(F, abs=1e-13)
    assert d.Q.min() >= 0.0
    assert d.f_min >= -model.shift - 1e-15
    K = np.linspace(d.f_min, F + 3 * (F - d.f_min), 200)
    c = d.call(K)
    assert np.all(np.diff(c, 2) >= -1e-14)  # convex in strike
    assert np.all(np.diff(c) <= 1e-15)


def test_close_to_hagan_for_moderate_parameters() -> None:
    m, F = SABR(0.035, 0.5, -0.3, 0.4), 0.03
    d = ArbitrageFreeSABR(m, F, 1.0, n=500, n_steps=100).solve()
    K = F * np.exp(np.linspace(-0.6, 0.6, 7))
    gap = d.implied_vol(K) - m.lognormal_vol(F, K, 1.0)
    assert abs(gap[3]) < 1e-4
    assert np.max(np.abs(gap)) < 2e-3
    assert d.P_L < 1e-4


def test_where_hagan_has_negative_density() -> None:
    m, F, T = SABR(0.02, 0.5, -0.3, 0.5), 0.01, 10.0
    K = np.linspace(0.0005, 0.03, 300)
    hagan = np.asarray(m.price(F, K, T))
    assert np.min(np.diff(hagan, 2)) < 0  # butterfly arbitrage in the expansion
    d = ArbitrageFreeSABR(m, F, T, n=400, n_steps=100).solve()
    assert np.min(np.diff(d.call(K), 2)) > 0
    assert 0.05 < d.P_L < 0.15  # a sizeable mass is absorbed at zero
    with pytest.raises(ValueError):
        ArbitrageFreeSABR(m, -0.01, T)
