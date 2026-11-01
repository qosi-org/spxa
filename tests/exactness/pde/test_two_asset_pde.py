"""Exactness tests: two-asset Black–Scholes PDE (ADI) against closed forms."""

import numpy as np
import pytest
from scipy.stats import norm

from spxa.pde.reference import bs_price, margrabe_price, spread_option_price
from spxa.pde.two_asset import TwoAssetBlackScholesPDE

pytestmark = pytest.mark.exactness

R, S1, S2, RHO, Q1, Q2 = 0.05, 0.3, 0.2, 0.5, 0.02, 0.01
MODEL = TwoAssetBlackScholesPDE(R, S1, S2, RHO, Q1, Q2)


@pytest.mark.parametrize("rho, tol", [(-0.6, 1.5e-3), (0.0, 1.5e-3), (0.5, 1.5e-3), (0.9, 1.5e-2)])
def test_margrabe(rho, tol):
    # the payoff kink S1 = S2 is not aligned with the grid; at high correlation
    # the exchange volatility is small (0.148 for rho = 0.9) and the kink is
    # resolved more slowly, but the convergence stays second order
    m = TwoAssetBlackScholesPDE(R, S1, S2, rho, Q1, Q2)
    ref = margrabe_price(100.0, 95.0, 1.0, S1, S2, rho, Q1, Q2)
    errs = [
        abs(m.exchange(1.0, 100.0, 95.0, n1=n, n2=n, n_time=n // 2).price - ref)
        for n in (51, 101, 201)
    ]
    assert errs[2] < tol
    assert np.log2(errs[1] / errs[2]) > 1.6


def test_margrabe_greeks():
    res = MODEL.exchange(1.0, 100.0, 95.0, n1=201, n2=201, n_time=100)
    s = np.sqrt(S1**2 + S2**2 - 2 * RHO * S1 * S2)
    d1 = (np.log(100.0 / 95.0) + (Q2 - Q1 + 0.5 * s * s)) / s
    assert res.delta[0] == pytest.approx(np.exp(-Q1) * norm.cdf(d1), abs=1e-3)
    assert res.delta[1] == pytest.approx(-np.exp(-Q2) * norm.cdf(d1 - s), abs=1e-3)
    # degree-one homogeneity: S1 u_1 + S2 u_2 = u
    assert 100.0 * res.delta[0] + 95.0 * res.delta[1] == pytest.approx(res.price, rel=5e-4)


@pytest.mark.parametrize("K", [-5.0, 5.0, 15.0])
def test_spread_call(K):
    ref = spread_option_price(100.0, 95.0, K, 1.0, R, S1, S2, RHO, Q1, Q2)
    got = MODEL.spread(K, 1.0, 100.0, 95.0, n1=201, n2=201, n_time=100).price
    assert got == pytest.approx(ref, abs=1e-3)


def test_spread_parity():
    kw = dict(n1=101, n2=101, n_time=50)
    c = MODEL.spread(5.0, 1.0, 100.0, 95.0, "call", **kw).price
    p = MODEL.spread(5.0, 1.0, 100.0, 95.0, "put", **kw).price
    fwd = 100.0 * np.exp(-Q1) - 95.0 * np.exp(-Q2) - 5.0 * np.exp(-R)
    assert c - p == pytest.approx(fwd, abs=5e-3)


def test_one_dimensional_consistency():
    # a payoff depending on S1 only must reproduce Black–Scholes
    ref = bs_price(100.0, 105.0, 1.0, R, S1, Q1)
    errs = []
    for n in (51, 101, 201):
        res = MODEL.price(
            lambda a, b: np.maximum(a - 105.0, 0.0),
            1.0,
            100.0,
            95.0,
            n1=n,
            n2=n,
            n_time=n // 2,
            align=((105.0,), ()),
        )
        errs.append(abs(res.price - ref))
        assert abs(res.delta[1]) < 1e-6 and abs(res.cross_gamma) < 1e-6
    assert errs[2] < 3e-3
    assert 0.5 * np.log2(errs[0] / errs[2]) > 1.8


def test_best_plus_worst_equals_two_calls():
    kw = dict(n1=201, n2=201, n_time=100)
    best = MODEL.best_of(100.0, 1.0, 100.0, 95.0, **kw).price
    worst = MODEL.worst_of(100.0, 1.0, 100.0, 95.0, **kw).price
    ref = bs_price(100.0, 100.0, 1.0, R, S1, Q1) + bs_price(95.0, 100.0, 1.0, R, S2, Q2)
    assert best + worst == pytest.approx(ref, abs=5e-3)
    assert best > worst > 0.0


def test_american_exchange():
    # without a dividend on the delivered asset, early exercise is never optimal
    m = TwoAssetBlackScholesPDE(R, S1, S2, RHO, 0.0, Q2)
    am = m.exchange(1.0, 100.0, 95.0, n1=101, n2=101, n_time=50, american=True).price
    assert am == pytest.approx(margrabe_price(100.0, 95.0, 1.0, S1, S2, RHO, 0.0, Q2), abs=1e-3)
    # with a dividend it is worth more than the European option
    eu = MODEL.exchange(1.0, 100.0, 95.0, n1=101, n2=101, n_time=50).price
    am = MODEL.exchange(1.0, 100.0, 95.0, n1=101, n2=101, n_time=50, american=True).price
    assert am > eu + 1e-3
