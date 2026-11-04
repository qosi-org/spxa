"""SABR-LMM."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.black import black_price
from spxa.rates.lmm import rebonato_correlation
from spxa.rates.sabr_lmm import SABRLMM, sabr_lmm_paths
from tests.unit.rates.test_lmm import CURVE, TIMES

RHO = rebonato_correlation(TIMES[:-1], 0.1, 0.4)
ALPHAS = np.full(20, 0.2)


def test_lognormal_limit_is_black() -> None:
    m = SABRLMM(CURVE, TIMES, ALPHAS, beta=1.0, rho=0.0, nu=1e-8, correlation=RHO)
    p = sabr_lmm_paths(m, 20_000, 4, seed=5)
    for k in (4, 12):
        d = p.deflator(k)
        assert abs(d.mean() - CURVE.discount(TIMES[k])) < 3 * d.std(ddof=1) / np.sqrt(d.size)
    for i, K in ((2, 0.03), (10, 0.035)):
        black = black_price(m.forwards[i], K, TIMES[i], 0.2, m.discounts[i + 1] * m.tau[i], "call")
        pv, se = p.price(m.tau[i] * np.maximum(p.fixings[:, i] - K, 0.0), i + 1)
        assert abs(pv - black) < 3 * se


def test_caplet_smile_close_to_hagan() -> None:
    # beta = 0.5: alpha = 0.2 * F^(1 - beta) gives about 20% lognormal ATM vol
    m = SABRLMM(
        CURVE, TIMES, 0.2 * 0.0305**0.5 * np.ones(20), beta=0.5, rho=-0.3, nu=0.4, correlation=RHO
    )
    p = sabr_lmm_paths(m, 40_000, 8, seed=6)
    for i in (0, 4):
        for K in (0.02, m.forwards[i], 0.045):
            pv, se = p.price(m.tau[i] * np.maximum(p.fixings[:, i] - K, 0.0), i + 1)
            assert abs(pv - m.caplet(i, K)) < 3 * se + 0.01 * pv


def test_validation() -> None:
    with pytest.raises(ValueError):
        SABRLMM(CURVE, TIMES, ALPHAS, 0.5, rho=0.99, nu=0.3, correlation=RHO)
    with pytest.raises(ValueError):
        SABRLMM(CURVE, TIMES, ALPHAS[:3], 0.5, 0.0, 0.3)
