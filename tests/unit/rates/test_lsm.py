"""Least-squares Monte Carlo for Bermudan swaptions."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.bermudan_engines import bermudan_pde
from spxa.rates.hull_white import HullWhite
from spxa.rates.hw_mc import hull_white_transition
from spxa.rates.hw_pde import HullWhitePDE
from spxa.rates.lsm import hull_white_exercise_problem, lsm
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_bermudan import DISC, PROJ, REF, _ber

HW = HullWhite(DISC, 0.05, 0.01)


def test_transition_is_exact() -> None:
    hw = HullWhite(DISC, 0.05, [0.01, 0.012], [1.5])
    rng = np.random.default_rng(0)
    x, D, t = np.zeros(200_000), np.ones(200_000), 0.0
    for t1 in (0.7, 1.9, 3.0):  # the middle step crosses the volatility knot
        x, d = hull_white_transition(hw, t, t1, x, rng)
        D, t = D * d, t1
        assert abs(D.mean() - DISC.discount(t1)) < 3 * D.std() / np.sqrt(D.size)
        v = D * hw.discount_bond(t1, 6.0, x)
        assert abs(v.mean() - DISC.discount(6.0)) < 3 * v.std() / np.sqrt(v.size)
    assert x.var() == pytest.approx(hw.y(3.0), rel=0.02)


def test_single_exercise_is_the_european() -> None:
    eu = swaption(REF, "1Y", "5Y", 0.035, notional=1e4)
    res = lsm(hull_white_exercise_problem(HW, _ber().restrict([0])), 100_000, seed=3)
    assert abs(res.price - HW.swaption(eu, PROJ)) < 3 * res.stderr


def test_lower_bound_close_to_pde() -> None:
    ber = _ber()
    pde = bermudan_pde(HullWhitePDE(HW, 401, 400), ber)
    res = lsm(hull_white_exercise_problem(HW, ber), 200_000, seed=1, n_train=50_000)
    # a feasible policy: below the true value up to noise, and within 0.5% of it
    assert res.price < pde + 3 * res.stderr
    assert res.price > 0.995 * pde - 3 * res.stderr
    assert res.stderr < 0.003 * pde
