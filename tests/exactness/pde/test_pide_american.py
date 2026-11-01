"""Exactness tests: American options under jump diffusions via the PIDE."""

import numpy as np
import pytest

from spxa.pde.black_scholes import BlackScholesPDE
from spxa.pde.pide import LevyPIDE
from spxa.pde.reference import merton_price
from spxa.zoo.levy import MertonJumpDiffusion

pytestmark = pytest.mark.exactness

# d'Halluin, Forsyth & Vetzal (2005) American put; reference values from the
# literature (confirmed by compact fourth-order schemes, arXiv:1804.09043)
REFERENCE = {90.0: 10.003866, 100.0: 3.241207, 110.0: 1.419790}


@pytest.fixture(scope="module")
def merton() -> LevyPIDE:
    proc = MertonJumpDiffusion(drift=0.0, sigma=0.15, rate=0.1, jump_mean=-0.9, jump_std=0.45)
    return LevyPIDE.from_process(proc, 0.05)


@pytest.mark.parametrize("S", sorted(REFERENCE))
def test_published_american_put(merton, S):
    res = merton.american(100.0, 0.25, S, "put", n_space=1601, n_time=400)
    assert res.price == pytest.approx(REFERENCE[S], abs=2e-4)


def test_lcp_solvers_agree(merton):
    pen = merton.american(100.0, 0.25, 100.0, n_space=401, n_time=100)
    bs = merton.american(100.0, 0.25, 100.0, method="brennan_schwartz", n_space=401, n_time=100)
    it = merton.american(100.0, 0.25, 100.0, method="ikonen_toivanen", n_space=401, n_time=100)
    assert bs.price == pytest.approx(pen.price, abs=1e-8)
    # operator splitting carries an O(dt) splitting error
    assert it.price == pytest.approx(pen.price, abs=5e-4)
    assert "Ikonen" in it.solution.info["lcp"]


def test_second_order_away_from_the_boundary(merton):
    vals = [
        merton.american(100.0, 0.25, 95.0, n_space=n, n_time=n // 4).price for n in (401, 801, 1601)
    ]
    d1, d2 = vals[1] - vals[0], vals[2] - vals[1]
    assert np.log2(abs(d1 / d2)) > 1.7


def test_early_exercise_premium_and_boundary(merton):
    eu = merton_price(95.0, 100.0, 0.25, 0.05, 0.15, 0.1, -0.9, 0.45, option="put")
    am = merton.american(100.0, 0.25, 95.0, n_space=801, n_time=200)
    assert am.price > eu + 0.05
    eb = am.exercise_boundary
    assert eb is not None and eb.side == "left"
    final = eb.locations[-1]
    # the free boundary sits just below 90 (S = 90 is barely in continuation)
    assert 89.0 < final < 90.0
    loc = eb.locations[np.isfinite(eb.locations)]
    h = float(np.diff(am.solution.grid.nodes)[0])
    # non-increasing in tau, up to one grid cell (~0.33 at S = 90)
    assert np.all(np.diff(loc) <= 100.0 * np.expm1(h) + 1e-9)
    assert loc[0] == pytest.approx(100.0, abs=1.0)
    # value equals intrinsic in the exercise region and exceeds it outside
    assert am.solution.value(85.0) == pytest.approx(15.0, abs=1e-6)
    assert am.solution.value(92.0) > 8.0


def test_reduces_to_black_scholes_american():
    P = LevyPIDE(0.06, 0.0, 0.2, lambda y: np.zeros_like(y))
    pde = BlackScholesPDE(0.06, 0.2).american(40.0, 1.0, 36.0, "put", n_space=801, n_time=400)
    res = P.american(40.0, 1.0, 36.0, "put", n_space=1601, n_time=400)
    # Longstaff-Schwartz test case: 4.48644 (LR tree), both PDEs agree
    assert res.price == pytest.approx(4.48644, abs=3e-4)
    assert res.price == pytest.approx(pde.price, abs=3e-4)


def test_american_call_without_dividends_is_european(merton):
    eu = merton.european(100.0, 0.25, 100.0, "call", n_space=801)
    am = merton.american(100.0, 0.25, 100.0, "call", n_space=801)
    assert am.price == pytest.approx(eu.price, abs=1e-6)
