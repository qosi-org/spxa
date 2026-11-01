"""Exactness tests: American and Bermudan options under Black-Scholes."""

import math

import numpy as np
import pytest

from spxa.pde.black_scholes import BlackScholesPDE
from spxa.pde.reference import binomial_price, bs_price, perpetual_put_boundary

pytestmark = pytest.mark.exactness


def crr_bermudan_put(S, K, T, r, sigma, dates, n=3000):
    """CRR tree with exercise only at the steps nearest to the given dates."""
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    ex_steps = {int(round(t / dt)) for t in dates}
    j = np.arange(n + 1)
    v = np.maximum(K - S * u ** (n - j) * d**j, 0.0)
    for step in range(n - 1, -1, -1):
        v = disc * (p * v[:-1] + (1 - p) * v[1:])
        if step in ex_steps:
            jj = np.arange(step + 1)
            v = np.maximum(v, K - S * u ** (step - jj) * d**jj)
    return float(v[0])


class TestAmericanPut:
    @pytest.mark.parametrize("S", [36.0, 40.0, 44.0])
    def test_longstaff_schwartz_table(self, S):
        pde = BlackScholesPDE(0.06, 0.2).american(40.0, 1.0, S, "put").price
        tree = binomial_price(S, 40.0, 1.0, 0.06, 0.2, 0.0, "put", True, 4001)
        assert pde == pytest.approx(tree, abs=1e-3)

    def test_at_the_money_benchmark(self):
        pde = BlackScholesPDE(0.05, 0.2).american(100.0, 1.0, 100.0, "put", n_space=801, n_time=400)
        tree = binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "put", True, 4001)
        assert pde.price == pytest.approx(tree, abs=5e-4)
        assert pde.price == pytest.approx(6.0903, abs=1e-3)

    def test_methods_agree(self):
        m = BlackScholesPDE(0.06, 0.2)
        kw = dict(n_space=201, n_time=100)
        prices = {
            meth: m.american(40.0, 1.0, 40.0, "put", method=meth, **kw).price
            for meth in ("penalty", "brennan_schwartz", "ikonen_toivanen")
        }
        assert prices["penalty"] == pytest.approx(prices["brennan_schwartz"], abs=1e-8)
        assert prices["ikonen_toivanen"] == pytest.approx(prices["penalty"], abs=5e-4)
        psor = m.american(40.0, 1.0, 40.0, "put", method="psor", n_space=81, n_time=40).price
        pen = m.american(40.0, 1.0, 40.0, "put", n_space=81, n_time=40).price
        assert psor == pytest.approx(pen, abs=1e-6)

    def test_convergence_under_refinement(self):
        m = BlackScholesPDE(0.05, 0.2)
        vals = [
            m.american(100, 1.0, 100, n_space=100 * 2**k + 1, n_time=50 * 2**k).price
            for k in range(4)
        ]
        d = np.abs(np.diff(vals))
        assert np.all(d[:-1] / d[1:] > 2.8)  # at least ~1.5 order

    def test_bounds_and_greeks(self):
        m = BlackScholesPDE(0.06, 0.2)
        res = m.american(40.0, 1.0, 40.0, "put")
        euro = float(bs_price(40.0, 40.0, 1.0, 0.06, 0.2, 0.0, "put"))
        assert res.price > euro
        s = np.linspace(20, 80, 61)
        assert np.all(res.solution.value(s) >= np.maximum(40 - s, 0) - 1e-6)
        assert -1.0 <= res.delta <= 0.0 and res.gamma > 0.0
        assert "penalty" in res.__story__()


class TestExerciseBoundary:
    def test_boundary_properties(self):
        K, r, sig = 40.0, 0.06, 0.2
        res = BlackScholesPDE(r, sig).american(K, 1.0, 40.0, "put", n_space=801, n_time=400)
        eb = res.exercise_boundary
        assert eb.side == "left"
        locs = eb.locations[np.isfinite(eb.locations)]
        # S*(tau) is non-increasing in tau, below K and above the perpetual boundary
        assert np.all(np.diff(locs) <= 1e-9)
        assert np.all(locs < K)
        assert locs[-1] > perpetual_put_boundary(K, r, sig)
        # deep in the exercise region the option is worth its intrinsic value
        s_star = eb.at(1.0)
        assert res.solution.value(s_star - 2.0) == pytest.approx(K - s_star + 2.0, abs=1e-6)
        assert res.solution.value(s_star + 2.0) > K - s_star - 2.0
        t, loc = eb.calendar(1.0)
        assert t[0] == pytest.approx(0.0) and loc[0] == pytest.approx(s_star)

    def test_tree_consistent_boundary(self):
        # at the boundary today the tree price equals intrinsic, slightly above it does not
        K, r, sig = 40.0, 0.06, 0.2
        res = BlackScholesPDE(r, sig).american(K, 1.0, 40.0, "put", n_space=801, n_time=400)
        s_star = res.exercise_boundary.at(1.0)
        below = binomial_price(s_star - 0.3, K, 1.0, r, sig, 0.0, "put", True, 2001)
        above = binomial_price(s_star + 0.5, K, 1.0, r, sig, 0.0, "put", True, 2001)
        assert below == pytest.approx(K - s_star + 0.3, abs=2e-3)
        assert above > K - s_star - 0.5 + 1e-3


class TestAmericanCall:
    def test_no_dividends_equals_european(self):
        res = BlackScholesPDE(0.05, 0.2).american(100.0, 1.0, 100.0, "call")
        assert res.price == pytest.approx(float(bs_price(100, 100, 1.0, 0.05, 0.2)), abs=2e-4)

    @pytest.mark.parametrize("method", ["penalty", "brennan_schwartz"])
    def test_with_dividend_yield(self, method):
        res = BlackScholesPDE(0.05, 0.25, 0.06).american(100.0, 1.0, 100.0, "call", method=method)
        tree = binomial_price(100, 100, 1.0, 0.05, 0.25, 0.06, "call", True, 4001)
        assert res.price == pytest.approx(tree, abs=1e-3)
        assert res.exercise_boundary.side == "right"

    def test_bump_greeks(self):
        res = BlackScholesPDE(0.06, 0.2).american(
            40.0, 1.0, 40.0, "put", bump_greeks=True, n_space=201, n_time=100
        )
        assert res.vega > 0.0 and res.rho < 0.0


class TestBermudan:
    def test_against_tree(self):
        dates = [0.25, 0.5, 0.75]
        pde = BlackScholesPDE(0.05, 0.2).bermudan(100.0, 1.0, 100.0, dates, "put")
        tree = crr_bermudan_put(100.0, 100.0, 1.0, 0.05, 0.2, dates)
        assert pde.price == pytest.approx(tree, abs=3e-3)
        assert pde.solution.info["exercise_dates"] == 3

    def test_ordering(self):
        m = BlackScholesPDE(0.05, 0.2)
        euro = m.european(100.0, 1.0, 100.0, "put").price
        b4 = m.bermudan(100.0, 1.0, 100.0, [0.25, 0.5, 0.75], "put").price
        b12 = m.bermudan(100.0, 1.0, 100.0, np.arange(1, 12) / 12, "put").price
        am = m.american(100.0, 1.0, 100.0, "put").price
        assert euro < b4 < b12 < am

    def test_exercise_today_only(self):
        m = BlackScholesPDE(0.05, 0.2)
        euro = m.european(110.0, 1.0, 90.0, "put").price
        b = m.bermudan(110.0, 1.0, 90.0, [0.0], "put").price
        assert b == pytest.approx(max(euro, 20.0), abs=1e-6)

    def test_validation(self):
        with pytest.raises(ValueError):
            BlackScholesPDE(0.05, 0.2).bermudan(100, 1.0, 100, [-0.1])
