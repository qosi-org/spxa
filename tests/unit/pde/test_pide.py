"""Unit tests for the exponential-Lévy PIDE solver."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from spxa.pde.grid import Grid1D
from spxa.pde.payoffs import Payoff, digital, vanilla
from spxa.pde.pide import LevyPIDE
from spxa.pde.reference import bs_price
from spxa.zoo.levy import MertonJumpDiffusion


def no_jumps(y):
    return np.zeros_like(np.asarray(y, dtype=float))


def merton_density(y, lam=0.5, mu=-0.1, delta=0.2):
    return lam * norm.pdf(y, mu, delta)


class TestConstruction:
    def test_validation_and_repr(self):
        with pytest.raises(ValueError):
            LevyPIDE(0.05, 0.0, -0.1, no_jumps)
        P = LevyPIDE(0.05, 0.01, 0.2, no_jumps, name="BS")
        assert "BS" in repr(P) and "r=0.05" in repr(P)

    def test_from_process(self):
        m = MertonJumpDiffusion(drift=0.0, sigma=0.15, rate=0.1, jump_mean=-0.9, jump_std=0.45)
        P = LevyPIDE.from_process(m, 0.05, 0.01)
        assert P.sigma == pytest.approx(0.15)
        assert P.name == "MertonJumpDiffusion"
        y = np.array([-0.9, 0.0])
        np.testing.assert_allclose(P.density(y), 0.1 * norm.pdf(y, -0.9, 0.45), rtol=1e-12)

    def test_default_half_width(self):
        P = LevyPIDE(0.05, 0.0, 0.2, no_jumps)
        assert P.default_half_width(0.01) == 1.5
        assert P.default_half_width(4.0) == pytest.approx(6 * 0.2 * 2.0)
        J = LevyPIDE(0.05, 0.0, 0.2, merton_density)
        assert J.default_half_width(4.0) > P.default_half_width(4.0)

    def test_grid_contains_strike_and_spot(self):
        P = LevyPIDE(0.05, 0.0, 0.2, merton_density)
        g = P._grid(95.0, 100.0, 1.0, 401, None)
        h = np.diff(g.nodes)
        np.testing.assert_allclose(h, h[0], rtol=1e-10)
        assert np.min(np.abs(g.nodes - math.log(100.0))) < 1e-12
        assert np.min(np.abs(g.nodes - math.log(95.0))) < 1e-12
        assert g.n == 401

    def test_far_field(self):
        P = LevyPIDE(0.05, 0.02, 0.2, no_jumps)
        x = np.log(np.array([10.0, 1000.0]))
        cl, cr = P.far_field(100.0, "call", american=False)
        assert cl(x, 1.0)[0] == 0.0
        assert cr(x, 1.0)[1] == pytest.approx(1000 * math.exp(-0.02) - 100 * math.exp(-0.05))
        pl, pr = P.far_field(100.0, "put", american=False)
        assert pl(x, 1.0)[0] == pytest.approx(100 * math.exp(-0.05) - 10 * math.exp(-0.02))
        assert pr(x, 1.0)[1] == 0.0
        al, _ = P.far_field(100.0, "put", american=True)
        assert al(x, 1.0)[0] == pytest.approx(90.0)
        _, ar = P.far_field(100.0, "call", american=True)
        assert ar(x, 1.0)[1] == pytest.approx(900.0)


class TestSolve:
    def test_reduces_to_black_scholes(self):
        P = LevyPIDE(0.05, 0.02, 0.25, no_jumps)
        res = P.european(100.0, 1.0, 100.0, "call", n_space=801, n_time=200)
        assert res.price == pytest.approx(bs_price(100.0, 100.0, 1.0, 0.05, 0.25, 0.02), abs=2e-4)
        info = res.solution.info
        assert info["jump_intensity"] == 0.0
        assert info["local_diffusion"] == pytest.approx(0.5 * 0.25**2)

    def test_put_call_parity(self):
        P = LevyPIDE(0.03, 0.01, 0.2, merton_density)
        c = P.european(110.0, 0.5, 100.0, "call", n_space=801).price
        p = P.european(110.0, 0.5, 100.0, "put", n_space=801).price
        assert c - p == pytest.approx(100 * math.exp(-0.005) - 110 * math.exp(-0.015), abs=1e-4)

    def test_generic_payoff_and_info(self):
        P = LevyPIDE(0.05, 0.0, 0.2, merton_density)
        grid = P._grid(100.0, 100.0, 1.0, 401, None)

        def zero(x, tau):
            return np.zeros_like(x)

        def disc(x, tau):
            return np.full_like(x, math.exp(-0.05 * tau))

        sol = P.solve(digital(100.0), 1.0, grid, zero, disc, n_time=100)
        val = sol.value(100.0)
        assert 0.0 < val < math.exp(-0.05)
        info = sol.info
        assert info["n_steps"] == 100
        assert info["fixed_point_iterations"] >= 100
        assert info["convection_scheme"] == "central"
        assert info["neglected_tail_mass"] < 1e-12
        # a constant payoff is a zero-coupon bond
        bond = Payoff(lambda s: np.ones_like(s), name="one")
        b = P.solve(bond, 1.0, grid, disc, disc, n_time=100)
        assert b.value(100.0) == pytest.approx(math.exp(-0.05), abs=1e-6)

    def test_grid_and_scheme_validation(self):
        P = LevyPIDE(0.05, 0.0, 0.2, merton_density)
        fl, fr = P.far_field(100.0, "call", american=False)
        bad = Grid1D(np.log(np.linspace(50.0, 200.0, 101)))
        with pytest.raises(ValueError, match="uniform"):
            P.solve(vanilla(100.0), 1.0, bad, fl, fr)
        grid = P._grid(100.0, 100.0, 1.0, 101, None)
        with pytest.raises(ValueError, match="convection_scheme"):
            P.solve(vanilla(100.0), 1.0, grid, fl, fr, convection_scheme="quick")

    def test_upwind_is_first_order(self):
        P = LevyPIDE(0.05, 0.0, 0.02, merton_density)
        ref = P.european(100.0, 1.0, 100.0, n_space=1601, n_time=400).price
        central = P.european(100.0, 1.0, 100.0, n_space=401, n_time=100)
        upwind = P.european(100.0, 1.0, 100.0, n_space=401, n_time=100, convection_scheme="upwind")
        auto = P.european(100.0, 1.0, 100.0, n_space=401, n_time=100, convection_scheme="auto")
        assert upwind.solution.info["convection_scheme"] == "upwind"
        assert auto.solution.info["convection_scheme"] in ("central", "upwind")
        assert abs(upwind.price - ref) > 3 * abs(central.price - ref)

    def test_greeks_signs(self):
        P = LevyPIDE(0.05, 0.0, 0.2, merton_density)
        c = P.european(100.0, 1.0, 100.0, "call", n_space=401)
        p = P.european(100.0, 1.0, 100.0, "put", n_space=401)
        assert 0 < c.delta < 1 and -1 < p.delta < 0
        assert c.gamma > 0 and c.gamma == pytest.approx(p.gamma, rel=1e-3)
        assert c.delta - p.delta == pytest.approx(1.0, abs=1e-3)
        assert c.theta < 0
        assert "PIDE European call" in c.method


class TestAmericanUnit:
    def test_methods_and_result(self):
        P = LevyPIDE(0.05, 0.0, 0.2, merton_density)
        res = P.american(100.0, 0.5, 95.0, "put", n_space=201, n_time=50)
        eu = P.european(100.0, 0.5, 95.0, "put", n_space=201, n_time=50)
        assert res.price >= eu.price
        assert res.exercise_boundary is res.solution.exercise_boundary
        assert res.exercise_boundary.taus.size == 51
        assert "Penalty" in res.solution.info["lcp"] or "penalty" in res.solution.info["lcp"]
        assert "American put (penalty)" in res.method
        with pytest.raises(ValueError):
            P.american(100.0, 0.5, 95.0, method="simplex", n_space=101)

    def test_value_dominates_obstacle(self):
        P = LevyPIDE(0.05, 0.0, 0.2, merton_density)
        res = P.american(100.0, 1.0, 100.0, "put", n_space=201, n_time=50, method="psor")
        s = np.exp(res.solution.grid.nodes)
        assert np.all(res.solution.values >= np.maximum(100.0 - s, 0.0) - 1e-8)

    def test_american_call_with_dividends_has_right_boundary(self):
        P = LevyPIDE(0.02, 0.08, 0.25, merton_density)
        am = P.american(100.0, 1.0, 100.0, "call", n_space=401, n_time=100)
        eu = P.european(100.0, 1.0, 100.0, "call", n_space=401, n_time=100)
        assert am.price > eu.price + 1e-3
        assert am.exercise_boundary.side == "right"
        assert np.nanmin(am.exercise_boundary.locations) > 100.0
