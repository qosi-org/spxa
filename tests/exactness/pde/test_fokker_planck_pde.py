"""Exactness tests: transition densities from the Fokker-Planck solver."""

import math

import numpy as np
import pytest

from spxa.pde.fokker_planck import FokkerPlanck1D
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.theta import ThetaSolver1D
from spxa.zoo.beyond import CIRProcess

pytestmark = pytest.mark.exactness


def gaussian(x, m, v):
    return np.exp(-((x - m) ** 2) / (2 * v)) / math.sqrt(2 * math.pi * v)


class TestOrnsteinUhlenbeck:
    K, TH, S, X0, T = 1.5, 0.5, 0.4, 0.0, 1.0

    def solve(self, n):
        g = Grid1D.sinh(-2.5, 3.0, n, center=self.X0, alpha=1.0)
        fp = FokkerPlanck1D(lambda x, t: self.K * (self.TH - x), self.S, g)
        return fp.solve(self.X0, self.T, n_time=n // 2)

    def moments(self):
        m = self.TH + (self.X0 - self.TH) * math.exp(-self.K * self.T)
        v = self.S**2 / (2 * self.K) * (1 - math.exp(-2 * self.K * self.T))
        return m, v

    def test_density_mean_variance(self):
        sol = self.solve(401)
        m, v = self.moments()
        assert sol.mass() == pytest.approx(1.0, abs=1e-12)
        assert sol.mean() == pytest.approx(m, abs=5e-6)
        assert sol.variance() == pytest.approx(v, abs=1e-7)
        np.testing.assert_allclose(sol.density, gaussian(sol.grid.nodes, m, v), atol=3e-4)

    def test_second_order_convergence(self):
        m, v = self.moments()
        errs = []
        for n in (201, 401, 801):
            sol = self.solve(n)
            errs.append(
                np.dot(np.abs(sol.density - gaussian(sol.grid.nodes, m, v)), sol.grid.cell_widths)
            )
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 1.9)

    def test_cdf(self):
        sol = self.solve(401)
        m, v = self.moments()
        from scipy.stats import norm

        x = np.array([-0.3, 0.0, 0.3, 0.6])
        np.testing.assert_allclose(sol.cdf(x), norm.cdf(x, m, math.sqrt(v)), atol=2e-3)


class TestDuality:
    def test_forward_and_backward_agree_to_rounding(self):
        g = Grid1D.sinh(-2.5, 3.0, 201, center=0.0, alpha=1.0)
        fp = FokkerPlanck1D(lambda x, t: 1.5 * (0.5 - x) + 0.1 * np.sin(x), 0.4, g)
        sol = fp.solve(0.0, 1.0, n_time=50, rannacher_steps=0)
        payoffs = [np.maximum(g.nodes - 0.3, 0.0), (g.nodes > 0.2).astype(float), np.cos(g.nodes)]
        j = int(np.argmin(np.abs(g.nodes)))
        for pay in payoffs:
            back = ThetaSolver1D(fp.operator, theta=0.5, rannacher_steps=0).solve(
                pay, TimeGrid.uniform(1.0, 50)
            )
            fwd = float(np.dot(sol.density * g.cell_widths, pay))
            assert fwd == pytest.approx(back.values[j], abs=1e-13)


class TestOtherModels:
    def test_cir_noncentral_chi_square(self):
        k, th, s, r0 = 0.8, 0.05, 0.15, 0.03
        cir = CIRProcess(kappa=k, theta=th, sigma=s, r0=r0)
        g = Grid1D.sinh(0.0, 0.4, 401, center=r0, alpha=0.05)
        fp = FokkerPlanck1D(
            lambda x, t: k * (th - x), lambda x, t: s * np.sqrt(np.maximum(x, 0.0)), g
        )
        sol = fp.solve(r0, 1.0, n_time=200)
        x = np.array([0.01, 0.03, 0.05, 0.08, 0.12])
        np.testing.assert_allclose(
            sol.pdf(x), cir.transition_density(x, 1.0, r0), rtol=3e-4, atol=2e-3
        )
        assert sol.mass() == pytest.approx(1.0, abs=1e-12)
        assert sol.mean() == pytest.approx(th + (r0 - th) * math.exp(-k), abs=2e-6)

    def test_geometric_brownian_motion_in_log_space(self):
        mu, sig, T = 0.05, 0.3, 2.0
        g = Grid1D.uniform(-2.5, 2.5, 801)
        fp = FokkerPlanck1D(mu - 0.5 * sig**2, sig, g)
        sol = fp.solve(0.0, T, n_time=400)
        m, v = (mu - 0.5 * sig**2) * T, sig**2 * T
        np.testing.assert_allclose(sol.density, gaussian(g.nodes, m, v), atol=4e-5)
        assert sol.expectation(np.exp) == pytest.approx(math.exp(mu * T), rel=2e-6)

    def test_time_dependent_volatility(self):
        g = Grid1D.uniform(-3.0, 3.0, 601)
        fp = FokkerPlanck1D(0.1, lambda x, t: 0.2 + 0.3 * t, g)
        sol = fp.solve(0.0, 1.0, n_time=200)
        var = 0.04 + 0.2 * 0.3 / 1 + 0.09 / 3  # int_0^1 (0.2 + 0.3 t)^2 dt
        np.testing.assert_allclose(sol.density, gaussian(g.nodes, 0.1, var), atol=2e-4)
        assert sol.variance() == pytest.approx(var, abs=5e-6)
