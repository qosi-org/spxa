"""Tests for the theta-scheme time stepper."""

import numpy as np
import pytest

from spxa.pde.boundary import Dirichlet, Linear, Natural, Neumann
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D
from spxa.pde.theta import ThetaSolver1D


def _heat_error(n_space, n_time, theta, T=0.5):
    grid = Grid1D.uniform(0.0, np.pi, n_space)
    op = DiscreteOperator1D(LinearPDE1D(diffusion=1.0), grid, Dirichlet(0.0), Dirichlet(0.0))
    res = ThetaSolver1D(op, theta=theta).solve(np.sin, TimeGrid.uniform(T, n_time))
    return np.max(np.abs(res.values - np.exp(-T) * np.sin(grid.nodes)))


class TestThetaScheme:
    def test_crank_nicolson_second_order(self):
        errs = [_heat_error(20 * 2**k + 1, 10 * 2**k, 0.5) for k in range(3)]
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 1.9)

    def test_backward_euler_first_order_in_time(self):
        errs = [_heat_error(801, 10 * 2**k, 1.0) for k in range(3)]
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(np.abs(orders - 1.0) < 0.15)

    def test_exact_for_linear_in_x_with_source_and_moving_dirichlet(self):
        # u = x + tau solves u_tau = u_xx + 1 exactly on any grid
        grid = Grid1D.sinh(0.0, 2.0, 31, center=0.7, alpha=0.2)
        pde = LinearPDE1D(diffusion=0.8, source=1.0)
        op = DiscreteOperator1D(pde, grid, Dirichlet(lambda t: t), Dirichlet(lambda t: 2.0 + t))
        res = ThetaSolver1D(op).solve(lambda x: x, TimeGrid.uniform(1.3, 7))
        np.testing.assert_allclose(res.values, grid.nodes + 1.3, atol=1e-12)

    def test_neumann_cosine_mode(self):
        grid = Grid1D.uniform(0.0, np.pi, 401)
        op = DiscreteOperator1D(LinearPDE1D(diffusion=1.0), grid, Neumann(0.0), Neumann(0.0))
        res = ThetaSolver1D(op).solve(np.cos, TimeGrid.uniform(1.0, 200))
        np.testing.assert_allclose(res.values, np.exp(-1.0) * np.cos(grid.nodes), atol=2e-5)

    def test_linear_boundary_exact_for_linear_data(self):
        # u = x e^{-tau} solves u_tau = u_xx - u; zero curvature holds at both ends
        grid = Grid1D.sinh(1.0, 3.0, 21, center=2.0, alpha=0.3)
        op = DiscreteOperator1D(LinearPDE1D(diffusion=1.0, reaction=-1.0), grid, Linear(), Linear())
        res = ThetaSolver1D(op, theta=1.0).solve(lambda x: x, TimeGrid.uniform(1.0, 4000))
        np.testing.assert_allclose(res.values, grid.nodes * np.exp(-1.0), rtol=2e-4)

    def test_time_dependent_reaction_second_order(self):
        grid = Grid1D.uniform(0.0, 1.0, 5)
        pde = LinearPDE1D(reaction=lambda x, t: t * np.ones_like(x))
        op = DiscreteOperator1D(pde, grid, Natural(), Natural())
        errs = []
        for n in (10, 20, 40):
            res = ThetaSolver1D(op).solve(np.ones(5), TimeGrid.uniform(1.0, n))
            errs.append(abs(res.values[2] - np.exp(0.5)))
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 1.95)

    def test_factorisation_cached_for_homogeneous_problems(self):
        grid = Grid1D.uniform(0.0, 1.0, 11)
        op = DiscreteOperator1D(LinearPDE1D(diffusion=1.0), grid, Dirichlet(0.0), Dirichlet(0.0))
        res = ThetaSolver1D(op, rannacher_steps=0).solve(np.zeros(11), TimeGrid.uniform(1.0, 50))
        assert res.info["n_factorisations"] == 1
        res = ThetaSolver1D(op).solve(np.zeros(11), TimeGrid.uniform(1.0, 50))
        assert res.info["n_factorisations"] == 2  # CN step + backward-Euler half step

    def test_history_and_callback(self):
        grid = Grid1D.uniform(0.0, 1.0, 11)
        op = DiscreteOperator1D(LinearPDE1D(diffusion=1.0), grid, Dirichlet(0.0), Dirichlet(0.0))
        seen = []
        res = ThetaSolver1D(op).solve(
            np.sin(np.pi * grid.nodes),
            TimeGrid.uniform(0.1, 5),
            store_history=True,
            callback=lambda k, t, u: seen.append(k),
        )
        assert res.history.shape == (6, 11)
        assert seen == list(range(6))
        np.testing.assert_array_equal(res.history[-1], res.values)
        np.testing.assert_array_equal(res.history[-2], res.previous_values)


def _digital_heat(n_space, n_time, rannacher_steps):
    """u_tau = u_xx with a unit step at x=0; exact u = N(x / sqrt(2 tau))."""
    from scipy.stats import norm

    grid = Grid1D.uniform(-6.0, 6.0, n_space)
    op = DiscreteOperator1D(LinearPDE1D(diffusion=1.0), grid, Dirichlet(0.0), Dirichlet(1.0))
    x = grid.nodes
    u0 = np.where(x > 0, 1.0, np.where(x == 0, 0.5, 0.0))
    res = ThetaSolver1D(op, rannacher_steps=rannacher_steps).solve(
        u0, TimeGrid.uniform(1.0, n_time)
    )
    near = np.abs(x) <= 1.0
    return np.max(np.abs(res.values - norm.cdf(x / np.sqrt(2.0)))[near])


class TestRannacher:
    def test_removes_crank_nicolson_oscillations(self):
        err_cn = _digital_heat(481, 10, 0)
        err_rn = _digital_heat(481, 10, 2)
        assert err_cn > 1e-2
        assert err_rn < 1e-3

    def test_second_order_with_discontinuous_data(self):
        errs = [_digital_heat(240 * 2**k + 1, 10 * 2**k, 2) for k in range(3)]
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 1.8)

    def test_restart_after_events(self):
        grid = Grid1D.uniform(0.0, 1.0, 11)
        op = DiscreteOperator1D(LinearPDE1D(diffusion=1.0), grid, Dirichlet(0.0), Dirichlet(0.0))
        tg = TimeGrid.uniform(1.0, 10, event_taus=[0.5])
        ev = {0.5: lambda u, x, t: u}
        res = ThetaSolver1D(op).solve(np.zeros(11), tg, events=ev)
        assert res.info["n_rannacher_steps"] == 4
        res = ThetaSolver1D(op, rannacher_after_events=False).solve(np.zeros(11), tg, events=ev)
        assert res.info["n_rannacher_steps"] == 2
        res = ThetaSolver1D(op, theta=1.0).solve(np.zeros(11), tg)
        assert res.info["n_rannacher_steps"] == 0

    def test_negative_steps_rejected(self):
        grid = Grid1D.uniform(0.0, 1.0, 5)
        op = DiscreteOperator1D(LinearPDE1D(diffusion=1.0), grid, Dirichlet(0.0), Dirichlet(0.0))
        with pytest.raises(ValueError):
            ThetaSolver1D(op, rannacher_steps=-1)


class TestEvents:
    def _op(self):
        grid = Grid1D.uniform(0.0, 1.0, 11)
        return DiscreteOperator1D(LinearPDE1D(), grid, Natural(), Natural())

    def test_hook_applied_at_event(self):
        tg = TimeGrid.uniform(1.0, 10, event_taus=[0.35])
        seen = []

        def hook(u, x, tau):
            seen.append(tau)
            return u + 1.0

        res = ThetaSolver1D(self._op()).solve(np.zeros(11), tg, events={0.35: hook})
        np.testing.assert_allclose(res.values, 1.0)
        assert seen == [pytest.approx(0.35)]
        assert res.info["n_events"] == 1

    def test_multiple_hooks_same_time(self):
        tg = TimeGrid.uniform(1.0, 4, event_taus=[0.5])
        hooks = [lambda u, x, t: u + 1.0, lambda u, x, t: 2.0 * u]
        res = ThetaSolver1D(self._op()).solve(np.zeros(11), tg, events=[(0.5, hooks)])
        np.testing.assert_allclose(res.values, 2.0)

    def test_event_validation(self):
        solver = ThetaSolver1D(self._op())
        with pytest.raises(ValueError):
            solver.solve(np.zeros(11), TimeGrid.uniform(1.0, 4), events={0.5: lambda u, x, t: u})
        tg = TimeGrid.uniform(1.0, 4, event_taus=[0.5])
        with pytest.raises(TypeError):
            solver.solve(np.zeros(11), tg, events={0.5: "not callable"})
        with pytest.raises(ValueError):
            solver.solve(np.zeros(11), tg, events={0.5: lambda u, x, t: u[:-1]})


class TestLCPHook:
    class Projection:
        def reset(self, n):
            self.calls = 0

        def step(self, M, solve, rhs, obstacle, guess, dt, free):
            self.calls += 1
            u = solve(rhs)
            u[free] = np.maximum(u[free], obstacle[free])
            return u

    def test_obstacle_respected(self):
        grid = Grid1D.uniform(0.0, 1.0, 21)
        op = DiscreteOperator1D(
            LinearPDE1D(diffusion=0.1, reaction=-1.0), grid, Dirichlet(0.0), Dirichlet(0.0)
        )
        lcp = self.Projection()
        g = 0.5 * np.sin(np.pi * grid.nodes)
        res = ThetaSolver1D(op).solve(g, TimeGrid.uniform(1.0, 10), lcp=lcp, obstacle=g)
        assert np.all(res.values >= g - 1e-14)
        assert lcp.calls == 12  # two Rannacher steps use two half steps each
        assert res.values[0] == pytest.approx(0.0, abs=1e-14)


class TestValidation:
    def test_bad_inputs(self):
        grid = Grid1D.uniform(0.0, 1.0, 5)
        op = DiscreteOperator1D(LinearPDE1D(diffusion=1.0), grid, Dirichlet(0.0), Dirichlet(0.0))
        with pytest.raises(TypeError):
            ThetaSolver1D("op")
        with pytest.raises(ValueError):
            ThetaSolver1D(op, theta=1.5)
        solver = ThetaSolver1D(op)
        with pytest.raises(TypeError):
            solver.solve(np.zeros(5), np.linspace(0, 1, 3))
        with pytest.raises(ValueError):
            solver.solve(np.zeros(4), TimeGrid.uniform(1.0, 2))
        with pytest.raises(ValueError):
            solver.solve(np.zeros(5), TimeGrid.uniform(1.0, 2), obstacle=np.zeros(5))
