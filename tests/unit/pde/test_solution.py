"""Tests for PDESolution1D and ExerciseBoundary."""

import numpy as np
import pytest

from spxa.pde.boundary import Dirichlet
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D
from spxa.pde.solution import ExerciseBoundary, PDESolution1D
from spxa.pde.theta import ThetaSolver1D


class TestPDESolution1D:
    def test_s_space_greeks(self):
        g = Grid1D.sinh(0.0, 200.0, 201, center=100.0, alpha=20.0)
        sol = PDESolution1D(g, g.nodes**2, previous_values=g.nodes**2 - 0.1, dt_last=0.01)
        assert sol.value(101.3) == pytest.approx(101.3**2, rel=1e-10)
        assert sol.delta(101.3) == pytest.approx(2 * 101.3, rel=1e-8)
        assert sol.gamma(101.3) == pytest.approx(2.0, rel=1e-6)
        assert sol.theta(101.3) == pytest.approx(-10.0)
        out = sol.greeks(90.0)
        assert set(out) == {"value", "delta", "gamma", "theta"}
        np.testing.assert_allclose(sol([50.0, 150.0]), [2500.0, 22500.0], rtol=1e-10)

    def test_log_space_greeks(self):
        g = Grid1D.uniform(np.log(20.0), np.log(500.0), 801)
        sol = PDESolution1D(g, np.exp(2 * g.nodes), variable="log")
        assert sol.value(100.0) == pytest.approx(1e4, rel=1e-8)
        assert sol.delta(100.0) == pytest.approx(200.0, rel=3e-5)
        assert sol.gamma(100.0) == pytest.approx(2.0, rel=1e-4)
        np.testing.assert_allclose(sol.underlying_nodes, np.exp(g.nodes))
        with pytest.raises(ValueError):
            sol.value(-1.0)

    def test_errors(self):
        g = Grid1D.uniform(0.0, 1.0, 11)
        with pytest.raises(ValueError):
            PDESolution1D(g, np.zeros(10))
        with pytest.raises(ValueError):
            PDESolution1D(g, np.zeros(11), variable="sqrt")
        sol = PDESolution1D(g, np.zeros(11))
        with pytest.raises(ValueError):
            sol.value(2.0)
        with pytest.raises(ValueError):
            sol.theta(0.5)
        with pytest.raises(ValueError):
            sol.value_at(0.5, 0.1)
        assert "theta" not in sol.greeks(0.5)

    def test_from_theta_result_and_value_at(self):
        g = Grid1D.uniform(0.0, np.pi, 201)
        op = DiscreteOperator1D(LinearPDE1D(diffusion=1.0), g, Dirichlet(0.0), Dirichlet(0.0))
        res = ThetaSolver1D(op).solve(np.sin, TimeGrid.uniform(1.0, 100), store_history=True)
        sol = PDESolution1D.from_theta_result(res, label="heat", info={"pde": "u_t = u_xx"})
        assert sol.value(1.0) == pytest.approx(np.exp(-1.0) * np.sin(1.0), abs=1e-4)
        assert sol.value_at(1.0, 0.505) == pytest.approx(np.exp(-0.505) * np.sin(1.0), abs=1e-4)
        # u_tau = -u for this mode, so theta = +u
        assert sol.theta(1.0) == pytest.approx(np.exp(-1.0) * np.sin(1.0), rel=2e-2)
        with pytest.raises(ValueError):
            sol.value_at(1.0, 2.0)
        story = sol.__story__()
        assert "heat" in story and "Rannacher" in story and "u_t = u_xx" in story
        assert "PDESolution1D" in repr(sol)


class TestExerciseBoundary:
    def test_interpolation_and_calendar(self):
        eb = ExerciseBoundary(np.array([0.0, 0.5, 1.0]), np.array([100.0, 90.0, 85.0]))
        assert eb.at(0.25) == pytest.approx(95.0)
        t, loc = eb.calendar(1.0)
        np.testing.assert_allclose(t, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(loc, [85.0, 90.0, 100.0])

    def test_story_mentions_exercise(self):
        g = Grid1D.uniform(0.0, 1.0, 5)
        eb = ExerciseBoundary(np.array([0.0, 1.0]), np.array([np.nan, np.nan]))
        assert np.isnan(eb.at(0.5))
        sol = PDESolution1D(g, np.zeros(5), exercise_boundary=eb)
        assert "exercise" in sol.__story__()


class TestPDEPricingResult:
    def test_container(self):
        from spxa.pde.solution import PDEPricingResult

        g = Grid1D.uniform(0.0, 2.0, 21)
        sol = PDESolution1D(g, g.nodes**2, label="toy")
        eb = ExerciseBoundary(np.array([0.0, 1.0]), np.array([1.0, 0.8]))
        res = PDEPricingResult(1.0, 2.0, 2.0, -0.1, 1.0, sol, exercise_boundary=eb, method="m")
        assert float(res) == 1.0
        assert set(res.as_dict()) == {"price", "delta", "gamma", "theta"}
        story = res.__story__()
        assert "m at S = 1" in story and "exercise boundary today" in story
        assert "PDEPricingResult" in repr(res)
        res2 = PDEPricingResult(1.0, 2.0, 2.0, -0.1, 1.0, sol, vega=3.0, rho=4.0)
        assert res2.as_dict()["vega"] == 3.0 and "bump" in res2.__story__()
