"""Unit tests for ADISolver and PDESolution2D."""

import numpy as np
import pytest

from spxa.pde.adi import (
    ADISolver,
    Dirichlet2D,
    DiscreteOperator2D,
    Grid2D,
    LinearPDE2D,
    PDESolution2D,
)
from spxa.pde.grid import Grid1D


def make_op(nx=15, ny=12):
    g = Grid2D(Grid1D.uniform(0.0, 1.0, nx), Grid1D.uniform(0.0, 2.0, ny))
    edges = {k: Dirichlet2D(0.0) for k in ("x_lo", "x_hi", "y_lo", "y_hi")}
    pde = LinearPDE2D(axx=0.1, ayy=0.05, axy=0.02, bx=0.1, c=-0.03)
    return DiscreteOperator2D(pde, g, edges), g


class TestADISolver:
    def test_schemes(self):
        op, _ = make_op()
        assert ADISolver(op, "hv").theta == pytest.approx(0.5 + np.sqrt(3) / 6)
        assert ADISolver(op, "mcs").theta == pytest.approx(1 / 3)
        assert ADISolver(op, "cs").scheme == "craig_sneyd"
        assert ADISolver(op, "Douglas", theta=0.7).theta == 0.7
        with pytest.raises(ValueError, match="unknown"):
            ADISolver(op, "peaceman")
        with pytest.raises(ValueError):
            ADISolver(op, damping_steps=-1)
        with pytest.raises(ValueError):
            ADISolver(op).solve(np.zeros(op.grid.shape), 0.0)

    def test_heat_decay_and_info(self):
        op, g = make_op()
        X, Y = g.mesh()
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y / 2)
        seen = []
        res = ADISolver(op, "hv", damping_steps=1).solve(
            u0, 1.0, 20, store_history=True, callback=lambda k, t, u: seen.append(k)
        )
        assert seen == list(range(21))
        assert len(res.history) == 21
        assert np.abs(res.values).max() < np.abs(u0).max()
        assert np.allclose(res.values[0], 0.0) and np.allclose(res.values[:, -1], 0.0)
        info = res.info
        assert info["scheme"] == "hundsdorfer_verwer" and info["n_steps"] == 20
        # 2 directions x (damping theta=1 half step, HV step); factorisations are cached
        assert info["n_factorisations"] == 4
        assert res.multiplier is None and not info["american"]

    def test_obstacle_ikonen_toivanen(self):
        op, g = make_op()
        X, Y = g.mesh()
        obstacle = 0.3 * np.sin(np.pi * X) * np.sin(np.pi * Y / 2)
        res = ADISolver(op, "mcs").solve(np.zeros(g.shape), 1.0, 20, obstacle=obstacle)
        assert np.all(res.values >= obstacle - 1e-12)
        assert np.all(res.multiplier >= 0.0)
        # the decaying solution would fall below the obstacle: it binds somewhere
        assert np.any(np.isclose(res.values, obstacle) & (obstacle > 0))
        assert res.info["american"]


class TestPDESolution2D:
    def test_spline_reproduces_cubics(self):
        g = Grid2D(Grid1D.sinh(0.0, 2.0, 15, 1.0, 0.5), Grid1D.uniform(-1.0, 1.0, 11))
        X, Y = g.mesh()
        sol = PDESolution2D(g, X**3 * Y + Y**2, previous_values=X**3 * Y, dt_last=0.5)
        assert sol.value(1.3, 0.2) == pytest.approx(1.3**3 * 0.2 + 0.04, abs=1e-10)
        assert sol.derivative(1.3, 0.2, dx=1) == pytest.approx(3 * 1.3**2 * 0.2, abs=1e-9)
        assert sol.derivative(1.3, 0.2, dx=1, dy=1) == pytest.approx(3 * 1.3**2, abs=1e-8)
        assert sol.derivative(1.3, 0.2, dy=2) == pytest.approx(2.0, abs=1e-8)
        np.testing.assert_allclose(sol.value(np.array([0.5, 1.0]), 0.0), [0.0, 0.0], atol=1e-12)
        assert sol.theta(1.3, 0.2) == pytest.approx(-(0.04) / 0.5, abs=1e-10)
        with pytest.raises(ValueError, match="outside"):
            sol.value(2.5, 0.0)
        assert "PDESolution2D" in repr(sol)

    def test_theta_requires_previous(self):
        g = Grid2D(Grid1D.uniform(0.0, 1.0, 5), Grid1D.uniform(0.0, 1.0, 5))
        with pytest.raises(ValueError, match="previous"):
            PDESolution2D(g, np.zeros(g.shape)).theta(0.5, 0.5)
