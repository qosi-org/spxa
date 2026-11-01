"""Tests for spxa.pde.operators: stencils and operator assembly."""

import numpy as np
import pytest

from spxa.pde.boundary import Dirichlet, Linear, Natural, Neumann
from spxa.pde.grid import Grid1D
from spxa.pde.operators import (
    DiscreteOperator1D,
    LinearPDE1D,
    central_first_weights,
    central_second_weights,
    derivative_matrices,
    one_sided_first_weights,
)


def _random_grid(n=15, seed=1):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.uniform(0.05, 0.3, n))


def _apply3(weights, u):
    lo, di, up = weights
    return lo * u[:-2] + di * u[1:-1] + up * u[2:]


class TestStencils:
    def test_first_derivative_exact_for_quadratics(self):
        x = _random_grid()
        u = 3.0 - 2.0 * x + 0.7 * x**2
        np.testing.assert_allclose(
            _apply3(central_first_weights(x), u), -2.0 + 1.4 * x[1:-1], atol=1e-10
        )

    def test_second_derivative_exact_for_quadratics(self):
        x = _random_grid()
        u = 3.0 - 2.0 * x + 0.7 * x**2
        np.testing.assert_allclose(_apply3(central_second_weights(x), u), 1.4, atol=1e-9)

    def test_weights_annihilate_constants(self):
        x = _random_grid()
        for w in (central_first_weights(x), central_second_weights(x)):
            np.testing.assert_allclose(sum(w), 0.0, atol=1e-10)

    @pytest.mark.parametrize("deriv", [1, 2])
    def test_second_order_on_sinh_grid(self, deriv):
        errs = []
        for n in (81, 161, 321):
            x = Grid1D.sinh(0.0, 3.0, n, center=1.0, alpha=0.3).nodes
            w = central_first_weights(x) if deriv == 1 else central_second_weights(x)
            exact = np.cos(x[1:-1]) if deriv == 1 else -np.sin(x[1:-1])
            errs.append(np.max(np.abs(_apply3(w, np.sin(x)) - exact)))
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 1.85)

    @pytest.mark.parametrize("pts", [(0.0, 0.3, 0.7), (1.0, 0.8, 0.5)])
    def test_one_sided_exact_for_quadratics(self, pts):
        x0, x1, x2 = pts
        w = one_sided_first_weights(x0, x1, x2)
        f = lambda z: 1.0 + 2.0 * z - 5.0 * z**2  # noqa: E731
        assert w[0] * f(x0) + w[1] * f(x1) + w[2] * f(x2) == pytest.approx(2.0 - 10.0 * x0)

    def test_derivative_matrices(self):
        x = _random_grid(12)
        D1, D2 = derivative_matrices(x)
        u = 1.0 + x - 0.5 * x**2
        np.testing.assert_allclose(D1 @ u, 1.0 - x, atol=1e-9)
        np.testing.assert_allclose(D2 @ u, -1.0, atol=1e-8)
        with pytest.raises(ValueError):
            derivative_matrices(np.array([0.0, 1.0]))


class TestLinearPDE1D:
    def test_time_homogeneity_inference(self):
        assert LinearPDE1D(diffusion=1.0).time_homogeneous
        assert not LinearPDE1D(diffusion=lambda x, t: x).time_homogeneous
        assert LinearPDE1D(diffusion=lambda x, t: x, time_homogeneous=True).time_homogeneous

    def test_negative_diffusion_rejected(self):
        with pytest.raises(ValueError):
            LinearPDE1D(diffusion=-1.0).coefficients(np.linspace(0, 1, 4), 0.0)


class TestDiscreteOperator1D:
    def setup_method(self):
        self.grid = Grid1D(1.0 + _random_grid(20, seed=3))
        self.pde = LinearPDE1D(
            diffusion=lambda x, t: x**2,
            convection=lambda x, t: x,
            reaction=-1.0,
            source=0.5,
            time_homogeneous=True,
        )

    def test_interior_exact_on_quadratics(self):
        op = DiscreteOperator1D(self.pde, self.grid, Dirichlet(0.0), Dirichlet(0.0), "central")
        x = self.grid.nodes
        Lu = op.apply(x**2, 0.0)
        np.testing.assert_allclose(Lu[1:-1], 3.0 * x[1:-1] ** 2 + 0.5, rtol=1e-9)
        assert Lu[0] == 0.0 and Lu[-1] == 0.0

    def test_neumann_rows_exact_on_quadratics(self):
        x = self.grid.nodes
        op = DiscreteOperator1D(
            self.pde, self.grid, Neumann(2.0 * x[0]), Neumann(2.0 * x[-1]), "central"
        )
        Lu = op.apply(x**2, 0.0)
        np.testing.assert_allclose(Lu, 3.0 * x**2 + 0.5, rtol=1e-9)

    def test_natural_rows_exact_on_quadratics(self):
        pde = LinearPDE1D(diffusion=0.0, convection=lambda x, t: 2.0 - x, reaction=-0.3)
        x = self.grid.nodes
        op = DiscreteOperator1D(pde, self.grid, Natural(), Natural(), "central")
        u = 1.0 + x - x**2
        np.testing.assert_allclose(op.apply(u, 0.0), (2.0 - x) * (1 - 2 * x) - 0.3 * u, atol=1e-9)

    def test_constraints(self):
        x = self.grid.nodes
        op = DiscreteOperator1D(self.pde, self.grid, Dirichlet(lambda t: 1.0 + t), Linear())
        C, g = op.constraints(2.0)
        assert g[0] == 3.0 and g[-1] == 0.0
        line = 4.0 - 0.5 * x
        assert (C @ line)[-1] == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_array_equal(op.constraint_rows, [0, x.size - 1])
        u = op.enforce_constraints(line + 1.0, 2.0)
        assert u[0] == 3.0 and u[-1] == pytest.approx(line[-1] + 1.0)
        L, s = op.assemble(0.0)
        assert L.getrow(0).nnz == 0 and s[0] == 0.0 and s[-1] == 0.0
        assert not op.time_homogeneous and op.matrix_time_homogeneous

    def test_auto_upwinding_gives_m_matrix(self):
        grid = Grid1D.uniform(0.0, 1.0, 51)
        pde = LinearPDE1D(diffusion=1e-4, convection=lambda x, t: np.sin(6 * x))
        op = DiscreteOperator1D(pde, grid, Dirichlet(0.0), Dirichlet(0.0), "auto")
        L, _ = op.assemble(0.0)
        off = L - L.multiply(np.eye(51))
        assert off.min() >= 0.0
        assert op.upwinded_nodes().sum() > 40
        op_c = DiscreteOperator1D(pde, grid, Dirichlet(0.0), Dirichlet(0.0), "central")
        Lc, _ = op_c.assemble(0.0)
        assert (Lc - Lc.multiply(np.eye(51))).min() < 0.0
        assert not op_c.upwinded_nodes().any()

    def test_auto_keeps_central_when_diffusion_dominates(self):
        grid = Grid1D.uniform(0.0, 1.0, 51)
        pde = LinearPDE1D(diffusion=1.0, convection=3.0)
        op = DiscreteOperator1D(pde, grid, Dirichlet(0.0), Dirichlet(0.0), "auto")
        assert not op.upwinded_nodes().any()
        assert DiscreteOperator1D(pde, grid, Natural(), Natural(), "upwind").upwinded_nodes().any()

    def test_upwind_is_first_order_consistent(self):
        x = self.grid.nodes
        pde = LinearPDE1D(convection=lambda z, t: np.where(z > 2.0, 1.0, -1.0))
        op = DiscreteOperator1D(pde, self.grid, Dirichlet(0.0), Dirichlet(0.0), "upwind")
        Lu = op.apply(3.0 * x + 1.0, 0.0)  # exact for linear functions
        np.testing.assert_allclose(Lu[1:-1], 3.0 * np.where(x[1:-1] > 2.0, 1.0, -1.0))

    def test_validation_and_repr(self):
        with pytest.raises(ValueError):
            DiscreteOperator1D(self.pde, self.grid, Natural(), Natural(), "weird")
        with pytest.raises(TypeError):
            DiscreteOperator1D(self.pde, self.grid, "dirichlet", Natural())
        assert "Natural" in repr(DiscreteOperator1D(self.pde, self.grid, Natural(), Natural()))
