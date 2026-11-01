"""Unit tests for the two-dimensional operator and its ADI splitting."""

import numpy as np
import pytest

from spxa.pde.adi import (
    Dirichlet2D,
    DiscreteOperator2D,
    Grid2D,
    Linear2D,
    LinearPDE2D,
    Natural2D,
    Neumann2D,
)
from spxa.pde.grid import Grid1D

COEF = dict(axx=0.5, axy=0.3, ayy=0.4, bx=0.2, by=-0.1, c=-0.05)


def grid(nx=12, ny=9):
    return Grid2D(Grid1D.sinh(0.0, 1.0, nx, 0.4, 0.3), Grid1D.sinh(-1.0, 1.0, ny, 0.2, 0.5))


def poly(X, Y):
    """Quadratic without an xy term (its cross derivative vanishes)."""
    return 1.0 + X + 2.0 * Y + X**2 + 3.0 * Y**2


def poly_rhs(X, Y, coef):
    ux, uy = 1.0 + 2.0 * X, 2.0 + 6.0 * Y
    return (
        coef["axx"] * 2.0
        + coef["ayy"] * 6.0
        + coef["bx"] * ux
        + coef["by"] * uy
        + coef["c"] * poly(X, Y)
    )


def dirichlet_edges():
    return {
        "x_lo": Dirichlet2D(1.0),
        "x_hi": Dirichlet2D(lambda y, t: 2.0 + y + t),
        "y_lo": Dirichlet2D(0.0),
        "y_hi": Dirichlet2D(0.0),
    }


class TestGrid2D:
    def test_basic(self):
        g = grid(5, 4)
        assert g.shape == (5, 4) and g.n == 20
        X, Y = g.mesh()
        assert X.shape == (5, 4) and np.all(X[:, 0] == g.x.nodes) and np.all(Y[0] == g.y.nodes)
        assert g.index(2, 3) == 11
        assert "5x4" in repr(g)


class TestValidation:
    def test_edges(self):
        pde = LinearPDE2D(**COEF)
        e = dirichlet_edges()
        with pytest.raises(ValueError, match="missing"):
            DiscreteOperator2D(pde, grid(), {k: v for k, v in e.items() if k != "y_hi"})
        with pytest.raises(ValueError, match="unknown"):
            DiscreteOperator2D(pde, grid(), dict(e, z_lo=Natural2D()))
        with pytest.raises(TypeError):
            DiscreteOperator2D(pde, grid(), dict(e, y_hi="free"))
        with pytest.raises(ValueError, match="upwind"):
            DiscreteOperator2D(pde, grid(), e, upwind="always")
        with pytest.raises(ValueError, match="4 nodes"):
            DiscreteOperator2D(pde, grid(3, 9), e)

    def test_coefficients(self):
        X, Y = grid().mesh()
        with pytest.raises(ValueError, match="parabolic"):
            LinearPDE2D(axx=0.1, ayy=0.1, axy=0.5).coefficients(X, Y, 0.0)
        with pytest.raises(ValueError, match="non-negative"):
            LinearPDE2D(axx=-0.1).coefficients(X, Y, 0.0)
        assert LinearPDE2D(**COEF).time_homogeneous
        assert not LinearPDE2D(axx=lambda X, Y, t: 1.0 + t).time_homogeneous


class TestExactness:
    @pytest.mark.parametrize("upwind", ["none", "auto"])
    def test_interior_exact_for_quadratics(self, upwind):
        g = grid()
        coef = dict(COEF, bx=5.0, by=-4.0)  # strong convection triggers upwinding
        op = DiscreteOperator2D(LinearPDE2D(**coef), g, dirichlet_edges(), upwind=upwind)
        X, Y = g.mesh()
        got = op.apply(poly(X, Y), 0.0)
        np.testing.assert_allclose(got[1:-1, 1:-1], poly_rhs(X, Y, coef)[1:-1, 1:-1], atol=1e-10)
        np.testing.assert_allclose(got[0], 0.0)  # Dirichlet rows are zero
        if upwind == "auto":
            assert op.upwind_mask[0].any() or op.upwind_mask[1].any()

    def test_mixed_term_is_exact_for_xy(self):
        g = grid()
        op = DiscreteOperator2D(LinearPDE2D(axx=1.0, ayy=1.0, axy=0.7), g, dirichlet_edges())
        X, Y = g.mesh()
        got = op.apply(X * Y, 0.0)
        np.testing.assert_allclose(got[1:-1, 1:-1], 0.7, atol=1e-12)

    def test_neumann_and_natural_edges_exact(self):
        g = grid()
        coef = dict(COEF, axy=0.0)
        X, Y = g.mesh()
        edges = {
            "x_lo": Natural2D(),
            "x_hi": Neumann2D(lambda y, t: 1.0 + 2.0 * 1.0 + 0.0 * y),
            "y_lo": Neumann2D(lambda x, t: 2.0 + 6.0 * (-1.0) + 0.0 * x),
            "y_hi": Natural2D(),
        }
        op = DiscreteOperator2D(LinearPDE2D(**coef), g, edges)
        np.testing.assert_allclose(op.apply(poly(X, Y), 0.0), poly_rhs(X, Y, coef), atol=1e-9)
        assert op.dirichlet_nodes.size == 0

    def test_linear_edge_drops_normal_second_derivative(self):
        g = grid()
        coef = dict(COEF, axy=0.0)
        X, Y = g.mesh()
        e = dict(dirichlet_edges(), x_hi=Linear2D())
        op = DiscreteOperator2D(LinearPDE2D(**coef), g, e)
        got = op.apply(poly(X, Y), 0.0)
        expected = poly_rhs(X, Y, coef) - coef["axx"] * 2.0
        np.testing.assert_allclose(got[-1, 1:-1], expected[-1, 1:-1], atol=1e-9)


class TestSplitting:
    def test_structure(self):
        g = grid()
        op = DiscreteOperator2D(LinearPDE2D(**COEF), g, dirichlet_edges())
        (A0, A1, A2), (b0, b1, b2) = op.split(0.0)
        ny = g.y.n
        r1, c1 = A1.nonzero()
        assert np.all((c1 - r1) % ny == 0)  # x-direction couples equal j
        r2, c2 = A2.nonzero()
        assert np.all(r2 // ny == c2 // ny)  # y-direction couples equal i
        r0, _ = A0.nonzero()
        i0, j0 = r0 // ny, r0 % ny
        assert i0.min() >= 1 and i0.max() <= g.x.n - 2 and j0.min() >= 1 and j0.max() <= ny - 2
        X, Y = g.mesh()
        u = np.sin(X + Y).ravel()
        np.testing.assert_allclose(
            (A0 + A1 + A2) @ u + b0 + b1 + b2, op.apply(u, 0.0).ravel(), atol=1e-12
        )
        # derivative stencils annihilate constants: each direction carries c/2
        ones = np.ones(g.n)
        inner = np.zeros(g.shape, dtype=bool)
        inner[1:-1, 1:-1] = True
        for A in (A1, A2):
            np.testing.assert_allclose((A @ ones)[inner.ravel()], -0.025, atol=1e-10)
        np.testing.assert_allclose(A0 @ ones, 0.0, atol=1e-10)

    def test_dirichlet_values_and_sources(self):
        g = grid()
        e = dict(dirichlet_edges(), y_lo=Neumann2D(lambda x, t: t * np.ones_like(x)))
        op = DiscreteOperator2D(LinearPDE2D(axx=0.5, ayy=0.4, f=2.0), g, e)
        vals = op.dirichlet_values(0.5).reshape(g.shape)
        np.testing.assert_allclose(vals[-1], 2.0 + g.y.nodes + 0.5)
        assert vals[0, -1] == 1.0  # x edge owns the corner
        b0, b1, b2 = op.sources(0.5)
        assert np.all(b0.reshape(g.shape)[1:-1, 1:-1] == 2.0)
        assert np.all(b0[op.dirichlet_nodes] == 0.0)
        hy = g.y.nodes[1] - g.y.nodes[0]
        assert b2.reshape(g.shape)[3, 0] == pytest.approx(-2 * 0.4 * 0.5 / hy)
        assert np.allclose(op.sources(0.0)[2], 0.0)

    def test_time_dependent_coefficients(self):
        g = grid()
        pde = LinearPDE2D(axx=lambda X, Y, t: 0.1 + t * np.ones_like(X), ayy=0.2)
        op = DiscreteOperator2D(pde, g, dirichlet_edges())
        assert not op.time_homogeneous
        A_early = op.matrices(0.0)[1]
        A_late = op.matrices(1.0)[1]
        assert abs(A_late).sum() > abs(A_early).sum()
        assert "DiscreteOperator2D" in repr(op)
