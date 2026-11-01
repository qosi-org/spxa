"""Tests for spxa.pde.boundary."""

import numpy as np
import pytest

from spxa.pde.boundary import Dirichlet, Linear, Natural, Neumann


class TestBoundaryConditions:
    def test_dirichlet_constant_and_callable(self):
        assert Dirichlet(2.0)(0.7) == 2.0
        assert Dirichlet(lambda t: np.exp(-t))(1.0) == pytest.approx(np.exp(-1.0))
        assert Dirichlet().is_constraint

    def test_neumann(self):
        bc = Neumann(lambda t: 3.0 * t)
        assert bc(2.0) == 6.0
        assert not bc.is_constraint

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError):
            Dirichlet(lambda t: np.inf)(0.0)

    def test_linear_weights_reproduce_lines(self):
        for xb, x1, x2 in [(0.0, 1.0, 3.0), (10.0, 9.0, 7.5), (5.0, 4.9, 4.7)]:
            w1, w2 = Linear.weights(xb, x1, x2)
            f = lambda z: 2.0 - 0.3 * z  # noqa: E731
            assert w1 * f(x1) + w2 * f(x2) == pytest.approx(f(xb))
        assert Linear().is_constraint

    def test_natural_and_describe(self):
        assert not Natural().is_constraint
        for bc in (Dirichlet(), Neumann(), Linear(), Natural()):
            assert isinstance(bc.describe(), str)
