"""Unit tests for the Fokker-Planck discrete adjoint."""

import numpy as np
import pytest

from spxa.pde.fokker_planck import DensitySolution, FokkerPlanck1D
from spxa.pde.grid import Grid1D


@pytest.fixture
def ou():
    g = Grid1D.sinh(-3.0, 3.0, 121, center=0.0, alpha=1.0)
    return FokkerPlanck1D(lambda x, t: 1.2 * (0.3 - x), 0.5, g, time_homogeneous=True)


class TestForwardMatrix:
    def test_is_weighted_transpose_of_generator(self, ou):
        A, _ = ou.operator.assemble(0.0)
        P = ou.forward_matrix()
        V = ou.grid.cell_widths
        lhs = V[:, None] * P.toarray()
        np.testing.assert_allclose(lhs, A.toarray().T * V[None, :], atol=1e-12)

    def test_generator_rows_sum_to_zero(self, ou):
        A, _ = ou.operator.assemble(0.0)
        np.testing.assert_allclose(A @ np.ones(ou.grid.n), 0.0, atol=1e-9)
        # hence V-weighted column sums of P vanish: mass is conserved
        P = ou.forward_matrix()
        np.testing.assert_allclose(ou.grid.cell_widths @ P, 0.0, atol=1e-9)

    def test_cached_for_homogeneous_problems(self, ou):
        assert ou.forward_matrix() is ou.forward_matrix(1.0)
        g = Grid1D.uniform(-1.0, 1.0, 21)
        fp = FokkerPlanck1D(0.0, lambda x, t: 0.2 + 0.1 * t, g)
        assert fp.forward_matrix(0.0) is not fp.forward_matrix(0.0)


class TestSolve:
    def test_delta_has_unit_mass(self, ou):
        p = ou.delta(0.0)
        assert np.dot(p, ou.grid.cell_widths) == pytest.approx(1.0)
        assert np.count_nonzero(p) == 1

    def test_mass_conservation_and_positivity(self, ou):
        sol = ou.solve(0.0, 1.0, n_time=50, store_history=True)
        masses = sol.history @ ou.grid.cell_widths
        np.testing.assert_allclose(masses, 1.0, atol=1e-12)
        assert sol.density.min() > -1e-12
        assert sol.times.shape == (51,)
        assert sol.info["x0"] == pytest.approx(0.0)

    def test_density_input_and_validation(self, ou):
        p0 = np.exp(-(ou.grid.nodes**2) / 0.02)
        p0 /= np.dot(p0, ou.grid.cell_widths)
        sol = ou.solve(p0, 0.5, n_time=20)
        assert sol.mass() == pytest.approx(1.0, abs=1e-12)
        assert sol.info["x0"] is None
        with pytest.raises(ValueError):
            ou.solve(np.ones(3), 0.5)

    def test_density_solution_helpers(self):
        g = Grid1D.uniform(-1.0, 1.0, 201)
        p = np.full(201, 0.5)
        p[[0, -1]] = 0.5
        sol = DensitySolution(grid=g, density=p, time=1.0)
        assert sol.mass() == pytest.approx(1.0)
        assert sol.mean() == pytest.approx(0.0, abs=1e-12)
        assert sol.variance() == pytest.approx(1.0 / 3.0, abs=1e-4)
        assert sol.cdf(0.0) == pytest.approx(0.5)
        assert sol.pdf(0.3) == pytest.approx(0.5)
        np.testing.assert_allclose(sol.cdf(np.array([-1.0, 1.0])), [0.0, 1.0])
