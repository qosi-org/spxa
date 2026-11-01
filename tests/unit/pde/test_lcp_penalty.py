"""Tests for the penalty LCP solver."""

import numpy as np
import pytest
from scipy.sparse.linalg import splu

from spxa.pde.american import PenaltyLCP, complementarity_residual

from ._lcp_problems import model_lcp, qp_solution


class TestPenaltyLCP:
    def test_matches_quadratic_program(self):
        M, rhs, g, u_prev, dt, free = model_lcp()
        ref = qp_solution(M, rhs, g, free)
        pen = PenaltyLCP()
        pen.reset(rhs.size)
        u = pen.step(M, splu(M.tocsc()).solve, rhs, g, u_prev, dt, free)
        np.testing.assert_allclose(u, ref, atol=1e-6)
        assert complementarity_residual(M, u, rhs, g, free) < 1e-6
        n_active = np.sum(np.isclose(u[free], g[free], atol=1e-6))
        assert 0 < n_active < free.sum()  # both regions are present

    def test_nonsymmetric_residual(self):
        M, rhs, g, u_prev, dt, free = model_lcp(convection=3.0)
        u = PenaltyLCP().step(M, splu(M.tocsc()).solve, rhs, g, u_prev, dt, free)
        assert complementarity_residual(M, u, rhs, g, free) < 1e-6

    def test_records_iterations(self):
        M, rhs, g, u_prev, dt, free = model_lcp()
        pen = PenaltyLCP()
        pen.reset(rhs.size)
        pen.step(M, splu(M.tocsc()).solve, rhs, g, u_prev, dt, free)
        assert 1 <= pen.iterations[0] <= 10

    def test_without_active_set_is_linear_solve(self):
        M, rhs, g, u_prev, dt, free = model_lcp()
        lu = splu(M.tocsc())
        u = PenaltyLCP().step(M, lu.solve, rhs, g - 10.0, u_prev, dt, free)
        np.testing.assert_allclose(u, lu.solve(rhs))

    def test_residual_detects_violation(self):
        M, rhs, g, u_prev, dt, free = model_lcp()
        assert complementarity_residual(M, g - 0.1, rhs, g, free) > 1e-3

    def test_validation(self):
        with pytest.raises(ValueError):
            PenaltyLCP(penalty=-1)
        assert "penalty" in PenaltyLCP().describe()
