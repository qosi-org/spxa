"""Tests for the projected SOR LCP solver."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from spxa.pde.american import PenaltyLCP, PSORLCP, complementarity_residual

from ._lcp_problems import model_lcp, qp_solution


class TestPSOR:
    def test_matches_quadratic_program(self):
        M, rhs, g, u_prev, dt, free = model_lcp()
        psor = PSORLCP(omega=1.3, tol=1e-13)
        psor.reset(rhs.size)
        u = psor.step(M, None, rhs, g, u_prev, dt, free)
        np.testing.assert_allclose(u, qp_solution(M, rhs, g, free), atol=1e-6)
        assert psor.iterations[0] > 1

    def test_agrees_with_penalty_on_nonsymmetric_problem(self):
        M, rhs, g, u_prev, dt, free = model_lcp(convection=3.0)
        u = PSORLCP(tol=1e-13).step(M, None, rhs, g, u_prev, dt, free)
        assert complementarity_residual(M, u, rhs, g, free) < 1e-8
        u_pen = PenaltyLCP(penalty=1e10).step(M, splu(M.tocsc()).solve, rhs, g, u_prev, dt, free)
        np.testing.assert_allclose(u, u_pen, atol=1e-7)

    def test_validation(self):
        with pytest.raises(ValueError):
            PSORLCP(omega=2.5)
        M = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 1.0]]))
        with pytest.raises(ValueError):
            PSORLCP().step(M, None, np.ones(2), np.zeros(2), np.zeros(2), 0.1, np.ones(2, bool))
        assert "PSOR" in PSORLCP().describe()
