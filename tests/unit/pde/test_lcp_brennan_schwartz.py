"""Tests for the Brennan-Schwartz direct LCP solver."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from spxa.pde.american import BrennanSchwartzLCP, PenaltyLCP, complementarity_residual

from ._lcp_problems import model_lcp, qp_solution


class TestBrennanSchwartz:
    def test_matches_quadratic_program(self):
        M, rhs, g, u_prev, dt, free = model_lcp()
        u = BrennanSchwartzLCP("left").step(M, None, rhs, g, u_prev, dt, free)
        np.testing.assert_allclose(u, qp_solution(M, rhs, g, free), atol=1e-6)

    def test_exact_agreement_with_tight_penalty(self):
        M, rhs, g, u_prev, dt, free = model_lcp(convection=1.0)
        u_bs = BrennanSchwartzLCP().step(M, None, rhs, g, u_prev, dt, free)
        assert complementarity_residual(M, u_bs, rhs, g, free) < 1e-12
        pen = PenaltyLCP(penalty=1e10, tol=1e-14)
        u_pen = pen.step(M, splu(M.tocsc()).solve, rhs, g, u_prev, dt, free)
        np.testing.assert_allclose(u_bs, u_pen, atol=1e-8)

    def test_right_side_by_symmetry(self):
        M, rhs, g, u_prev, dt, free = model_lcp()
        P = sp.csr_matrix(np.eye(rhs.size)[::-1])
        Mr = (P @ M @ P).tocsr()
        u_left = BrennanSchwartzLCP("left").step(M, None, rhs, g, u_prev, dt, free)
        u_right = BrennanSchwartzLCP("right").step(
            Mr, None, rhs[::-1], g[::-1], u_prev[::-1], dt, free[::-1]
        )
        np.testing.assert_allclose(u_right[::-1], u_left, atol=1e-12)

    def test_reduces_wide_boundary_rows(self):
        # zero-curvature closure in the last row: u_n - 2 u_{n-1} + u_{n-2} = 0
        M, rhs, g, u_prev, dt, free = model_lcp()
        M = sp.lil_matrix(M)
        n = rhs.size
        M[n - 1, n - 3 :] = [1.0, -2.0, 1.0]
        M = M.tocsr()
        rhs = rhs.copy()
        rhs[-1] = 0.0
        u = BrennanSchwartzLCP().step(M, None, rhs, g, u_prev, dt, free)
        assert u[-1] - 2 * u[-2] + u[-3] == pytest.approx(0.0, abs=1e-12)
        assert complementarity_residual(M, u, rhs, g, free) < 1e-10

    def test_rejects_wide_band(self):
        M = sp.csr_matrix(np.ones((5, 5)) + 5 * np.eye(5))
        with pytest.raises(ValueError):
            BrennanSchwartzLCP().step(
                M, None, np.ones(5), np.zeros(5), np.zeros(5), 0.1, np.ones(5, bool)
            )

    def test_validation(self):
        with pytest.raises(ValueError):
            BrennanSchwartzLCP("middle")
        assert "Brennan" in BrennanSchwartzLCP().describe()
