"""Tests for the Ikonen-Toivanen operator splitting and the LCP factory."""

import numpy as np
import pytest
from scipy.sparse.linalg import splu

from spxa.pde.american import (
    BrennanSchwartzLCP,
    IkonenToivanenLCP,
    PenaltyLCP,
    PSORLCP,
    make_lcp,
)

from ._lcp_problems import model_lcp


class TestIkonenToivanen:
    def test_feasible_and_complementary_multiplier(self):
        M, rhs, g, u_prev, dt, free = model_lcp()
        lu = splu(M.tocsc())
        it = IkonenToivanenLCP()
        it.reset(rhs.size)
        u = u_prev
        for _ in range(5):
            u = it.step(M, lu.solve, u.copy(), g, u, dt, free)
        assert np.all(u[free] >= g[free] - 1e-14)
        assert np.all(it.multiplier >= 0.0)
        assert np.any(it.multiplier > 0.0)
        on = it.multiplier > 0
        np.testing.assert_allclose(u[on], g[on])

    def test_auto_reset_on_size_change(self):
        M, rhs, g, u_prev, dt, free = model_lcp(n=20)
        it = IkonenToivanenLCP()
        u = it.step(M, splu(M.tocsc()).solve, rhs, g, u_prev, dt, free)
        assert u.shape == rhs.shape
        assert "Ikonen" in it.describe()


class TestFactory:
    def test_make_lcp(self):
        assert isinstance(make_lcp("penalty"), PenaltyLCP)
        assert isinstance(make_lcp("psor"), PSORLCP)
        assert make_lcp("brennan-schwartz", side="right").side == "right"
        assert isinstance(make_lcp("ikonen_toivanen"), IkonenToivanenLCP)
        assert isinstance(make_lcp("IT"), IkonenToivanenLCP)
        assert isinstance(BrennanSchwartzLCP(), BrennanSchwartzLCP)
        with pytest.raises(ValueError):
            make_lcp("magic")
