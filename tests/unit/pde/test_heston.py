"""Unit tests for the Heston PDE set-up."""

import numpy as np
import pytest

from spxa.pde.adi import Dirichlet2D, Natural2D, Neumann2D
from spxa.pde.heston import HestonPDE, HestonPDEResult
from spxa.zoo.beyond import HestonProcess


def model():
    return HestonPDE(0.03, 2.0, 0.04, 0.5, -0.7, q=0.01)


class TestConstruction:
    def test_validation(self):
        with pytest.raises(ValueError):
            HestonPDE(0.03, -1.0, 0.04, 0.5, -0.7)
        with pytest.raises(ValueError):
            HestonPDE(0.03, 1.0, 0.04, 0.0, -0.7)
        with pytest.raises(ValueError, match="rho"):
            HestonPDE(0.03, 1.0, 0.04, 0.5, -1.0)

    def test_feller_and_repr(self):
        assert HestonPDE(0.0, 2.0, 0.04, 0.3, 0.0).feller
        assert not HestonPDE(0.0, 2.0, 0.04, 0.5, 0.0).feller
        assert "kappa=2" in repr(model())

    def test_from_process(self):
        proc = HestonProcess(v0=0.05, kappa=3.0, theta=0.06, sigma_v=0.4, rho=-0.3)
        m = HestonPDE.from_process(proc, r=0.02, q=0.01)
        assert (m.kappa, m.theta, m.sigma, m.rho, m.r, m.q) == (3.0, 0.06, 0.4, -0.3, 0.02, 0.01)


class TestGridAndEdges:
    def test_grid(self):
        g = model().grid(100.0, 41, 21, spot=97.0)
        S, v = g.x.nodes, g.y.nodes
        assert S[0] == 0.0 and S[-1] == pytest.approx(800.0) and v[-1] == pytest.approx(5.0)
        assert np.any(np.isclose(S, 100.0, atol=1e-10)) and np.any(np.isclose(S, 97.0, atol=1e-10))
        h = np.diff(v)
        assert h[0] < h[-1] / 50  # clustered at zero variance
        hs = np.diff(S)
        assert hs[np.searchsorted(S, 100.0)] < hs[-1] / 5

    @pytest.mark.parametrize("option, american", [("call", False), ("put", False), ("put", True)])
    def test_edges(self, option, american):
        m = model()
        g = m.grid(100.0, 21, 11)
        op = m.operator(g, 100.0, option, american)
        e = op.edges
        assert isinstance(e["x_lo"], Dirichlet2D) and isinstance(e["x_hi"], Neumann2D)
        assert isinstance(e["y_lo"], Natural2D) and isinstance(e["y_hi"], Neumann2D)
        left = op.dirichlet_values(0.5).reshape(g.shape)[0]
        expected = 0.0 if option == "call" else (100.0 if american else 100.0 * np.exp(-0.015))
        np.testing.assert_allclose(left, expected)
        # no mixed coupling touches the edges
        A0 = op.matrices(0.0)[0]
        rows = A0.nonzero()[0]
        i, j = rows // g.y.n, rows % g.y.n
        assert i.min() >= 1 and j.min() >= 1


class TestPricing:
    def test_result(self):
        res = model().european(100.0, 0.5, 100.0, 0.04, n_s=41, n_v=21, n_time=20)
        assert isinstance(res, HestonPDEResult)
        assert float(res) == res.price and 0.0 < res.price < 100.0
        assert 0.0 < res.delta < 1.0 and res.gamma > 0.0 and res.dv > 0.0
        assert res.exercise_boundary is None
        assert "hundsdorfer_verwer" in res.method and res.solution.variables == ("S", "v")
        assert res.solution.info["feller"] is False

    def test_put_call_parity(self):
        m = model()
        kw = dict(n_s=81, n_v=41, n_time=40)
        c = m.european(100.0, 0.5, 105.0, 0.04, "call", **kw).price
        p = m.european(100.0, 0.5, 105.0, 0.04, "put", **kw).price
        assert c - p == pytest.approx(105.0 * np.exp(-0.005) - 100.0 * np.exp(-0.015), abs=5e-3)

    def test_input_validation(self):
        m = model()
        with pytest.raises(ValueError):
            m.european(100.0, 0.5, 100.0, -0.01)
        with pytest.raises(ValueError, match="outside"):
            m.european(100.0, 0.5, 900.0, 0.04, n_s=21, n_v=11, n_time=5)
        with pytest.raises(ValueError):
            m.european(100.0, 0.5, 100.0, 0.04, option="straddle")
        with pytest.raises(ValueError):
            m.european(100.0, 0.0, 100.0, 0.04)
