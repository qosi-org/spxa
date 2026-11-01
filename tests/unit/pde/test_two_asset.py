"""Unit tests for the two-asset PDE set-up and the closed-form references."""

import numpy as np
import pytest

from spxa.pde.adi import Dirichlet2D, Linear2D, Natural2D
from spxa.pde.two_asset import TwoAssetBlackScholesPDE, TwoAssetPDEResult


def test_validation_and_repr():
    with pytest.raises(ValueError):
        TwoAssetBlackScholesPDE(0.05, 0.0, 0.2, 0.3)
    with pytest.raises(ValueError, match="rho"):
        TwoAssetBlackScholesPDE(0.05, 0.2, 0.2, 1.0)
    m = TwoAssetBlackScholesPDE(0.05, 0.2, 0.3, 0.3)
    assert "rho=0.3" in repr(m)
    with pytest.raises(ValueError):
        m.spread(5.0, 1.0, 100.0, 100.0, option="straddle")


def test_grid_and_edges():
    m = TwoAssetBlackScholesPDE(0.05, 0.2, 0.3, 0.3)
    g = m.grid(100.0, 50.0, 1.0, 31, 25, align=((110.0,), (500.0,)))
    assert g.shape == (31, 25)
    assert np.any(np.isclose(g.x.nodes, 100.0)) and np.any(np.isclose(g.x.nodes, 110.0))
    assert g.x.nodes[-1] == pytest.approx(100.0 * np.exp(1.2 + 0.05))
    op = m.operator(g)
    assert isinstance(op.edges["x_lo"], Natural2D) and isinstance(op.edges["y_hi"], Linear2D)
    op2 = m.operator(g, {"x_lo": Dirichlet2D(0.0)})
    assert op2.dirichlet_nodes.size == g.y.n


def test_result_fields():
    m = TwoAssetBlackScholesPDE(0.05, 0.2, 0.3, 0.3)
    res = m.exchange(0.5, 100.0, 100.0, n1=31, n2=31, n_time=10)
    assert isinstance(res, TwoAssetPDEResult) and float(res) == res.price
    assert res.delta[0] > 0 > res.delta[1] and res.spots == (100.0, 100.0)
    assert res.solution.variables == ("S1", "S2") and "exchange" in res.method
