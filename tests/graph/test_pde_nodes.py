"""Tests for the PDE-backed graph nodes."""

import numpy as np
import pytest

from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ParameterNode
from spxa.graph.pde import bs_pde_price_node, heston_pde_price_node
from spxa.graph.pricing.greeks import compute_graph_greeks
from spxa.pde.reference import bs_greeks, bs_price, heston_cf, lewis_call_price


def bs_graph(style="european", option="call"):
    s = ParameterNode(100.0, "s0")
    v = ParameterNode(0.2, "volatility")
    r = ParameterNode(0.05, "rate")
    node = bs_pde_price_node(
        s,
        v,
        r,
        strike=105.0,
        maturity=1.0,
        option=option,
        style=style,
        dividend=0.01,
        n_space=401,
        n_time=200,
    )
    g = ComputationalGraph("bs_pde")
    g.add_output(node, name="price")
    return g, node, (s, v, r)


def test_price_and_tape_gradients():
    g, node, (s, v, r) = bs_graph()
    price = g.evaluate()["price"]
    assert price == pytest.approx(bs_price(100.0, 105.0, 1.0, 0.05, 0.2, 0.01), abs=2e-3)
    grads = GradientTape(g).gradient("price")
    ref = bs_greeks(100.0, 105.0, 1.0, 0.05, 0.2, 0.01)
    assert grads[s] == pytest.approx(ref["delta"], abs=1e-3)
    assert grads[v] == pytest.approx(ref["vega"], rel=2e-3)
    assert grads[r] == pytest.approx(ref["rho"], rel=2e-3)


def test_compute_graph_greeks_and_cache():
    g, node, _ = bs_graph()
    out = compute_graph_greeks(g)
    ref = bs_greeks(100.0, 105.0, 1.0, 0.05, 0.2, 0.01)
    assert out["gamma"] == pytest.approx(ref["gamma"], rel=1e-2)
    assert out["delta"] == pytest.approx(ref["delta"], abs=1e-3)
    n_cached = len(node.pde_pricer._cache)
    g.evaluate()
    assert len(node.pde_pricer._cache) == n_cached  # base point re-used


def test_parameters_flow_through_graph():
    g, _, _ = bs_graph()
    g.set_parameters({"volatility": 0.3})
    assert g.evaluate()["price"] == pytest.approx(
        bs_price(100.0, 105.0, 1.0, 0.05, 0.3, 0.01), abs=3e-3
    )


def test_american_node():
    g, _, (s, _, _) = bs_graph(style="american", option="put")
    price = g.evaluate()["price"]
    eu = bs_price(100.0, 105.0, 1.0, 0.05, 0.2, 0.01, option="put")
    assert price > eu
    assert -1.0 < GradientTape(g).gradient("price")[s] < 0.0


def test_validation():
    with pytest.raises(ValueError, match="style"):
        bs_pde_price_node(100.0, 0.2, 0.05, strike=100.0, maturity=1.0, style="asian")
    with pytest.raises(ValueError, match="maturity"):
        heston_pde_price_node(100.0, 0.04, 1.5, 0.04, 0.3, -0.9, 0.025, strike=100.0, maturity=0.0)


def test_heston_node():
    s = ParameterNode(100.0, "s0")
    v0 = ParameterNode(0.04, "v0")
    kappa = ParameterNode(1.5, "kappa")
    node = heston_pde_price_node(
        s, v0, kappa, 0.04, 0.3, -0.9, 0.025, strike=100.0, maturity=1.0, n_s=101, n_v=51, n_time=50
    )
    g = ComputationalGraph("heston")
    g.add_output(node, name="price")

    def lewis(S=100.0, v=0.04, k=1.5):
        return lewis_call_price(heston_cf(1.0, v, k, 0.04, 0.3, -0.9), S, 100.0, 1.0, 0.025)

    assert g.evaluate()["price"] == pytest.approx(lewis(), abs=5e-3)
    grads = GradientTape(g).gradient("price")
    assert grads[s] == pytest.approx((lewis(100.5) - lewis(99.5)) / 1.0, abs=2e-3)
    assert grads[v0] == pytest.approx((lewis(v=0.041) - lewis(v=0.039)) / 0.002, rel=1e-2)
    dk = (lewis(k=1.51) - lewis(k=1.49)) / 0.02
    assert grads[kappa] == pytest.approx(dk, rel=5e-2, abs=1e-2)
    assert np.isfinite(list(grads.values())).all()
