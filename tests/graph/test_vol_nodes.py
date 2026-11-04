"""Implied-volatility graph nodes."""

from __future__ import annotations

import pytest

from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ParameterNode
from spxa.graph.vol import surface_price_node, svi_total_variance_node
from spxa.market.curves import EquityForward
from spxa.vol import SSVI, PowerLawPhi, SVIRaw
from spxa.vol.surface import VolSurface

SVI = dict(a=0.02, b=0.1, rho=-0.4, m=0.05, sigma=0.2)
SURF = VolSurface(
    SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t),
    EquityForward(100.0, 0.03, dividend_yield=0.01),
)


@pytest.mark.parametrize("k", [-0.5, 0.05, 0.3])
def test_svi_value_and_gradient(k: float) -> None:
    params = {n: ParameterNode(v, n) for n, v in SVI.items()}
    node = svi_total_variance_node(*params.values(), k=k)
    g = ComputationalGraph("svi")
    g.add_output(node, name="w")
    assert g.evaluate()["w"] == pytest.approx(float(SVIRaw(**SVI).total_variance(k)), abs=1e-15)
    grads = GradientTape(g).gradient("w")
    for n, p in params.items():
        up, dn = dict(SVI), dict(SVI)
        up[n] += 1e-6
        dn[n] -= 1e-6
        fd = (node.op_fn(*up.values()) - node.op_fn(*dn.values())) / 2e-6
        assert grads[p] == pytest.approx(fd, rel=1e-7, abs=1e-10)


@pytest.mark.parametrize("dynamics", ["sticky_strike", "sticky_moneyness"])
def test_surface_price_node(dynamics: str) -> None:
    s, dv = ParameterNode(100.0, "spot"), ParameterNode(0.0, "dvol")
    node = surface_price_node(
        s, dv, surface=SURF, strike=90.0, maturity=1.0, option="put", dynamics=dynamics
    )
    g = ComputationalGraph("surface")
    g.add_output(node, name="price")
    assert g.evaluate()["price"] == pytest.approx(float(SURF.price(90.0, 1.0, "put")), abs=1e-12)
    grads = GradientTape(g).gradient("price")
    greeks = SURF.greeks(90.0, 1.0, "put", dynamics=dynamics)
    carry = float(SURF.forward(1.0)) / 100.0
    assert grads[s] == pytest.approx(float(greeks["delta"]) * carry, rel=1e-6)
    assert grads[dv] == pytest.approx(float(greeks["vega"]), rel=1e-6)
    shifted = node.op_fn(100.0, 0.01)
    assert shifted == pytest.approx(float(SURF.bumped(0.01).price(90.0, 1.0, "put")), abs=1e-12)


def test_validation() -> None:
    with pytest.raises(ValueError):
        surface_price_node(100.0, surface=SURF, strike=90.0, maturity=1.0, option="straddle")
