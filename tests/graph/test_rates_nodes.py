"""Zero-curve graph nodes and their exact adjoints."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ParameterNode
from spxa.graph.rates import discounted_cashflows_node, par_swap_rate_node, zero_curve_weights
from spxa.rates import curve_price
from spxa.risk.instruments import InterestRateSwap

PILLARS = [1.0, 2.0, 5.0, 10.0]
ZEROS = [0.030, 0.032, 0.035, 0.037]


def graph(builder, **kw):
    params = [ParameterNode(z, f"z{T:g}") for z, T in zip(ZEROS, PILLARS)]
    node = builder(params, PILLARS, **kw)
    g = ComputationalGraph("rates")
    g.add_output(node, name="out")
    return g, node, params


def bumped(node, k: int, h: float) -> float:
    z = list(ZEROS)
    z[k] += h
    up = node.op_fn(*z)
    z[k] -= 2 * h
    return (up - node.op_fn(*z)) / (2 * h)


def test_weights() -> None:
    W = zero_curve_weights(PILLARS, [0.5, 1.0, 3.5, 10.0, 12.0])
    assert np.allclose(W.sum(axis=1), 1.0)
    assert W[0].tolist() == [1, 0, 0, 0] and W[4].tolist() == [0, 0, 0, 1]
    assert W[2].tolist() == pytest.approx([0, 0.5, 0.5, 0])
    with pytest.raises(ValueError):
        zero_curve_weights([2.0, 1.0], [1.0])


def test_cashflow_pv_and_key_rates() -> None:
    times, amounts = [0.5, 1.5, 3.0, 7.0, 12.0], [5.0, 5.0, 5.0, 5.0, 105.0]
    g, node, params = graph(discounted_cashflows_node, times=times, amounts=amounts)
    z = zero_curve_weights(PILLARS, times) @ np.array(ZEROS)
    want = float(np.dot(amounts, np.exp(-z * np.array(times))))
    assert g.evaluate()["out"] == pytest.approx(want, abs=1e-12)
    grads = GradientTape(g).gradient("out")
    for k, p in enumerate(params):
        assert grads[p] == pytest.approx(bumped(node, k, 1e-6), rel=1e-7)
    duration = -sum(a * t * math.exp(-zi * t) for a, t, zi in zip(amounts, times, z))
    assert sum(grads[p] for p in params) == pytest.approx(duration, rel=1e-12)


def test_par_rate_matches_the_risk_adapter_and_gradients() -> None:
    g, node, params = graph(par_swap_rate_node, maturity=7.25, frequency=0.5)
    grads = GradientTape(g).gradient("out")
    for k, p in enumerate(params):
        assert grads[p] == pytest.approx(bumped(node, k, 1e-6), rel=1e-6)
    flat = par_swap_rate_node([0.03] * 4, PILLARS, 5.0)
    swap = InterestRateSwap("S", "USD", 0.03, 5.0, 0.5)
    assert flat.op_fn(*[0.03] * 4) == pytest.approx(
        curve_price(swap, {"r": 0.03}).par_rate, abs=1e-15
    )


def test_validation() -> None:
    with pytest.raises(ValueError):
        discounted_cashflows_node(ZEROS, PILLARS, [1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        par_swap_rate_node(ZEROS[:3], PILLARS, 5.0)
    with pytest.raises(ValueError):
        par_swap_rate_node(ZEROS, PILLARS, 0.0)
