"""spxa.vol on curves bootstrapped by spxa.rates, and the cross-package bridges."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.graph.rates import discounted_cashflows_node
from spxa.market.curves import EquityForward
from spxa.rates import ESTR, MonotoneConvex, SwapHelper, bootstrap_curve, ois_swap
from spxa.risk.instruments import EuropeanOption
from spxa.vol import SSVI, PowerLawPhi, surface_price
from spxa.vol.surface import VolSurface

REF = "2024-01-15"
OIS = {"3M": 3.905, "6M": 3.83, "1Y": 3.56, "2Y": 3.02, "3Y": 2.78, "5Y": 2.60,
       "7Y": 2.56, "10Y": 2.58, "15Y": 2.62, "20Y": 2.58, "30Y": 2.42}  # fmt: skip
CURVE = bootstrap_curve(
    [SwapHelper(ois_swap(REF, t, 0.0, ESTR), q / 100) for t, q in OIS.items()],
    REF,
    MonotoneConvex(),
).curve
SSVI_MODEL = SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t)
SURF = VolSurface(SSVI_MODEL, EquityForward(4800.0, CURVE, dividend_yield=0.013))


@pytest.mark.parametrize("T", [0.25, 1.0, 3.5, 10.0])
def test_forwards_and_parity_on_the_bootstrapped_curve(T: float) -> None:
    P = float(CURVE.discount(T))
    assert float(SURF.discount(T)) == P
    assert float(SURF.forward(T)) == pytest.approx(4800.0 * math.exp(-0.013 * T) / P, rel=1e-14)
    for K in (3600.0, 4800.0, 6000.0):
        c, p = float(SURF.price(K, T, "call")), float(SURF.price(K, T, "put"))
        assert c - p == pytest.approx(P * (float(SURF.forward(T)) - K), abs=1e-9)


@pytest.mark.parametrize("shift", [-0.02, 0.035])
def test_rate_shock_equals_a_shifted_curve_under_sticky_moneyness(shift: float) -> None:
    opt = EuropeanOption("P", "SPX", 4300.0, 2.0, is_call=False, notional=5.0)
    base = {"SPX": 4800.0, "SPX_SURFACE": SURF, "r": 0.03}
    stressed = dict(base, r=0.03 + shift)
    moved = VolSurface(
        SSVI_MODEL, EquityForward(4800.0, CURVE.shifted(shift), dividend_yield=0.013)
    )
    got = surface_price(opt, stressed, base, dynamics="sticky_moneyness").price
    want = surface_price(opt, {"SPX_SURFACE": moved}, dynamics="sticky_moneyness").price
    assert got == pytest.approx(want, rel=1e-12)


def test_zero_curve_node_reprices_the_bootstrapped_nodes() -> None:
    times = np.asarray(CURVE.node_times, float)
    zeros = np.asarray(CURVE.node_zero_rates, float)
    node = discounted_cashflows_node(list(zeros), list(times), list(times), [1.0] * len(times))
    assert node.op_fn(*zeros) == pytest.approx(float(np.sum(CURVE.discount(times))), abs=1e-13)
