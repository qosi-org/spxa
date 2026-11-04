"""P&L explain with node deltas and gammas."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from spxa.rates import ESTR, ois_swap
from spxa.rates.bootstrap import bootstrap_curve
from spxa.rates.interpolation import LogLinearDiscount, PiecewiseCurve
from spxa.rates.risk import node_gamma, pnl_explain, with_node_values
from tests.unit.rates.test_bootstrap import MARKET, REF, _estr_helpers

T = np.array([1.0, 2.0, 5.0, 10.0])
CURVE = PiecewiseCurve(T, np.exp(-0.03 * T), reference_date=REF)


def test_zero_coupon_bond_expansion() -> None:
    # V = P(5) = exp(-x_3): delta -P, gamma P, remainder -P dx^3 / 6
    def pv(c):  # type: ignore[no-untyped-def]
        return float(c["discount"].discount(5.0))

    p, h = math.exp(-0.15), 1e-3
    x = -np.log(CURVE.node_discounts)
    after = with_node_values(CURVE, x + h * (np.arange(4) == 2))
    ex = pnl_explain(pv, {"discount": CURVE}, {"discount": after})
    assert ex.first_order == pytest.approx(-p * h, rel=1e-9)
    assert ex.second_order == pytest.approx(0.5 * p * h * h, rel=1e-5)
    assert ex.unexplained == pytest.approx(-p * h**3 / 6, rel=2e-3)
    G = node_gamma(pv, {"discount": CURVE})
    expected = np.zeros((4, 4))
    expected[2, 2] = p
    assert G == pytest.approx(expected, abs=1e-6)


def test_cross_gamma_of_a_product() -> None:
    # V = P(2) P(10) = exp(-x_1 - x_3): every second derivative on {1, 3} equals V
    def pv(c):  # type: ignore[no-untyped-def]
        d = c["discount"]
        return float(d.discount(2.0) * d.discount(10.0))

    G = node_gamma(pv, {"discount": CURVE})
    v = pv({"discount": CURVE})
    for i, j in [(1, 1), (1, 3), (3, 1), (3, 3)]:
        assert G[i, j] == pytest.approx(v, rel=1e-6)
    assert G[0, 0] == pytest.approx(0.0, abs=1e-6) and G[0, 3] == pytest.approx(0.0, abs=1e-6)


def test_swap_explain_after_a_market_move() -> None:
    hs = _estr_helpers(MARKET)
    before = bootstrap_curve(hs, REF, LogLinearDiscount())
    swap = ois_swap(REF, "8Y", 0.03, ESTR, notional=1e8)

    def pv(c):  # type: ignore[no-untyped-def]
        return swap.pv(c["discount"], c["discount"])

    rng = np.random.default_rng(4)
    moves = rng.normal(0.0, 1.0, len(hs)) + 1.0  # a parallel rise plus noise, in units
    left = []
    for size in (5e-4, 2.5e-3):  # 5 bp and 25 bp
        bumped = [replace(h, quote=h.quote + size * m) for h, m in zip(hs, moves)]
        after = bootstrap_curve(bumped, REF, LogLinearDiscount())
        ex = pnl_explain(pv, before.curves, after.curves)
        assert abs(ex.unexplained) < 1e-2 * abs(ex.second_order)
        assert abs(ex.second_order) < 0.1 * abs(ex.first_order)
        left.append(abs(ex.unexplained))
    # third order: five times the move leaves about 125 times the remainder
    assert 60 < left[1] / left[0] < 250


def test_validation() -> None:
    other = PiecewiseCurve(T[:-1], np.exp(-0.03 * T[:-1]), reference_date=REF)
    with pytest.raises(ValueError):
        pnl_explain(lambda c: 0.0, {"discount": CURVE}, {"discount": other})
