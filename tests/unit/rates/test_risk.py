"""Node and quote (par-point) deltas from fitted curves."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spxa.market.curves import FlatCurve
from spxa.rates import ESTR, EURIBOR6M, ois_swap, turn_of_year, vanilla_swap
from spxa.rates.bootstrap import CurveSpec, bootstrap_curve, fit_curves
from spxa.rates.interpolation import LogLinearDiscount, MonotoneConvex
from spxa.rates.risk import (
    curve_nodes,
    node_deltas,
    quote_deltas,
    quote_jacobian,
    with_node_values,
)
from tests.unit.rates.test_bootstrap import MARKET, REF, _estr_helpers
from tests.unit.rates.test_global_fit import _projection_helpers

HS = _estr_helpers(MARKET)
LL = bootstrap_curve(HS, REF, LogLinearDiscount())


def _pv(swap):  # type: ignore[no-untyped-def]
    return lambda c: swap.pv(c["discount"], c["discount"])


def test_par_swap_risk_sits_on_its_own_quote() -> None:
    # PV = (S - K) A; at par only S moves, and S is the helper's own quote
    i = 6  # the 5Y OIS
    h = HS[i]
    swap = h.swap.with_fixed_rate(h.quote)
    risk = quote_deltas(_pv(swap), LL)
    annuity = swap.annuity(LL.curve)
    expected = np.zeros(len(HS))
    expected[i] = annuity
    assert risk.deltas == pytest.approx(expected, abs=1e-9)
    assert risk.per_bp[i] == pytest.approx(1e-4 * annuity, rel=1e-8)


def test_matches_rebootstrap() -> None:
    swap = ois_swap(REF, "6Y", 0.031, ESTR)  # off-market, between pillars
    pv = _pv(swap)
    risk = quote_deltas(pv, LL)
    h = 1e-6
    for i in (5, 6, 7):
        up = [replace(x, quote=x.quote + h) if k == i else x for k, x in enumerate(HS)]
        dn = [replace(x, quote=x.quote - h) if k == i else x for k, x in enumerate(HS)]
        fd = (
            pv(bootstrap_curve(up, REF, LogLinearDiscount()).curves)
            - pv(bootstrap_curve(dn, REF, LogLinearDiscount()).curves)
        ) / (2 * h)
        assert risk.deltas[i] == pytest.approx(fd, rel=1e-6, abs=1e-10)
    # a 6Y swap reads nothing beyond the 7Y pillar
    assert np.all(np.abs(risk.deltas[8:]) < 1e-10)
    # total = response to moving every quote together
    up = [replace(x, quote=x.quote + h) for x in HS]
    dn = [replace(x, quote=x.quote - h) for x in HS]
    fd = (
        pv(bootstrap_curve(up, REF, LogLinearDiscount()).curves)
        - pv(bootstrap_curve(dn, REF, LogLinearDiscount()).curves)
    ) / (2 * h)
    assert risk.total == pytest.approx(fd, rel=1e-6)


def test_jacobian_matches_bootstrap_result() -> None:
    J = quote_jacobian(HS, LL.curves, ["discount"])
    assert J == pytest.approx(LL.jacobian(), abs=1e-9)
    risk = quote_deltas(_pv(HS[4].swap.with_fixed_rate(0.03)), LL, jacobian=J)
    again = quote_deltas(_pv(HS[4].swap.with_fixed_rate(0.03)), helpers=HS, curves=LL.curves)
    assert risk.deltas == pytest.approx(again.deltas, abs=1e-12)
    assert "SwapHelper" in risk.table() and HS[4].pillar.isoformat() in risk.table()


def test_node_values_round_trip_with_turns() -> None:
    res = bootstrap_curve(HS, REF, LogLinearDiscount(), spreads=[turn_of_year(2024, 0.002)])
    t, x = curve_nodes(res.curve)
    again = with_node_values(res.curve, x)
    grid = np.linspace(0.01, 30, 300)
    assert again.discount(grid) == pytest.approx(res.curve.discount(grid), rel=1e-15)
    nd = node_deltas(lambda c: float(-np.log(c["discount"].discount(t[3]))), res.curves)
    expected = np.zeros(t.size)
    expected[3] = 1.0  # x_3 itself
    assert nd.deltas == pytest.approx(expected, abs=1e-8)
    assert nd.for_curve("discount") == pytest.approx(nd.deltas)


@pytest.fixture(scope="module")
def joint():  # type: ignore[no-untyped-def]
    return fit_curves(
        [CurveSpec("discount", HS), CurveSpec("EUR-6M", _projection_helpers(), MonotoneConvex())],
        REF,
    )


def test_multi_curve_par_swap(joint) -> None:  # type: ignore[no-untyped-def]
    proj = [h for h in joint.helpers if getattr(h, "forward", None) == "EUR-6M"]
    h = proj[3]  # the 5Y 6M swap
    swap = h.swap.with_fixed_rate(h.quote)
    risk = quote_deltas(lambda c: swap.pv(c["discount"], c["EUR-6M"]), joint)
    i = joint.helpers.index(h)
    expected = np.zeros(len(joint.helpers))
    expected[i] = swap.annuity(joint.curves["discount"])
    # every other quote, OIS ones included, is fixed, so the par rate does not move
    assert risk.deltas == pytest.approx(expected, abs=1e-8)


def test_multi_curve_off_market_has_cross_risk(joint) -> None:  # type: ignore[no-untyped-def]
    swap = vanilla_swap(REF, "8Y", 0.05, EURIBOR6M)
    risk = quote_deltas(lambda c: swap.pv(c["discount"], c["EUR-6M"]), joint)
    n_ois = len(HS)
    assert np.max(np.abs(risk.deltas[:n_ois])) > 1e-3  # discounting risk off par
    assert np.max(np.abs(risk.deltas[n_ois:])) > np.max(np.abs(risk.deltas[:n_ois]))


def test_validation() -> None:
    with pytest.raises(ValueError):
        quote_deltas(lambda c: 0.0)
    with pytest.raises(ValueError):
        quote_deltas(lambda c: 0.0, helpers=HS[:-1], curves=LL.curves)
    with pytest.raises(KeyError):
        node_deltas(lambda c: 0.0, LL.curves, ["EUR-6M"])
    with pytest.raises(ValueError):
        with_node_values(LL.curve, [0.0])
    with pytest.raises(TypeError):
        curve_nodes(FlatCurve(0.03))
