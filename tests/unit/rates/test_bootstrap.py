"""Sequential curve bootstrapping."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from spxa.market.curves import FlatCurve
from spxa.rates import ESTR, EURIBOR6M, FRA, Deposit, ois_swap, vanilla_swap
from spxa.rates.bootstrap import bootstrap_curve
from spxa.rates.helpers import DepositHelper, FRAHelper, SwapHelper
from spxa.rates.interpolation import (
    LinearZero,
    LogLinearDiscount,
    MonotoneConvex,
    MonotoneCubicRT,
    PiecewiseCurve,
)

REF = dt.date(2024, 1, 15)
MARKET = PiecewiseCurve.from_zero_rates(
    [0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30],
    [0.039, 0.0385, 0.037, 0.034, 0.032, 0.0305, 0.030, 0.0302, 0.031, 0.0308, 0.029],
    MonotoneConvex(),
    reference_date=REF,
)
OIS_TENORS = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]


def _estr_helpers(curve: object) -> list:
    hs: list = []
    for t in ("1M", "3M", "6M"):
        d = Deposit.from_tenor(REF, t, 0.0)
        hs.append(DepositHelper(d, quote=d.implied_rate(curve)))  # type: ignore[arg-type]
    for t in OIS_TENORS:
        s = ois_swap(REF, t, 0.0, ESTR)
        hs.append(SwapHelper(s, quote=s.par_rate(curve)))  # type: ignore[arg-type]
    return hs


@pytest.mark.parametrize("scheme", [LogLinearDiscount(), LinearZero()], ids=repr)
def test_local_schemes_reprice_in_one_pass(scheme: object) -> None:
    res = bootstrap_curve(_estr_helpers(MARKET), REF, scheme)  # type: ignore[arg-type]
    assert res.passes == 1
    assert res.max_residual < 1e-15
    assert res.curve.node_times.size == 12
    assert res.pillars == sorted(res.pillars)
    # close to the market curve it was generated from
    t = np.linspace(0.1, 30, 300)
    assert (
        np.max(np.abs(res.curve.zero_rate(t) - MARKET.zero_rate(t))) < 5e-4
    )  # interpolation, not error


def test_recovers_log_linear_nodes_exactly() -> None:
    hs0 = _estr_helpers(MARKET)
    pillars = sorted(h.pillar for h in hs0)
    true = PiecewiseCurve(pillars, MARKET.discount(pillars), reference_date=REF)
    res = bootstrap_curve(_estr_helpers(true), REF)
    assert res.curve.node_discounts == pytest.approx(true.node_discounts, rel=1e-14)


@pytest.mark.parametrize("scheme", [MonotoneConvex(), MonotoneCubicRT()], ids=repr)
def test_non_local_schemes_iterate(scheme: object) -> None:
    hs = _estr_helpers(MARKET)
    one = bootstrap_curve(hs, REF, scheme, max_passes=1)  # type: ignore[arg-type]
    assert one.max_residual > 1e-9  # later nodes moved earlier instruments
    res = bootstrap_curve(hs, REF, scheme)  # type: ignore[arg-type]
    assert 1 < res.passes < 50
    assert res.max_residual < 1e-13


def test_projection_curve_on_known_discount() -> None:
    ois = FlatCurve(0.03, reference_date=REF)
    proj = PiecewiseCurve.from_zero_rates(
        [0.5, 2, 5, 10], [0.036, 0.035, 0.034, 0.035], reference_date=REF
    )
    curves = {"discount": ois, "EUR-6M": proj}
    hs: list = []
    for st in ("1M", "3M", "6M", "12M"):
        fra = FRA.from_index(EURIBOR6M, REF, st, 0.0)
        hs.append(FRAHelper(fra, quote=fra.forward_rate(proj), curve="EUR-6M"))
    for t in ("2Y", "3Y", "5Y", "7Y", "10Y"):
        s = vanilla_swap(REF, t, 0.0, EURIBOR6M)
        h = SwapHelper(s, forward="EUR-6M")
        h.quote = h.implied_quote(curves)
        hs.append(h)
    res = bootstrap_curve(hs, REF, name="EUR-6M", known={"discount": ois})
    assert res.max_residual < 1e-15
    assert set(res.curves) == {"discount", "EUR-6M"}
    # the swap quotes depend on the discount curve: a different OIS curve gives other nodes
    other = bootstrap_curve(
        hs, REF, name="EUR-6M", known={"discount": FlatCurve(0.01, reference_date=REF)}
    )
    assert not np.allclose(other.curve.node_discounts, res.curve.node_discounts, rtol=1e-8)


def test_validation() -> None:
    hs = _estr_helpers(MARKET)
    with pytest.raises(ValueError, match="no helpers"):
        bootstrap_curve([], REF)
    with pytest.raises(ValueError, match="share a pillar"):
        bootstrap_curve(hs + [hs[-1]], REF)
    with pytest.raises(ValueError, match="after the reference"):
        bootstrap_curve(hs, dt.date(2060, 1, 1))
    with pytest.raises(ValueError, match="does not read"):
        bootstrap_curve(hs, REF, name="EUR-6M")
    with pytest.raises(ValueError, match="both known"):
        bootstrap_curve(hs, REF, known={"discount": MARKET})
    with pytest.raises(ValueError):
        bootstrap_curve(hs, REF, max_passes=0)
