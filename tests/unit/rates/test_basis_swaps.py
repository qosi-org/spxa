"""Floating-for-floating basis swaps."""

from __future__ import annotations

import datetime as dt

import pytest

from spxa.market import InterpolatedDiscountCurve
from spxa.rates import (
    ESTR,
    EURIBOR3M,
    EURIBOR6M,
    BasisSwap,
    IborCoupon,
    OvernightCoupon,
    basis_swap,
)

D = dt.date
REF = "2024-01-15"
TIMES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
OIS = InterpolatedDiscountCurve(
    TIMES, [0.991, 0.982, 0.965, 0.932, 0.84, 0.70, 0.36], reference_date=REF
)
E3M = InterpolatedDiscountCurve(
    TIMES, [0.990, 0.980, 0.961, 0.926, 0.83, 0.685, 0.34], reference_date=REF
)
E6M = InterpolatedDiscountCurve(
    TIMES, [0.989, 0.978, 0.958, 0.921, 0.822, 0.675, 0.33], reference_date=REF
)


def test_structure() -> None:
    sw = basis_swap(REF, "5Y", EURIBOR3M, EURIBOR6M, spread=0.001)
    assert len(sw.first) == 20 and len(sw.second) == 10
    assert all(isinstance(c, IborCoupon) and c.spread == 0.001 for c in sw.first)
    assert all(isinstance(c, IborCoupon) and c.spread == 0.0 for c in sw.second)
    assert sw.maturity == D(2029, 1, 17)
    ois = basis_swap(REF, "2Y", ESTR, EURIBOR3M)
    assert len(ois.first) == 8 and all(isinstance(c, OvernightCoupon) for c in ois.first)
    with pytest.raises(ValueError):
        BasisSwap([], sw.second)


def test_single_curve_has_zero_basis() -> None:
    sw = basis_swap(REF, "10Y", EURIBOR3M, EURIBOR6M, notional=1e6, use_index_dates=False)
    p1, p2 = sw.leg_pvs(OIS, OIS, OIS)
    full = 1e6 * (OIS.discount(D(2024, 1, 17)) - OIS.discount(D(2034, 1, 17)))
    assert p1 == pytest.approx(full, rel=1e-12) and p2 == pytest.approx(full, rel=1e-12)
    assert abs(sw.par_spread(OIS, OIS, OIS)) < 1e-14


def test_multi_curve_par_spread() -> None:
    sw = basis_swap(REF, "10Y", EURIBOR3M, EURIBOR6M, notional=1e6)
    s = sw.par_spread(OIS, E3M, E6M)
    assert 0 < s < 0.005  # 6M trades above 3M
    p1, p2 = sw.leg_pvs(OIS, E3M, E6M)
    a1 = sum(c.notional * c.accrual * OIS.discount(c.payment_date) for c in sw.first)  # type: ignore[union-attr]
    assert s == pytest.approx((p2 - p1) / a1, rel=1e-14)
    at_par = basis_swap(REF, "10Y", EURIBOR3M, EURIBOR6M, spread=s, notional=1e6)
    assert at_par.pv(OIS, E3M, E6M) == pytest.approx(0.0, abs=1e-7)


def test_ois_ibor_basis() -> None:
    sw = basis_swap(REF, "5Y", ESTR, EURIBOR3M, notional=1e6)
    s = sw.par_spread(OIS, OIS, E3M)
    assert s > 0  # EURIBOR above ESTR
    at_par = basis_swap(REF, "5Y", ESTR, EURIBOR3M, spread=s, notional=1e6)
    assert at_par.pv(OIS, OIS, E3M) == pytest.approx(0.0, abs=1e-7)
