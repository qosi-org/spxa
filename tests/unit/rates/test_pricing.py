"""Tests for spxa.rates.pricing."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from spxa.market import FlatCurve, InterpolatedDiscountCurve, Schedule
from spxa.rates import (
    EURIBOR6M,
    SOFR,
    FixingHistory,
    SimpleCashflow,
    cashflow_table,
    fixed_leg,
    ibor_leg,
    leg_annuity,
    leg_pv,
    overnight_leg,
)

D = dt.date
REF = "2024-01-02"
DISC = InterpolatedDiscountCurve(
    [0.5, 1.0, 2.0, 5.0, 10.0], [0.98, 0.961, 0.925, 0.83, 0.70], reference_date=REF
)
PROJ = InterpolatedDiscountCurve(
    [0.5, 1.0, 2.0, 5.0, 10.0], [0.978, 0.957, 0.918, 0.82, 0.68], reference_date=REF
)


def test_fixed_leg_on_flat_curve() -> None:
    curve = FlatCurve(0.03, reference_date=REF)
    sch = Schedule("2024-01-02", "2027-01-04", "1Y", "TARGET")
    leg = fixed_leg(sch, 1e6, 0.04, "ACT/360")
    ref = sum(
        1e6
        * 0.04
        * (c.end - c.start).days
        / 360
        * math.exp(-0.03 * (c.payment_date - D(2024, 1, 2)).days / 365)
        for c in leg
    )
    assert leg_pv(leg, curve) == pytest.approx(ref, rel=1e-14)
    assert leg_pv(leg, curve) == pytest.approx(0.04 * leg_annuity(leg, curve), rel=1e-14)


def test_single_curve_float_leg_telescopes() -> None:
    sch = Schedule("2024-01-02", "2029-01-02", "3M", "US-SIFMA")
    leg = overnight_leg(sch, 1e6, SOFR)
    start, end = sch.dates[0], sch.dates[-1]
    ref = 1e6 * (DISC.discount(start) - DISC.discount(end))
    assert leg_pv(leg, DISC) == pytest.approx(ref, rel=1e-11)


def test_ibor_leg_single_and_multi_curve() -> None:
    sch = Schedule("2024-01-04", "2029-01-04", "6M", "TARGET")
    leg = ibor_leg(sch, 1e6, EURIBOR6M, use_index_dates=False)
    ref = 1e6 * (DISC.discount(sch.dates[0]) - DISC.discount(sch.dates[-1]))
    assert leg_pv(leg, DISC) == pytest.approx(ref, rel=1e-12)
    # multi-curve: projected forwards discounted on another curve
    pv = leg_pv(leg, DISC, PROJ)
    manual = 0.0
    for c in leg:
        tau = EURIBOR6M.dc.year_fraction(c.start, c.end)
        L = (PROJ.discount(c.start) / PROJ.discount(c.end) - 1) / tau
        manual += 1e6 * L * c.accrual * DISC.discount(c.payment_date)
    assert pv == pytest.approx(manual, rel=1e-14)
    assert pv > leg_pv(leg, DISC)  # projection curve has higher forwards


def test_valuation_date_excludes_past_cashflows() -> None:
    sch = Schedule("2023-07-03", "2025-07-03", "6M", "TARGET")
    leg = fixed_leg(sch, 100.0, 0.05)
    full = leg_pv(leg, DISC, valuation_date="2023-07-03")
    later = leg_pv(
        leg, DISC
    )  # curve reference date 2024-01-02: first coupon (3 Jan 2024) still alive
    assert later == pytest.approx(full, rel=1e-15)
    after = leg_pv(leg, DISC, valuation_date="2024-01-03")
    first = leg[0].amount() * DISC.discount(leg[0].payment_date)
    assert after == pytest.approx(full - first, rel=1e-14)


def test_fixings_used_for_past_reset() -> None:
    sch = Schedule("2023-12-04", "2024-12-04", "6M", "TARGET")
    leg = ibor_leg(sch, 1e6, EURIBOR6M)
    fix = FixingHistory.from_pairs("EURIBOR6M", [(leg[0].fixing, 0.0395)])
    rows = cashflow_table(leg, DISC, PROJ, fix)
    assert rows[0].rate == 0.0395
    assert rows[0].amount == pytest.approx(1e6 * 0.0395 * rows[0].accrual, rel=1e-15)
    assert sum(r.pv for r in rows) == pytest.approx(leg_pv(leg, DISC, PROJ, fix), rel=1e-15)
    with pytest.raises(KeyError, match="missing past EURIBOR6M fixing"):
        leg_pv(leg, DISC, PROJ)


def test_table_and_simple_cashflows() -> None:
    leg = [SimpleCashflow(D(2025, 1, 2), 100.0)]
    (row,) = cashflow_table(leg, DISC)
    assert row.rate is None and row.start is None
    assert row.pv == pytest.approx(100.0 * DISC.discount(D(2025, 1, 2)), rel=1e-15)
    assert leg_annuity(leg, DISC) == 0.0
    with pytest.raises(ValueError, match="valuation_date"):
        leg_pv(leg, InterpolatedDiscountCurve([1.0], [0.97]))


def test_missing_past_overnight_fixings() -> None:
    sch = Schedule("2023-12-01", "2024-06-03", "3M", "US-SIFMA")
    leg = overnight_leg(sch, 1e6, SOFR)
    # the 1-4 Dec stub has already been paid; the next period starts 4 Dec 2023
    with pytest.raises(KeyError, match="missing past SOFR fixing for 2023-12-04"):
        leg_pv(leg, DISC)
    # the averaged method goes through the daily projection and fails the same way
    avg = overnight_leg(sch, 1e6, SOFR, method="averaged")
    with pytest.raises(KeyError, match="missing past SOFR fixing"):
        leg_pv(avg, DISC)
