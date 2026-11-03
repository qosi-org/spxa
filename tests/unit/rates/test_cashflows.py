"""Tests for spxa.rates.cashflows."""

from __future__ import annotations

import datetime as dt

import pytest

from spxa.market import FlatCurve, InterpolatedDiscountCurve, Schedule
from spxa.rates import (
    EURIBOR3M,
    SOFR,
    FixedCoupon,
    FixingHistory,
    IborCoupon,
    OvernightCoupon,
    SimpleCashflow,
    compounded_overnight_rate,
    fixed_leg,
    ibor_leg,
    overnight_leg,
)

D = dt.date
CURVE = InterpolatedDiscountCurve(
    [0.5, 1.0, 2.0, 5.0], [0.98, 0.961, 0.925, 0.83], reference_date="2024-01-02"
)


class TestFixed:
    def test_semiannual_30_360(self) -> None:
        sch = Schedule("2024-01-15", "2026-01-15", "6M", "TARGET")
        leg = fixed_leg(sch, 1_000_000.0, 0.04)
        assert len(leg) == 4
        assert [c.payment_date for c in leg] == [
            D(2024, 7, 15),
            D(2025, 1, 15),
            D(2025, 7, 15),
            D(2026, 1, 15),
        ]
        assert all(c.amount() == pytest.approx(20_000.0, abs=1e-9) for c in leg)

    def test_adjusted_dates_change_act_360_amounts(self) -> None:
        # 15 June 2024 is a Saturday -> period ends Monday 17 June
        sch = Schedule("2024-03-15", "2024-09-15", "3M", "TARGET")
        leg = fixed_leg(sch, 100.0, 0.05, "ACT/360")
        assert leg[0].end == D(2024, 6, 17)
        assert leg[0].amount() == pytest.approx(100 * 0.05 * 94 / 360, rel=1e-15)

    def test_payment_lag_and_amortising(self) -> None:
        sch = Schedule("2024-01-15", "2025-01-15", "6M", "US-SIFMA")
        leg = fixed_leg(sch, [100.0, 50.0], 0.04, payment_lag=2)
        assert leg[0].payment_date == D(2024, 7, 17)
        assert leg[1].payment_date == D(2025, 1, 17)
        assert leg[1].amount() == pytest.approx(1.0, abs=1e-14)
        with pytest.raises(ValueError, match="notionals"):
            fixed_leg(sch, [100.0], 0.04)

    def test_act_act_icma_uses_reference_period(self) -> None:
        # ISDA regular semi-annual example: exactly half a year under ICMA
        c = FixedCoupon(
            D(2004, 5, 1),
            D(2003, 11, 1),
            D(2004, 5, 1),
            100.0,
            0.1,
            "ACT/ACT ICMA",
            D(2003, 11, 1),
            D(2004, 5, 1),
            2,
        )
        assert c.accrual == pytest.approx(0.5, abs=1e-15)
        assert c.amount() == pytest.approx(5.0, abs=1e-13)


class TestIbor:
    def test_fixing_and_amount(self) -> None:
        c = IborCoupon(D(2024, 7, 3), D(2024, 4, 3), D(2024, 7, 3), 1e6, EURIBOR3M)
        assert c.fixing == D(2024, 3, 28)  # two TARGET days before, across Easter
        fix = FixingHistory.from_pairs("EURIBOR3M", [("2024-03-28", 0.0389)])
        assert c.accrual == pytest.approx(91 / 360, abs=1e-16)
        assert c.amount(fixings=fix) == pytest.approx(1e6 * 0.0389 * 91 / 360, rel=1e-15)

    def test_projection_over_index_dates(self) -> None:
        c = IborCoupon(
            D(2024, 7, 3), D(2024, 4, 3), D(2024, 7, 3), 1.0, EURIBOR3M, spread=0.001, gearing=2.0
        )
        v0, v1 = D(2024, 4, 3), EURIBOR3M.maturity_date(D(2024, 4, 3))
        tau = EURIBOR3M.dc.year_fraction(v0, v1)
        L = (CURVE.discount(v0) / CURVE.discount(v1) - 1) / tau
        assert c.index_rate(CURVE) == pytest.approx(L, rel=1e-14)
        assert c.rate(CURVE) == pytest.approx(2 * L + 0.001, rel=1e-14)

    def test_coupon_dates_variant_and_missing(self) -> None:
        # a stub period that does not match the index tenor
        c = IborCoupon(
            D(2024, 6, 3), D(2024, 4, 3), D(2024, 6, 3), 1.0, EURIBOR3M, use_index_dates=False
        )
        tau = EURIBOR3M.dc.year_fraction(c.start, c.end)
        L = (CURVE.discount(c.start) / CURVE.discount(c.end) - 1) / tau
        assert c.index_rate(CURVE) == pytest.approx(L, rel=1e-14)
        with pytest.raises(KeyError, match="no EURIBOR3M fixing"):
            c.amount()

    def test_leg(self) -> None:
        sch = Schedule("2024-04-03", "2025-04-03", "3M", "TARGET")
        leg = ibor_leg(sch, 1e6, EURIBOR3M, spread=0.0005)
        assert len(leg) == 4 and all(isinstance(c, IborCoupon) for c in leg)
        assert leg[0].fixing == D(2024, 3, 28)
        assert all(c.spread == 0.0005 for c in leg)


class TestOvernight:
    fix = FixingHistory.from_pairs(
        "SOFR",
        [
            ("2024-07-01", 0.054),
            ("2024-07-02", 0.0535),
            ("2024-07-03", 0.0533),
            ("2024-07-05", 0.0531),
        ],
    )

    def test_fully_fixed(self) -> None:
        c = OvernightCoupon(D(2024, 7, 10), D(2024, 7, 1), D(2024, 7, 8), 1e6, SOFR)
        r = compounded_overnight_rate(SOFR, "2024-07-01", "2024-07-08", self.fix).rate
        assert c.rate(fixings=self.fix) == r
        assert c.amount(fixings=self.fix) == pytest.approx(1e6 * r * 7 / 360, rel=1e-15)

    def test_projected_and_averaged(self) -> None:
        flat = FlatCurve(0.05, reference_date="2024-07-01", day_count="ACT/360")
        c = OvernightCoupon(D(2024, 10, 3), D(2024, 7, 1), D(2024, 10, 1), 1.0, SOFR)
        tau = c.accrual
        assert c.rate(flat) == pytest.approx(
            (flat.discount(0.0) / flat.discount(tau) - 1) / tau, rel=1e-13
        )
        avg = OvernightCoupon(
            D(2024, 10, 3), D(2024, 7, 1), D(2024, 10, 1), 1.0, SOFR, method="averaged"
        )
        assert avg.rate(flat) < c.rate(flat)

    def test_leg_with_payment_delay(self) -> None:
        sch = Schedule("2024-07-01", "2025-07-01", "3M", "US-SIFMA")
        leg = overnight_leg(sch, 1e6, SOFR, payment_lag=2)
        # 1 Jan 2025 rolls to Thursday 2 Jan, then two SIFMA days: Friday 3, Monday 6
        assert [c.payment_date for c in leg][:2] == [D(2024, 10, 3), D(2025, 1, 6)]
        assert all(isinstance(c, OvernightCoupon) for c in leg)


def test_simple_cashflow() -> None:
    c = SimpleCashflow(D(2025, 1, 2), -1e6)
    assert c.amount() == -1e6 and c.amount(CURVE) == -1e6
