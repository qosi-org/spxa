"""Fixed-rate bonds."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from spxa.market import FlatCurve
from spxa.rates import FixedRateBond

D = dt.date
BOND = FixedRateBond("2024-01-15", "2034-01-15", 0.05)


def test_cashflows() -> None:
    cfs = BOND.cashflows()
    assert len(cfs) == 21 and cfs[-1].is_redemption and cfs[-1].amount == 100.0
    assert all(c.amount == pytest.approx(2.5, abs=1e-14) for c in cfs[:-1])
    # 15 Jan 2034 is a Sunday and Monday 16 January is MLK Day: paid on the 17th,
    # while accrual runs on the unadjusted date
    assert (cfs[-2].accrual_end, cfs[-2].payment_date) == (D(2034, 1, 15), D(2034, 1, 17))
    assert BOND.settlement_date("2024-01-12") == D(2024, 1, 16)  # MLK day skipped


def test_textbook_prices() -> None:
    # 10Y 5% semi-annual at 6%: 2.5 a(20, 3%) + 100 / 1.03^20
    ann = (1 - 1.03**-20) / 0.03
    assert BOND.dirty_price(0.06, "2024-01-15") == pytest.approx(
        2.5 * ann + 100 * 1.03**-20, rel=1e-14
    )
    assert BOND.clean_price(0.05, "2024-01-15") == pytest.approx(100.0, rel=1e-14)


def test_accrued_and_mid_period_price() -> None:
    s = D(2024, 3, 15)
    assert BOND.accrued_interest(s) == pytest.approx(2.5 * 60 / 182, rel=1e-15)
    w = 122 / 182  # remaining fraction of the period to 15 July
    dirty = sum(2.5 * 1.025 ** -(w + i) for i in range(20)) + 100 * 1.025 ** -(w + 19)
    assert BOND.dirty_price(0.05, s) == pytest.approx(dirty, rel=1e-14)
    assert BOND.yield_from_clean(BOND.clean_price(0.0437, s), s) == pytest.approx(0.0437, abs=1e-14)


def test_risk_measures_against_finite_differences() -> None:
    s, y, h = D(2025, 5, 2), 0.047, 1e-5
    P = BOND.dirty_price(y, s)
    up, dn = BOND.dirty_price(y + h, s), BOND.dirty_price(y - h, s)
    assert BOND.modified_duration(y, s) == pytest.approx(-(up - dn) / (2 * h) / P, rel=1e-8)
    assert BOND.convexity(y, s) == pytest.approx((up - 2 * P + dn) / (h * h) / P, rel=1e-5)
    assert BOND.macaulay_duration(y, s) == pytest.approx(
        BOND.modified_duration(y, s) * (1 + y / 2), rel=1e-15
    )


def test_zero_coupon_duration() -> None:
    z = FixedRateBond("2024-01-15", "2029-01-15", 0.0, "1Y")
    assert z.frequency == 1
    assert z.macaulay_duration(0.04, "2024-01-15") == pytest.approx(5.0, rel=1e-14)
    assert z.dirty_price(0.04, "2024-01-15") == pytest.approx(100 / 1.04**5, rel=1e-14)


def test_long_first_coupon_icma() -> None:
    # ISDA example: long first period 15 Aug 2002 - 15 Jul 2003, semi-annual
    b = FixedRateBond("2002-08-15", "2004-07-15", 0.10, first_coupon="2003-07-15")
    assert b.cashflows()[0].amount == pytest.approx(10 * 0.915760869565, abs=1e-11)


def test_curve_pricing_and_z_spread() -> None:
    curve = FlatCurve(0.04, reference_date="2024-01-15", day_count="ACT/365F")
    s = D(2024, 1, 15)
    manual = sum(
        c.amount * math.exp(-0.04 * (c.payment_date - s).days / 365) for c in BOND.cashflows()
    )
    assert BOND.dirty_price_from_curve(curve, s) == pytest.approx(manual, rel=1e-14)
    clean = BOND.dirty_price_from_curve(curve, s, z_spread=0.0125) - BOND.accrued_interest(s)
    assert BOND.z_spread(clean, curve, s) == pytest.approx(0.0125, abs=1e-12)


def test_validation() -> None:
    with pytest.raises(ValueError, match="outside"):
        BOND.accrued_interest("2035-01-01")
    with pytest.raises(ValueError, match="divide"):
        FixedRateBond("2024-01-15", "2034-01-15", 0.05, "5M")
