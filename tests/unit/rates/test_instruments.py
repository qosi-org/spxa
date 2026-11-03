"""Deposits and FRAs."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from spxa.market import FlatCurve, InterpolatedDiscountCurve
from spxa.rates import EURIBOR3M, EURIBOR6M, FRA, Deposit, FixingHistory

D = dt.date
REF = "2024-01-15"
DISC = InterpolatedDiscountCurve(
    [0.25, 0.5, 1.0, 2.0], [0.991, 0.982, 0.964, 0.93], reference_date=REF
)
PROJ = InterpolatedDiscountCurve(
    [0.25, 0.5, 1.0, 2.0], [0.989, 0.978, 0.957, 0.92], reference_date=REF
)


class TestDeposit:
    def test_dates(self) -> None:
        dep = Deposit.from_tenor("2024-01-15", "3M", 0.039)
        assert (dep.start, dep.end) == (D(2024, 1, 17), D(2024, 4, 17))
        assert dep.accrual == pytest.approx(91 / 360, abs=1e-16)
        # end of month: spot 29 Feb 2024 -> 31 May (not 29 May)
        eom = Deposit.from_tenor("2024-02-27", "3M", 0.039)
        assert (eom.start, eom.end) == (D(2024, 2, 29), D(2024, 5, 31))
        assert Deposit.from_tenor("2024-02-27", "3M", 0.039, end_of_month=False).end == D(
            2024, 5, 29
        )
        # overnight deposit: spot lag 0
        on = Deposit.from_tenor("2024-03-28", "1D", 0.039, spot_lag=0)
        assert (on.start, on.end) == (D(2024, 3, 28), D(2024, 4, 2))  # across Easter

    def test_pv_and_par_rate(self) -> None:
        dep = Deposit.from_tenor(REF, "6M", 0.04, notional=1e6)
        par = dep.implied_rate(DISC)
        assert Deposit(dep.start, dep.end, par).pv(DISC) == pytest.approx(0.0, abs=1e-15)
        p0, p1 = DISC.discount(dep.start), DISC.discount(dep.end)
        assert dep.pv(DISC) == pytest.approx(1e6 * ((1 + 0.04 * dep.accrual) * p1 - p0), rel=1e-14)
        flat = FlatCurve(0.03, reference_date=REF, day_count="ACT/360")
        assert dep.implied_rate(flat) == pytest.approx(
            (math.exp(0.03 * dep.accrual) - 1) / dep.accrual, rel=1e-13
        )
        with pytest.raises(ValueError):
            Deposit(D(2024, 2, 1), D(2024, 2, 1), 0.04)


class TestFRA:
    def test_dates(self) -> None:
        fra = FRA.from_index(EURIBOR3M, "2024-01-15", "3M", 0.038)
        assert (fra.start, fra.end) == (D(2024, 4, 17), D(2024, 7, 17))
        assert fra.fixing_date == D(2024, 4, 15)
        assert fra.accrual == pytest.approx(91 / 360, abs=1e-16)
        assert fra.maturity == fra.end
        f6 = FRA.from_index(EURIBOR6M, "2024-01-15", "6M", 0.038)  # 6x12
        assert (f6.start, f6.end) == (D(2024, 7, 17), D(2025, 1, 17))

    def test_single_curve_identity(self) -> None:
        fra = FRA.from_index(EURIBOR3M, REF, "3M", 0.038, notional=1e6)
        p0, p1 = DISC.discount(fra.start), DISC.discount(fra.end)
        assert fra.pv(DISC) == pytest.approx(1e6 * (p0 - (1 + 0.038 * fra.accrual) * p1), rel=1e-12)
        rec = FRA(fra.start, fra.end, 0.038, EURIBOR3M, notional=1e6, payer=False)
        assert rec.pv(DISC) == pytest.approx(-fra.pv(DISC), abs=1e-12)

    def test_multi_curve(self) -> None:
        fra = FRA.from_index(EURIBOR6M, REF, "6M", 0.04, notional=1e6)
        tau = fra.accrual
        F = (PROJ.discount(fra.start) / PROJ.discount(fra.end) - 1) / tau
        assert fra.forward_rate(PROJ) == pytest.approx(F, rel=1e-14)
        expected = 1e6 * tau * (F - 0.04) / (1 + tau * F) * DISC.discount(fra.start)
        assert fra.pv(DISC, PROJ) == pytest.approx(expected, rel=1e-13)
        atm = FRA(fra.start, fra.end, fra.implied_rate(PROJ), EURIBOR6M)
        assert atm.pv(DISC, PROJ) == pytest.approx(0.0, abs=1e-16)

    def test_fixings_and_settlement(self) -> None:
        # FRA fixed on 15 Jan 2024 (the valuation date) but not yet settled
        fra = FRA(D(2024, 1, 17), D(2024, 4, 17), 0.038, EURIBOR3M, notional=1e6)
        fix = FixingHistory.from_pairs("EURIBOR3M", [("2024-01-15", 0.0392)])
        amt = fra.settlement_amount(0.0392)
        assert amt == pytest.approx(1e6 * (91 / 360) * 0.0012 / (1 + 91 / 360 * 0.0392), rel=1e-12)
        assert fra.pv(DISC, PROJ, fix) == pytest.approx(amt * DISC.discount(fra.start), rel=1e-14)
        # a fixing in the past without data is an error, a settled FRA is worth nothing
        old = FRA(D(2024, 1, 17), D(2024, 4, 17), 0.038, EURIBOR3M)
        late = InterpolatedDiscountCurve([1.0], [0.96], reference_date="2024-01-16")
        with pytest.raises(KeyError, match="missing past EURIBOR3M fixing"):
            old.pv(late)
        settled = InterpolatedDiscountCurve([1.0], [0.96], reference_date="2024-01-17")
        assert old.pv(settled) == 0.0
        with pytest.raises(ValueError):
            FRA(D(2024, 2, 1), D(2024, 1, 1), 0.04)
