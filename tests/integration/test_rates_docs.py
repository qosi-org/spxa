"""Every numerical statement in docs/theory/rates.md, checked against the code."""

from __future__ import annotations

import datetime as dt
import math
import re
from pathlib import Path

import pytest

import spxa.rates as r
from spxa.market import FlatCurve, InterpolatedDiscountCurve, Schedule

DOC = Path(__file__).resolve().parents[2] / "docs" / "theory" / "rates.md"
D = dt.date
REF = "2024-01-15"
CURVE = FlatCurve(0.03, reference_date=REF, day_count="ACT/365F")
DISC = InterpolatedDiscountCurve(
    [0.5, 1.0, 2.0, 5.0, 10.0], [0.98, 0.961, 0.925, 0.83, 0.70], reference_date=REF
)
PROJ = InterpolatedDiscountCurve(
    [0.5, 1.0, 2.0, 5.0, 10.0], [0.978, 0.957, 0.918, 0.82, 0.68], reference_date=REF
)


def test_table_names_exist() -> None:
    table = DOC.read_text(encoding="utf-8").split("---", 1)[0]
    names = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", table))
    names -= {"spxa", "rates", "YieldCurve"}
    missing = sorted(n for n in names if not hasattr(r, n))
    assert not missing, missing


def test_doc_mentions_the_checked_numbers() -> None:
    text = DOC.read_text(encoding="utf-8")
    for s in (
        "92.56126256977227",
        "7.98945",
        "0.824176",
        "99.99319",
        "1.875 \\times 10^{-5}",
        "1.927083 \\times 10^{-5}",
        "5.208 \\times 10^{-7}",
        "362/360",
        "91/360",
    ):
        assert s in text, s


class TestIndices:
    def test_euribor_dates(self) -> None:
        idx = r.EURIBOR3M
        v = idx.value_date("2024-01-15")
        assert v == D(2024, 1, 17) and idx.maturity_date(v) == D(2024, 4, 17)

    def test_sofr_week_weights(self) -> None:
        _, obs, w, total = r.observation_schedule(r.SOFR, "2024-07-01", "2024-07-08")
        assert obs == [D(2024, 7, 1), D(2024, 7, 2), D(2024, 7, 3), D(2024, 7, 5)]
        assert [round(x * 360) for x in w] == [1, 1, 2, 3]
        assert total == pytest.approx(7 / 360, abs=1e-16)

    def test_isda_lockout(self) -> None:
        acc, obs, w, _ = r.observation_schedule(r.ESTR, "2020-09-22", "2020-12-22", lockout=5)
        assert obs[-5:] == [D(2020, 12, 15)] * 5 and obs[-6] == D(2020, 12, 14)
        assert acc[-5] == D(2020, 12, 15) and acc[-1] == D(2020, 12, 21)
        assert round(sum(w[-5:]) * 360) == 7

    def test_telescoping_projection(self) -> None:
        fast = r.projected_overnight_rate(r.SOFR, "2024-02-01", "2024-08-01", CURVE)
        daily = r.compounded_overnight_rate(
            r.SOFR, "2024-02-01", "2024-08-01", projection=r.curve_projection(r.SOFR, CURVE)
        ).rate
        assert fast == pytest.approx(daily, rel=1e-13)


def test_single_curve_floating_leg_telescopes() -> None:
    sch = Schedule("2024-01-17", "2029-01-17", "6M", "TARGET", "MF")
    leg = r.ibor_leg(sch, 1.0, r.EURIBOR6M, use_index_dates=False)
    pv = r.leg_pv(leg, DISC, DISC)
    assert pv == pytest.approx(
        float(DISC.discount(D(2024, 1, 17))) - float(DISC.discount(D(2029, 1, 17))), rel=1e-13
    )


class TestMoneyMarketAndFutures:
    def test_fra_dates_and_value(self) -> None:
        fra = r.FRA.from_index(r.EURIBOR3M, REF, "3M", 0.035, notional=1e6)
        assert (fra.start, fra.end) == (D(2024, 4, 17), D(2024, 7, 17))
        tau = fra.accrual
        F = (float(PROJ.discount(fra.start)) / float(PROJ.discount(fra.end)) - 1) / tau
        ref = 1e6 * tau * (F - 0.035) / (1 + tau * F) * float(DISC.discount(fra.start))
        assert fra.pv(DISC, PROJ) == pytest.approx(ref, rel=1e-12)

    def test_term_future_convexity(self) -> None:
        g = r.hull_white_futures_convexity(1.0, 1.25, 0.0, 0.01)
        assert g == pytest.approx(0.01**2 * 0.25 * 1.0 * 1.5 / 2, rel=1e-14)
        assert g == pytest.approx(1.875e-5, rel=1e-12)
        assert r.imm_code(D(2024, 3, 20)) == "H24"
        assert r.imm_from_code("H24", "2024-01-01") == D(2024, 3, 20)

    def test_sofr_future(self) -> None:
        fut = r.OvernightFuture.from_imm("2024-03-20", 94.75)
        assert (fut.start, fut.end) == (D(2024, 3, 20), D(2024, 6, 19))
        assert fut.accrual == pytest.approx(91 / 360, abs=1e-16)
        gc = r.hull_white_compounded_convexity(1.0, 1.25, 0.0, 0.01)
        assert round(gc * 1e5, 6) == 1.927083
        assert gc > r.hull_white_futures_convexity(1.0, 1.25, 0.0, 0.01)
        spot = r.hull_white_compounded_convexity(0.0, 0.25, 0.0, 0.01)
        assert spot == pytest.approx(0.01**2 * 0.25**3 / 3, rel=1e-14)
        assert round(spot * 1e7, 3) == 5.208


class TestSwaps:
    def test_identities(self) -> None:
        sw = r.vanilla_swap(REF, "7Y", 0.04, notional=1e6)
        S, A = sw.par_rate(DISC, PROJ), sw.annuity(DISC)
        assert sw.pv(DISC, PROJ) == pytest.approx((S - 0.04) * A, rel=1e-12)
        ois = r.ois_swap(REF, "5Y", 0.04, payment_lag=0)
        single = (
            float(CURVE.discount(ois.start)) - float(CURVE.discount(ois.fixed[-1].end))  # type: ignore[union-attr]
        ) / ois.annuity(CURVE)
        assert ois.par_rate(CURVE) == pytest.approx(single, rel=1e-13)

    def test_sofr_ois_dates(self) -> None:
        sw = r.ois_swap(REF, "5Y", 0.04)
        assert sw.start == D(2024, 1, 18)
        assert sw.fixed[0].payment_date == D(2025, 1, 23)
        assert sw.maturity == D(2029, 1, 22)

    def test_euribor_swap_dates(self) -> None:
        sw = r.vanilla_swap(REF, "10Y", 0.03)
        assert sw.start == D(2024, 1, 17)
        assert len(sw.fixed) == 10 and len(sw.floating) == 20
        second = sw.fixed[1]
        assert second.end == D(2026, 1, 19)  # type: ignore[union-attr]
        assert second.accrual == pytest.approx(362 / 360, abs=1e-15)  # type: ignore[union-attr]

    def test_basis_single_curve(self) -> None:
        sw = r.basis_swap(REF, "10Y", r.EURIBOR3M, r.EURIBOR6M, use_index_dates=False)
        assert abs(sw.par_spread(DISC, DISC, DISC)) < 1e-14

    def test_risk_measures(self) -> None:
        sw = r.ois_swap(REF, "10Y", 0.04, notional=1e6, payment_lag=0)
        assert sw.pv01(CURVE) == pytest.approx(1e-4 * sw.annuity(CURVE), rel=1e-15)
        t = CURVE.time
        s, e = sw.start, sw.fixed[-1].end  # type: ignore[union-attr]
        analytic = 1e-4 * (
            1e6 * (-t(s) * CURVE.discount(s) + t(e) * CURVE.discount(e))
            + sum(
                0.04 * 1e6 * c.accrual * t(c.payment_date) * CURVE.discount(c.payment_date)  # type: ignore[union-attr]
                for c in sw.fixed
            )
        )
        assert sw.dv01(CURVE) == pytest.approx(analytic, rel=1e-6)
        atm = sw.with_fixed_rate(sw.par_rate(CURVE))
        assert atm.dv01(CURVE) == pytest.approx(atm.pv01(CURVE), rel=0.05)

    def test_amortising_decomposition(self) -> None:
        amort = r.ois_swap(REF, "3Y", 0.04, notional=[3.0, 2.0, 1.0])
        bullets = [r.ois_swap(REF, f"{j}Y", 0.04) for j in (1, 2, 3)]
        assert amort.pv(DISC, PROJ) == pytest.approx(
            sum(b.pv(DISC, PROJ) for b in bullets), abs=1e-14
        )
        ann = [b.annuity(DISC) for b in bullets]
        avg = sum(a * b.par_rate(DISC, PROJ) for a, b in zip(ann, bullets)) / sum(ann)
        assert amort.par_rate(DISC, PROJ) == pytest.approx(avg, rel=1e-13)


class TestBond:
    bond = r.FixedRateBond("2024-01-15", "2034-01-15", 0.05)

    def test_table(self) -> None:
        b = self.bond
        assert b.dirty_price(0.06, "2024-01-15") == pytest.approx(92.56126256977227, abs=1e-11)
        assert b.clean_price(0.05, "2024-01-15") == pytest.approx(100.0, abs=1e-11)
        assert round(b.macaulay_duration(0.05, "2024-01-15"), 5) == 7.98945
        assert b.modified_duration(0.05, "2024-01-15") == pytest.approx(
            b.macaulay_duration(0.05, "2024-01-15") / 1.025, rel=1e-14
        )
        acc = b.accrued_interest("2024-03-15")
        assert acc == pytest.approx(2.5 * 60 / 182, abs=1e-14)
        assert round(acc, 6) == 0.824176
        assert round(b.clean_price(0.05, "2024-03-15"), 5) == 99.99319

    def test_final_coupon_after_mlk_day(self) -> None:
        last = self.bond.cashflows()[-1]
        assert last.payment_date == D(2034, 1, 17)
        assert D(2034, 1, 15).weekday() == 6 and D(2034, 1, 16).weekday() == 0

    def test_annuity_formula(self) -> None:
        # on a coupon date the street formula is the textbook annuity
        y, n = 0.06, 20
        ref = 2.5 * (1 - (1 + y / 2) ** -n) / (y / 2) + 100 * (1 + y / 2) ** -n
        assert self.bond.dirty_price(y, "2024-01-15") == pytest.approx(ref, rel=1e-14)
        assert math.isfinite(ref)
