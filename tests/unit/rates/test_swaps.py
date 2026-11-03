"""Swaps: OIS and fixed versus IBOR."""

from __future__ import annotations

import datetime as dt

import pytest

from spxa.market import InterpolatedDiscountCurve
from spxa.rates import (
    ESTR,
    EURIBOR3M,
    EURIBOR6M,
    SOFR,
    FixedCoupon,
    FixingHistory,
    IborCoupon,
    OvernightCoupon,
    Swap,
    ois_swap,
    vanilla_swap,
)

D = dt.date
REF = "2024-01-15"
DISC = InterpolatedDiscountCurve(
    [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    [0.987, 0.974, 0.951, 0.91, 0.80, 0.65, 0.30],
    reference_date=REF,
)


class TestOISDates:
    def test_sofr_5y(self) -> None:
        # 15 Jan 2024 is Martin Luther King Day: spot is two days after 16 Jan
        sw = ois_swap(REF, "5Y", 0.04)
        assert sw.start == D(2024, 1, 18)
        fx = sw.fixed
        assert len(fx) == 5 and len(sw.floating) == 5
        assert [c.end for c in fx] == [  # type: ignore[attr-defined]
            D(2025, 1, 21),  # 18 Jan 2025 is a Saturday; Monday 20 Jan is MLK day
            D(2026, 1, 20),
            D(2027, 1, 19),
            D(2028, 1, 18),
            D(2029, 1, 18),
        ]
        assert fx[0].payment_date == D(2025, 1, 23)
        assert all(isinstance(c, FixedCoupon) and c.day_count == "ACT/360" for c in fx)
        assert all(isinstance(c, OvernightCoupon) for c in sw.floating)
        assert sw.maturity == D(2029, 1, 22)

    def test_short_and_forward_start(self) -> None:
        short = ois_swap(REF, "6M", 0.05, ESTR)
        assert len(short.fixed) == 1
        assert (short.start, short.fixed[0].end) == (D(2024, 1, 17), D(2024, 7, 17))  # type: ignore[attr-defined]
        fwd = ois_swap(REF, "1Y", 0.04, forward_start="1Y")
        assert fwd.start == D(2025, 1, 21)


class TestOISPricing:
    def test_single_curve_par_rate(self) -> None:
        sw = ois_swap(REF, "10Y", 0.04, payment_lag=0, notional=1e7)
        s, e = sw.start, sw.fixed[-1].end  # type: ignore[attr-defined]
        par = (DISC.discount(s) - DISC.discount(e)) * 1e7 / sw.annuity(DISC)
        assert sw.par_rate(DISC) == pytest.approx(par, rel=1e-11)

    def test_identities(self) -> None:
        sw = ois_swap(REF, "7Y", 0.035, notional=1e6)
        S, A = sw.par_rate(DISC), sw.annuity(DISC)
        assert sw.pv(DISC) == pytest.approx((S - 0.035) * A, rel=1e-12)
        rec = ois_swap(REF, "7Y", 0.035, notional=1e6, payer=False)
        assert rec.pv(DISC) == pytest.approx(-sw.pv(DISC), rel=1e-15)
        at_par = ois_swap(REF, "7Y", S, notional=1e6)
        assert at_par.pv(DISC) == pytest.approx(0.0, abs=1e-8)
        # par spread: the floating margin that offsets the PV
        m = sw.par_spread(DISC)
        spread = ois_swap(REF, "7Y", 0.035, notional=1e6, spread=m)
        assert spread.pv(DISC) == pytest.approx(0.0, abs=1e-8)
        assert sw.fixed_rate == 0.035

    def test_seasoned_with_fixings(self) -> None:
        sw = ois_swap("2023-12-01", "2Y", 0.04, payment_lag=0)
        pairs = []
        d = D(2023, 12, 5)
        while d < D(2024, 1, 16):
            if SOFR.is_fixing_date(d):
                pairs.append((d, 0.0531))
            d += dt.timedelta(days=1)
        fix = FixingHistory.from_pairs("SOFR", pairs)
        bumped = FixingHistory.from_pairs("SOFR", [(d, r + 0.01) for d, r in pairs])
        first = sw.floating[0]
        assert isinstance(first, OvernightCoupon)
        df = DISC.discount(first.payment_date)
        moved = (first.amount(DISC, bumped) - first.amount(DISC, fix)) * df
        assert sw.pv(DISC, fixings=bumped) - sw.pv(DISC, fixings=fix) == pytest.approx(
            moved, rel=1e-12
        )
        assert moved > 0
        with pytest.raises(KeyError):
            sw.pv(DISC)

    def test_validation(self) -> None:
        sw = ois_swap(REF, "2Y", 0.04)
        with pytest.raises(ValueError, match="both legs"):
            Swap([], sw.floating)
        with pytest.raises(ValueError, match="FixedCoupon"):
            Swap(sw.floating, sw.floating)


PROJ = InterpolatedDiscountCurve(
    [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    [0.985, 0.970, 0.944, 0.90, 0.785, 0.63, 0.28],
    reference_date=REF,
)


class TestVanilla:
    def test_eur_10y_dates(self) -> None:
        sw = vanilla_swap(REF, "10Y", 0.03)
        assert sw.start == D(2024, 1, 17)
        assert len(sw.fixed) == 10 and len(sw.floating) == 20
        fl = sw.floating
        assert all(isinstance(c, IborCoupon) for c in fl)
        assert fl[0].fixing == D(2024, 1, 15)  # type: ignore[union-attr]
        assert all(c.day_count == "30/360" for c in sw.fixed)  # type: ignore[union-attr]
        # 17 Jan 2026 and 2027 fall on weekends
        assert [c.end for c in sw.fixed][1:4] == [D(2026, 1, 19), D(2027, 1, 18), D(2028, 1, 17)]  # type: ignore[union-attr]
        # the fixed coupon over a rolled period still accrues 30/360 between adjusted dates
        assert sw.fixed[1].accrual == pytest.approx(362 / 360, abs=1e-15)  # type: ignore[union-attr]

    def test_single_curve_telescopes(self) -> None:
        exact = vanilla_swap(REF, "10Y", 0.03, notional=1e6, use_index_dates=False)
        s, e = exact.start, exact.fixed[-1].end  # type: ignore[union-attr]
        par = 1e6 * (DISC.discount(s) - DISC.discount(e)) / exact.annuity(DISC)
        assert exact.par_rate(DISC) == pytest.approx(par, rel=1e-12)
        # with index dates, a period starting on a rolled date (Mon 19 Jan 2026)
        # projects EURIBOR to 20 July rather than to the coupon end on 17 July
        sw = vanilla_swap(REF, "10Y", 0.03, notional=1e6)
        c = sw.floating[4]
        assert isinstance(c, IborCoupon)
        assert (c.start, c.end, EURIBOR6M.maturity_date(c.start)) == (
            D(2026, 1, 19),
            D(2026, 7, 17),
            D(2026, 7, 20),
        )
        assert sw.par_rate(DISC) != exact.par_rate(DISC)
        assert sw.par_rate(DISC) == pytest.approx(par, rel=1e-4)

    def test_multi_curve(self) -> None:
        sw = vanilla_swap(REF, "5Y", 0.03, notional=1e6)
        num = 0.0
        for c in sw.floating:
            v0 = c.start  # type: ignore[union-attr]
            v1 = EURIBOR6M.maturity_date(v0)
            F = (PROJ.discount(v0) / PROJ.discount(v1) - 1) / EURIBOR6M.dc.year_fraction(v0, v1)
            num += 1e6 * F * c.accrual * DISC.discount(c.payment_date)  # type: ignore[union-attr]
        S = sw.par_rate(DISC, PROJ)
        assert S == pytest.approx(num / sw.annuity(DISC), rel=1e-13)
        assert S > sw.par_rate(DISC)  # the 6M curve sits above OIS
        at_par = vanilla_swap(REF, "5Y", S, notional=1e6, payer=False)
        assert at_par.pv(DISC, PROJ) == pytest.approx(0.0, abs=1e-8)
        assert sw.pv(DISC, PROJ) == pytest.approx((S - 0.03) * sw.annuity(DISC), rel=1e-12)

    def test_usd_style_and_forward_start(self) -> None:
        sw = vanilla_swap(REF, "2Y", 0.04, EURIBOR3M, fixed_tenor="6M", forward_start="1Y")
        assert sw.start == D(2025, 1, 17)
        assert len(sw.fixed) == 4 and len(sw.floating) == 8
        spread = sw.par_spread(DISC, PROJ)
        with_spread = vanilla_swap(
            REF, "2Y", 0.04, EURIBOR3M, fixed_tenor="6M", forward_start="1Y", spread=spread
        )
        assert with_spread.pv(DISC, PROJ) == pytest.approx(0.0, abs=1e-12)
