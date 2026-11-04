"""Every numerical statement in docs/theory/rates.md, checked against the code."""

from __future__ import annotations

import datetime as dt
import math
import re
from pathlib import Path

import numpy as np
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
        "0.05010625",
        "30/10 = 3",
        "$3/360$",
        "31 December 2024 to 2 January 2025",
        "30 December 2022 to 2 January 2023",
        "-5e^{-0.15} = -4.30354",
        "5^3 = 125",
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


HUMP_T = [0.1, 1.0, 4.0, 9.0, 20.0, 30.0]
HUMP_R = [0.081, 0.07, 0.044, 0.07, 0.04, 0.03]


class TestCurveConstruction:
    def test_monotone_convex_reference_and_averages(self) -> None:
        t = [1.0, 2.0, 3.0, 4.0, 5.0]
        c = r.PiecewiseCurve.from_zero_rates(
            t, [0.0275, 0.04, 0.0475, 0.05, 0.0475], r.MonotoneConvex()
        )
        f = c.instantaneous_forward([0.3, 1.3, 2.1, 4.5])
        assert list(f) == pytest.approx([0.0229375, 0.05010625, 0.0609, 0.03625], abs=1e-15)
        # each discrete forward is the average of f over its interval
        x = -np.log(c.discount([0.0] + t))
        for a, b, xa, xb in zip([0.0] + t, t, x, x[1:]):
            g = np.linspace(a, b, 20001)
            avg = float(np.trapezoid(c.instantaneous_forward(g), g)) / (b - a)
            assert avg == pytest.approx((xb - xa) / (b - a), abs=1e-10)

    def test_scheme_comparison(self) -> None:
        schemes = [r.LogLinearDiscount(), r.LinearZero(), r.CubicZero(), r.CubicSplineRT()]
        schemes += [r.MonotoneCubicRT(), r.MonotoneConvex()]
        rows = r.compare_schemes(HUMP_T, HUMP_R, schemes)
        d = {row.scheme.split("(")[0]: row for row in rows}
        assert d["LinearZero"].min_forward < 0 and d["CubicZero"].min_forward < 0
        assert d["MonotoneConvex"].min_forward > 0
        assert d["LogLinearDiscount"].min_forward == pytest.approx(
            (0.03 * 30 - 0.04 * 20) / 10, abs=1e-15
        )
        assert d["LogLinearDiscount"].sensitivity == pytest.approx(30 / 10, abs=1e-12)
        assert d["CubicSplineRT"].locality == 6 and d["CubicZero"].locality == 5
        assert d["MonotoneCubicRT"].locality <= 2 and d["MonotoneConvex"].locality <= 2

    def test_log_linear_bump_formula(self) -> None:
        t, rates = np.array(HUMP_T), np.array(HUMP_R)
        base = r.PiecewiseCurve.from_zero_rates(t, rates, r.LogLinearDiscount())
        j, h = 3, 1e-6
        bumped = r.PiecewiseCurve.from_zero_rates(
            t, rates + h * (np.arange(6) == j), r.LogLinearDiscount()
        )
        mid_left, mid_right = 0.5 * (t[j - 1] + t[j]), 0.5 * (t[j] + t[j + 1])
        dl = (bumped.instantaneous_forward(mid_left) - base.instantaneous_forward(mid_left)) / h
        dr = (bumped.instantaneous_forward(mid_right) - base.instantaneous_forward(mid_right)) / h
        assert dl == pytest.approx(t[j] / (t[j] - t[j - 1]), rel=1e-6)
        assert dr == pytest.approx(-t[j] / (t[j + 1] - t[j]), rel=1e-6)

    def test_front_end_stubs(self) -> None:
        on = r.Deposit.overnight("2024-03-08", 0.039)
        tn = r.Deposit.tom_next("2024-03-08", 0.0391)
        assert (on.start, on.end) == (D(2024, 3, 8), D(2024, 3, 11))
        assert on.accrual == pytest.approx(3 / 360, abs=1e-16)
        res = r.bootstrap_curve(
            [r.DepositHelper(on, 0.039), r.DepositHelper(tn, 0.0391)], "2024-03-08"
        )
        p_spot = res.curve.discount(res.curve.time(tn.end))
        expected = 1 / ((1 + on.accrual * 0.039) * (1 + tn.accrual * 0.0391))
        assert p_spot == pytest.approx(expected, rel=1e-13)

    def test_turns(self) -> None:
        a = r.turn_of_year(2024, 0.002)
        b = r.turn_of_year(2022, 0.002)
        assert (a.start, a.end) == (D(2024, 12, 31), D(2025, 1, 2))
        assert (b.start, b.end) == (D(2022, 12, 30), D(2023, 1, 2))
        assert D(2022, 12, 31).weekday() == 5
        base = FlatCurve(0.03, reference_date="2024-06-28", day_count="ACT/365F")
        adj = r.SpreadAdjustedCurve(base, [a])
        ta, tb = base.time(a.start), base.time(a.end)
        for t in (0.25, 0.5, ta + 0.5 * (tb - ta), 1.0):
            overlap = max(0.0, min(t, tb) - ta)
            assert adj.discount(t) == pytest.approx(
                base.discount(t) * math.exp(-0.002 * overlap), rel=1e-14
            )

    def test_nss_limits(self) -> None:
        c = r.NelsonSiegelSvensson(0.042, 0.01, -0.012, 0.008, 1.2, 9.0)
        assert c.instantaneous_forward(0.0) == pytest.approx(0.052, abs=1e-15)
        assert c.zero(1e-9) == pytest.approx(0.052, abs=1e-9)
        assert c.zero(1e5) == pytest.approx(0.042, abs=1e-5)

    def test_unidentified_meeting_period_raises(self) -> None:
        dep = r.Deposit.from_tenor("2024-01-15", "1M", 0.053)
        with pytest.raises(ValueError):
            r.meeting_date_curve(
                [
                    r.DepositHelper(dep, 0.053),
                    r.DepositHelper(r.Deposit.from_tenor("2024-01-15", "2M", 0.053), 0.053),
                ],
                "2024-01-15",
                ["2024-01-31", "2024-01-31", "2024-02-05", "2024-02-06"],
            )


class TestCurveRisk:
    flat = FlatCurve(0.03, reference_date=REF, day_count="ACT/365F")

    def test_zero_coupon_key_rate(self) -> None:
        risk = r.key_rate_deltas(
            lambda c: float(c["discount"].discount(5.0)), {"discount": self.flat}, [1, 2, 5, 10]
        )
        assert round(risk.deltas[2], 5) == -4.30354
        assert risk.deltas[2] == pytest.approx(-5 * math.exp(-0.15), rel=1e-7)
        assert risk.deltas[[0, 1, 3]] == pytest.approx([0, 0, 0], abs=1e-12)

    def test_par_swap_quote_deltas_and_hedge(self) -> None:
        from tests.unit.rates.test_bootstrap import MARKET, _estr_helpers

        hs = _estr_helpers(MARKET)
        fit = r.bootstrap_curve(hs, "2024-01-15", r.LogLinearDiscount())
        h = hs[5]
        par = h.swap.with_fixed_rate(h.quote)

        def pv(c):  # type: ignore[no-untyped-def]
            return par.pv(c["discount"], c["discount"])

        risk = r.quote_deltas(pv, fit)
        a = par.annuity(fit.curve)
        assert risk.deltas == pytest.approx(a * (np.arange(len(hs)) == 5), abs=1e-9)
        res = r.hedge_notionals(np.array([2.0, -1.0]), np.diag([0.5, 4.0]))
        assert res.notionals == pytest.approx([-4.0, 0.25]) and res.residual_norm < 1e-15

    def test_explain_remainder(self) -> None:
        t = np.array([1.0, 2.0, 5.0])
        c0 = r.PiecewiseCurve(t, np.exp(-0.03 * t), reference_date=REF)
        x = -np.log(c0.node_discounts)
        dx = 1e-3
        c1 = r.with_node_values(c0, x + dx * (t == 5.0))
        ex = r.pnl_explain(
            lambda c: float(c["discount"].discount(5.0)), {"discount": c0}, {"discount": c1}
        )
        assert ex.unexplained == pytest.approx(-math.exp(-0.15) * dx**3 / 6, rel=2e-3)

    def test_pca_factor_variance(self) -> None:
        rng = np.random.default_rng(0)
        tenors = np.array([1.0, 2, 5, 10, 30])
        v, _ = np.linalg.qr(rng.normal(size=(5, 2)))
        changes = rng.normal(size=(300, 2)) @ np.diag([8e-4, 3e-4]) @ v.T
        pca = r.curve_pca(changes, tenors, n_components=2, differences=False)
        level, slope = pca.components.T
        assert level.sum() > 0 and slope[-1] > slope[0]
        d = rng.normal(size=5)
        assert pca.factor_variance(d) == pytest.approx(np.var(changes @ d, ddof=1), rel=1e-10)


class TestVolatilityProducts:
    def test_section_numbers_in_text(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        for s in (
            "9 caplets",
            "$0.35902$",
            "$0.36024$",
            "$8.74\\%$",
            "$10.2$ bp",
            "$10.8$ bp",
            "$10^{-8}$",
        ):
            assert s in text, s

    def test_caps_and_swaptions(self) -> None:
        ref = D(2024, 3, 15)
        disc = FlatCurve(0.03, reference_date=ref)
        proj = FlatCurve(0.034, reference_date=ref)
        cap = r.cap_floor(ref, "5Y", 0.035, r.EURIBOR6M)
        floor = r.cap_floor(ref, "5Y", 0.035, r.EURIBOR6M, kind="floor")
        assert len(cap.caplets) == 9
        diff = cap.price(disc, proj, 0.2) - floor.price(disc, proj, 0.2)
        fwd = sum(c.weight(disc) * c.forward(proj) for c in cap.caplets)
        assert diff == pytest.approx(fwd - 0.035 * cap.annuity(disc), abs=1e-11)
        assert r.cash_annuity(0.04, 10, 1) == pytest.approx((1 - 1.04**-10) / 0.04, rel=1e-14)

    def test_sabr_wings_and_density(self) -> None:
        M = r.SABR(0.035, 0.5, -0.3, 0.4)
        K = 0.03 / math.e
        assert round(float(M.lognormal_vol(0.03, K, 0.0)), 5) == 0.35902
        assert round(float(M.lognormal_vol(0.03, K, 0.0, "obloj")), 5) == 0.36024
        m = r.SABR(0.02, 0.5, -0.3, 0.5)
        k = np.linspace(1e-4, 0.01, 400)
        c = m.price(0.01, k, 10.0)
        assert np.min(np.diff(c, 2)) < 0  # Hagan: negative density near zero
        d = r.ArbitrageFreeSABR(m, 0.01, 10.0).solve()
        assert round(d.P_L, 4) == 0.0874
        assert d.mass() == pytest.approx(1.0, abs=1e-12)

    def test_cms_closed_form_and_numbers(self) -> None:
        import warnings

        from spxa.market.black import black_price

        S0, T, s, A0, P = 0.03, 5.0, 0.2, 4.3, 0.85

        def price(K: float, o: str) -> float:
            return float(black_price(S0, max(K, 1e-12), T, s, 1.0, o))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lin = r.CMSReplication(S0, T, A0, P, price, n=10, mapping="linear")
            std = r.CMSReplication(S0, T, A0, P, price, n=10)
            closed = lin.slope * A0 / P * S0**2 * math.expm1(s * s * T)
            assert lin.convexity_adjustment() == pytest.approx(closed, rel=1e-8)
            assert round(lin.convexity_adjustment() * 1e4, 1) == 10.2
            assert round(std.convexity_adjustment() * 1e4, 1) == 10.8
