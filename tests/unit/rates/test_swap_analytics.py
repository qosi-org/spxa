"""Swap analytics: PV01, DV01, re-striking, amortisation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.market import FlatCurve, InterpolatedDiscountCurve, Schedule
from spxa.rates import EURIBOR6M, Swap, fixed_leg, ibor_leg, ois_swap, vanilla_swap

REF = "2024-01-15"
CURVE = FlatCurve(0.03, reference_date=REF, day_count="ACT/365F")
DISC = InterpolatedDiscountCurve(
    [0.5, 1.0, 2.0, 5.0, 10.0], [0.98, 0.961, 0.925, 0.83, 0.70], reference_date=REF
)
PROJ = InterpolatedDiscountCurve(
    [0.5, 1.0, 2.0, 5.0, 10.0], [0.978, 0.957, 0.918, 0.82, 0.68], reference_date=REF
)


def test_restrike_and_pv01() -> None:
    sw = vanilla_swap(REF, "7Y", 0.05, notional=1e6)
    par = sw.par_rate(DISC, PROJ)
    assert sw.with_fixed_rate(par).pv(DISC, PROJ) == pytest.approx(0.0, abs=1e-8)
    assert sw.with_fixed_rate(par).floating is sw.floating
    assert sw.pv01(DISC) == pytest.approx(1e-4 * sw.annuity(DISC), rel=1e-15)
    # one basis point on the strike moves the PV by exactly -PV01 for the payer
    assert sw.with_fixed_rate(0.0501).pv(DISC, PROJ) - sw.pv(DISC, PROJ) == pytest.approx(
        -sw.pv01(DISC), rel=1e-9
    )


def test_dv01_matches_analytic_derivative() -> None:
    sw = ois_swap(REF, "10Y", 0.04, notional=1e6, payment_lag=0)
    K, N = 0.04, 1e6
    t = CURVE.time
    s, e = sw.start, sw.fixed[-1].end  # type: ignore[union-attr]
    dfloat = N * (-t(s) * CURVE.discount(s) + t(e) * CURVE.discount(e))
    dfixed = sum(
        K * N * c.accrual * t(c.payment_date) * CURVE.discount(c.payment_date) for c in sw.fixed
    )  # type: ignore[union-attr]
    analytic = 1e-4 * (dfloat + dfixed)
    assert sw.dv01(CURVE) == pytest.approx(analytic, rel=1e-6)
    assert sw.dv01(CURVE) > 0  # payer of fixed gains when rates rise
    rec = ois_swap(REF, "10Y", 0.04, notional=1e6, payment_lag=0, payer=False)
    assert rec.dv01(CURVE) == pytest.approx(-sw.dv01(CURVE), rel=1e-12)
    # an at-the-money swap has DV01 close to its PV01
    atm = sw.with_fixed_rate(sw.par_rate(CURVE))
    assert atm.dv01(CURVE) == pytest.approx(atm.pv01(CURVE), rel=0.05)


def test_multi_curve_dv01_bumps_both_curves() -> None:
    sw = vanilla_swap(REF, "5Y", 0.03, notional=1e6)
    full = sw.dv01(DISC, PROJ)
    h = 1e-4
    manual = (
        sw.pv(DISC.shifted(h), PROJ.shifted(h)) - sw.pv(DISC.shifted(-h), PROJ.shifted(-h))
    ) / 2
    assert full == pytest.approx(manual, rel=1e-12)


def test_amortising_swap() -> None:
    sch = Schedule("2024-01-17", "2029-01-17", "1Y", "TARGET", "MF")
    fsch = Schedule("2024-01-17", "2029-01-17", "6M", "TARGET", "MF")
    notionals = [1e6, 8e5, 6e5, 4e5, 2e5]
    fl_notionals = list(np.repeat(notionals, 2))
    sw = Swap(fixed_leg(sch, notionals, 0.03), ibor_leg(fsch, fl_notionals, EURIBOR6M))
    num = 0.0
    for c in sw.floating:
        num += c.amount(PROJ) * DISC.discount(c.payment_date)
    den = sum(c.notional * c.accrual * DISC.discount(c.payment_date) for c in sw.fixed)  # type: ignore[union-attr]
    assert sw.par_rate(DISC, PROJ) == pytest.approx(num / den, rel=1e-14)
    bullet = vanilla_swap(REF, "5Y", 0.03, notional=1e6)
    assert sw.annuity(DISC) < bullet.annuity(DISC)
    assert math.isfinite(sw.dv01(DISC, PROJ))
