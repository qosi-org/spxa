"""Caps and floors on term-rate indices."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from spxa.market.black import bachelier_price, black_price
from spxa.market.curves import FlatCurve
from spxa.rates.capfloor import cap_floor
from spxa.rates.indices import EURIBOR3M, EURIBOR6M
from spxa.rates.pricing import leg_pv

REF = dt.date(2024, 3, 15)
DISC = FlatCurve(0.03, reference_date=REF)
PROJ = FlatCurve(0.034, reference_date=REF)


def test_schedule() -> None:
    cap = cap_floor(REF, "5Y", 0.035, EURIBOR6M)
    assert len(cap.caplets) == 9  # ten periods, the first already fixed
    first = cap.caplets[0].coupon
    assert first.fixing == EURIBOR6M.fixing_date(first.start)
    assert 0.45 < cap.caplets[0].expiry(DISC) < 0.55
    full = cap_floor(REF, "5Y", 0.035, EURIBOR6M, skip_first=False)
    assert len(full.caplets) == 10


def test_single_caplet_is_black() -> None:
    cap = cap_floor(REF, "2Y", 0.033, EURIBOR3M, notional=1e6)
    c = cap.caplets[2]
    cp = c.coupon
    v0 = EURIBOR3M.value_date(cp.fixing)
    v1 = EURIBOR3M.maturity_date(v0)
    tau_idx = EURIBOR3M.dc.year_fraction(v0, v1)
    F = (float(PROJ.discount(v0)) / float(PROJ.discount(v1)) - 1) / tau_idx
    T = (cp.fixing - REF).days / 365.0
    w = 1e6 * cp.accrual * float(DISC.discount(cp.payment_date))
    assert c.price(DISC, PROJ, 0.25) == pytest.approx(w * black_price(F, 0.033, T, 0.25), rel=1e-13)
    assert c.price(DISC, PROJ, 0.009, model="normal") == pytest.approx(
        w * bachelier_price(F, 0.033, T, 0.009), rel=1e-13
    )


def test_put_call_parity() -> None:
    for index in (EURIBOR3M, EURIBOR6M):
        cap = cap_floor(REF, "7Y", 0.031, index, notional=1e7)
        floor = cap.with_strike(0.031, "floor")
        float_pv = leg_pv([c.coupon for c in cap.caplets], DISC, PROJ)
        for model, vol in (("black", 0.3), ("normal", 0.01), ("shifted_black", 0.2)):
            diff = cap.price(DISC, PROJ, vol, model, shift=0.02) - floor.price(
                DISC, PROJ, vol, model, shift=0.02
            )
            assert diff == pytest.approx(float_pv - 0.031 * cap.annuity(DISC), rel=1e-11)


def test_atm_strike_and_limits() -> None:
    cap = cap_floor(REF, "5Y", 0.03, EURIBOR6M)
    k = cap.atm_strike(DISC, PROJ)
    # simple ACT/360 6M forwards of a 3.4% ACT/365F continuous curve: 3.4% x 360/365 plus compounding
    assert 0.0336 < k < 0.0340
    assert cap.with_strike(k).price(DISC, PROJ, 0.2) == pytest.approx(
        cap.with_strike(k, "floor").price(DISC, PROJ, 0.2), rel=1e-12
    )
    # a zero-strike cap with a tiny normal vol is the floating leg, the zero-strike floor is worthless in Black
    float_pv = leg_pv([c.coupon for c in cap.caplets], DISC, PROJ)
    assert cap.with_strike(0.0).price(DISC, PROJ, 1e-4, "normal") == pytest.approx(
        float_pv, rel=1e-12
    )
    assert cap.with_strike(1e-8, "floor").price(DISC, PROJ, 0.3) < 1e-15


def test_implied_flat_vol() -> None:
    cap = cap_floor(REF, "10Y", 0.036, EURIBOR6M)

    def term(T: float, K: float) -> float:
        return 0.18 + 0.05 * np.exp(-T)

    p = cap.price(DISC, PROJ, term)
    flat = cap.implied_flat_vol(p, DISC, PROJ)
    assert cap.price(DISC, PROJ, flat) == pytest.approx(p, rel=1e-12)
    Ts = [c.expiry(DISC) for c in cap.caplets]
    assert term(Ts[-1], 0) < flat < term(Ts[0], 0)
    assert cap.implied_flat_vol(
        cap.price(DISC, PROJ, 0.0123, "normal"), DISC, PROJ, "normal"
    ) == pytest.approx(0.0123, rel=1e-10)
    with pytest.raises(ValueError):
        cap.implied_flat_vol(10.0, DISC, PROJ)


def test_negative_rates_need_a_shift() -> None:
    neg = FlatCurve(-0.005, reference_date=REF)
    floor = cap_floor(REF, "3Y", -0.004, EURIBOR6M, kind="floor")
    with pytest.raises(ValueError):
        floor.price(neg, neg, 0.3)
    p = floor.price(neg, neg, 0.3, "shifted_black", shift=0.02)
    assert p > 0
    assert floor.price(neg, neg, 0.003, "normal") > 0
    with pytest.raises(ValueError):
        cap_floor(REF, "3Y", 0.0, kind="collar")
