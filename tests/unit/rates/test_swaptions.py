"""European swaptions: physical and cash settlement."""

from __future__ import annotations

import datetime as dt

import pytest

from spxa.market.black import bachelier_price, black_price
from spxa.market.curves import FlatCurve
from spxa.rates.swaptions import Swaption, cash_annuity, swaption

REF = dt.date(2024, 3, 15)
DISC = FlatCurve(0.03, reference_date=REF)
PROJ = FlatCurve(0.034, reference_date=REF)


def test_dates_and_frequency() -> None:
    sw = swaption(REF, "2Y", "5Y", 0.035)
    assert (sw.swap.start - sw.expiry).days in (2, 3, 4)  # two TARGET business days
    assert 1.95 < sw.time_to_expiry(DISC) < 2.05
    assert sw.frequency() == 1
    assert swaption(REF, "1Y", "3Y", 0.03, fixed_tenor="6M").frequency() == 2
    with pytest.raises(ValueError):
        Swaption(sw.swap, sw.swap.start + dt.timedelta(days=1))


def test_physical_is_black_on_the_annuity() -> None:
    sw = swaption(REF, "1Y", "10Y", 0.036, notional=1e6)
    A = sw.swap.annuity(DISC)
    S = sw.swap.par_rate(DISC, PROJ)
    T = sw.time_to_expiry(DISC)
    assert sw.price(DISC, PROJ, 0.2) == pytest.approx(A * black_price(S, 0.036, T, 0.2), rel=1e-13)
    assert sw.price(DISC, PROJ, 0.008, "normal") == pytest.approx(
        A * bachelier_price(S, 0.036, T, 0.008), rel=1e-13
    )


def test_payer_receiver_parity_and_atm() -> None:
    pay = swaption(REF, "3Y", "7Y", 0.031, notional=1e7)
    rec = swaption(REF, "3Y", "7Y", 0.031, notional=1e7, payer=False)
    for model, vol in (("black", 0.25), ("normal", 0.009), ("shifted_black", 0.15)):
        diff = pay.price(DISC, PROJ, vol, model, 0.01) - rec.price(DISC, PROJ, vol, model, 0.01)
        assert diff == pytest.approx(pay.swap.pv(DISC, PROJ), rel=1e-11)
    S = pay.forward_rate(DISC, PROJ)
    p = swaption(REF, "3Y", "7Y", S).price(DISC, PROJ, 0.2)
    r = swaption(REF, "3Y", "7Y", S, payer=False).price(DISC, PROJ, 0.2)
    assert p == pytest.approx(r, rel=1e-11)


def test_cash_annuity() -> None:
    assert cash_annuity(0.04, 10, 1) == pytest.approx(
        sum(1.04**-i for i in range(1, 11)), rel=1e-14
    )
    assert cash_annuity(0.04, 10, 2, 100.0) == pytest.approx(
        sum(50.0 * 1.02**-i for i in range(1, 11)), rel=1e-14
    )
    assert cash_annuity(0.0, 10, 2) == 5.0
    # no cancellation for tiny yields
    assert cash_annuity(1e-10, 10, 1) == pytest.approx(10 * (1 - 11e-10 / 2), rel=1e-15)
    # n - n(n+1)/2 S + n(n+1)(n+2)/6 S^2 + O(S^3)
    assert cash_annuity(1e-6, 30, 1) == pytest.approx(30 - 465e-6 + 4960e-12, rel=1e-14)


def test_cash_settled() -> None:
    pay = swaption(REF, "5Y", "10Y", 0.035, settlement="cash")
    rec = swaption(REF, "5Y", "10Y", 0.035, settlement="cash", payer=False)
    S = pay.forward_rate(DISC, PROJ)
    A_cash = float(DISC.discount(pay.swap.start)) * cash_annuity(S, 10, 1)
    assert pay.annuity(DISC, PROJ) == pytest.approx(A_cash, rel=1e-14)
    diff = pay.price(DISC, PROJ, 0.2) - rec.price(DISC, PROJ, 0.2)
    assert diff == pytest.approx(A_cash * (S - 0.035), rel=1e-11)
    # discount and projection on one flat curve: cash and physical annuities are close
    phys = swaption(REF, "5Y", "10Y", 0.035)
    assert pay.annuity(DISC, DISC) == pytest.approx(phys.annuity(DISC, DISC), rel=3e-3)


def test_implied_vol_round_trip() -> None:
    sw = swaption(REF, "2Y", "10Y", 0.04, payer=False, notional=1e8)
    for model, vol in (("black", 0.31), ("normal", 0.0105)):
        assert sw.implied_vol(sw.price(DISC, PROJ, vol, model), DISC, PROJ, model) == pytest.approx(
            vol, rel=1e-10
        )
    with pytest.raises(ValueError):
        sw.implied_vol(1e9, DISC, PROJ)
    with pytest.raises(ValueError):
        sw.price(DISC, PROJ, 0.2, model="sabr")
