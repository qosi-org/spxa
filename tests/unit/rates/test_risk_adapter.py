"""spxa.risk swaps valued on yield curves."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.market import FlatCurve, InterpolatedDiscountCurve
from spxa.rates import CurveQuote, curve_price, curve_pricing_fn
from spxa.risk.instruments import EuropeanOption, InterestRateSwap
from spxa.risk.scenarios import StressTestingEngine


def swap(K: float = 0.03, T: float = 5.0, receive: bool = False) -> InterestRateSwap:
    return InterestRateSwap("IRS", "USD_SOFR", K, T, 0.5, receive_fixed=receive, notional=1e6)


def test_flat_curve_closed_form_and_par() -> None:
    q = curve_price(swap(), {"r": 0.03})
    times = 0.5 * np.arange(1, 11)
    annuity = 0.5 * np.exp(-0.03 * times).sum()
    assert isinstance(q, CurveQuote)
    assert q.price == pytest.approx(1e6 * (1 - math.exp(-0.15) - 0.03 * annuity), abs=1e-8)
    assert q.annuity == pytest.approx(1e6 * annuity, abs=1e-8)
    at_par = curve_price(swap(q.par_rate), {"r": 0.03})
    assert abs(at_par.price) < 1e-8
    assert curve_price(swap(receive=True), {"r": 0.03}).price == pytest.approx(-q.price, abs=1e-9)


def test_short_last_period_and_curve_key() -> None:
    curve = InterpolatedDiscountCurve([0.0, 1.0, 3.0], [1.0, 0.97, 0.90])
    q = curve_price(swap(T=2.25), {"r": 0.10, "USD_SOFR_CURVE": curve})
    t = np.array([0.5, 1.0, 1.5, 2.0, 2.25])
    p = np.asarray(curve.discount(t))
    tau = np.array([0.5, 0.5, 0.5, 0.5, 0.25])
    assert q.price == pytest.approx(1e6 * (1 - p[-1] - 0.03 * tau @ p), abs=1e-8)


@pytest.mark.parametrize("receive", [False, True])
def test_dv01_is_the_parallel_derivative(receive: bool) -> None:
    curve = InterpolatedDiscountCurve([0.0, 2.0, 5.0, 10.0], [1.0, 0.95, 0.86, 0.72])
    s = swap(0.025, 7.0, receive)
    h = 1e-6
    up = curve_price(s, {"USD_SOFR_CURVE": curve.shifted(h)}).price
    dn = curve_price(s, {"USD_SOFR_CURVE": curve.shifted(-h)}).price
    dv01 = curve_price(s, {"USD_SOFR_CURVE": curve}).dv01
    assert dv01 == pytest.approx(1e-4 * (up - dn) / (2 * h), rel=1e-7)
    assert (dv01 > 0) == (not receive)  # payers gain when rates rise


def test_stress_scenarios_shift_the_curve() -> None:
    curve = InterpolatedDiscountCurve([0.0, 2.0, 5.0, 10.0], [1.0, 0.95, 0.86, 0.72])
    base = {"r": 0.03, "USD_SOFR_CURVE": curve}
    book = [(swap(0.03, 5.0), 2.0), (swap(0.028, 10.0, receive=True), 1.0)]
    engine = StressTestingEngine(curve_pricing_fn(book, base))
    out = engine.run_historical_suite(base)
    for name, shift in (("LEHMAN_2008", -0.02), ("INFLATION_SHOCK_2022", 0.035)):
        moved = {"USD_SOFR_CURVE": curve.shifted(shift)}
        want = sum(q * curve_price(s, moved).price for s, q in book)
        assert out[name]["stressed_pv"] == pytest.approx(want, abs=1e-6)
    assert out["LEHMAN_2008"]["base_pv"] == pytest.approx(curve_pricing_fn(book)(base), abs=1e-9)


def test_errors() -> None:
    with pytest.raises(TypeError):
        curve_price(EuropeanOption("C", "SPX", 100.0, 1.0), {"r": 0.03})
    with pytest.raises(KeyError):
        curve_price(swap(), {})
    with pytest.raises(ValueError):
        curve_price(swap(T=0.0), {"r": 0.03})
    assert curve_pricing_fn(swap())({"r": 0.03}) == pytest.approx(
        curve_price(swap(), {"r": 0.03}).price, abs=1e-12
    )
    assert FlatCurve(0.03).discount(1.0) == pytest.approx(math.exp(-0.03))
