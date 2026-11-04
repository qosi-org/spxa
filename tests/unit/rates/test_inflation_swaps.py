"""Zero-coupon and year-on-year inflation swaps on the bootstrapped curve."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from spxa.market.curves import FlatCurve
from spxa.rates.inflation import CPIFixings, InflationIndex, bootstrap_inflation_curve
from spxa.rates.inflation_swaps import YearOnYearInflationSwap, ZeroCouponInflationSwap
from spxa.rates.jarrow_yildirim import JarrowYildirim

LEVELS = 118.0 * np.exp(0.0025 * np.arange(27))
HICP = InflationIndex(
    "HICPxT", CPIFixings.from_series("HICPxT", "2023-01-01", LEVELS), lag=3, interpolated=False
)
TODAY = dt.date(2025, 4, 1)
NOMINAL = FlatCurve(0.025, reference_date=TODAY)
QUOTES = {1: 0.0210, 2: 0.0205, 3: 0.0200, 5: 0.0198, 10: 0.0210}
CURVE = bootstrap_inflation_curve(HICP, TODAY, QUOTES)


def test_zero_coupon_swaps_at_par() -> None:
    for n, k in QUOTES.items():
        swap = ZeroCouponInflationSwap(TODAY, n, k, notional=1e7)
        assert swap.present_value(NOMINAL, CURVE) == pytest.approx(0.0, abs=1e-6)
        assert swap.fair_rate(CURVE) == pytest.approx(k, abs=1e-14)
    off = ZeroCouponInflationSwap(TODAY, 5, 0.0298, notional=1e7)
    expected = -1e7 * float(NOMINAL.discount(off.maturity)) * (1.0298**5 - 1.0198**5)
    assert off.present_value(NOMINAL, CURVE) == pytest.approx(expected, rel=1e-12)


def test_year_on_year_swaps() -> None:
    flat = bootstrap_inflation_curve(HICP, TODAY, {n: 0.02 for n in range(1, 11)})
    swap = YearOnYearInflationSwap(TODAY, 10, 0.0)
    # without convexity each year's ratio is 1.02 and the accruals are whole years
    assert swap.fair_rate(NOMINAL, flat) == pytest.approx(0.02, abs=1e-14)
    # with Jarrow-Yildirim on the consistent real curve the leg is the sum of model swaplets
    real = CURVE.real_curve(NOMINAL, TODAY)
    jy = JarrowYildirim(
        NOMINAL, real, CURVE.reference_cpi(TODAY), 0.1, 0.01, 0.05, 0.008, 0.012, 0.6, 0.2, -0.3
    )
    times = [float(NOMINAL.time(d)) for d in swap.schedule()]
    legs = sum(jy.yoy_swaplet(a, b) for a, b in zip(times, times[1:]))
    assert swap.inflation_leg(NOMINAL, CURVE, jy) == pytest.approx(legs, rel=1e-12)
    gap = swap.fair_rate(NOMINAL, CURVE, jy) - swap.fair_rate(NOMINAL, CURVE)
    assert round(gap * 1e4, 2) == -3.02  # basis points of convexity
    assert swap.present_value(NOMINAL, CURVE, jy) == pytest.approx(
        swap.inflation_leg(NOMINAL, CURVE, jy), rel=1e-14
    )  # zero fixed rate
