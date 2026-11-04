"""Inflation curves bootstrapped from zero-coupon inflation swaps."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from spxa.market.curves import FlatCurve
from spxa.market.dates import add_months
from spxa.rates.inflation import (
    CPIFixings,
    InflationIndex,
    Seasonality,
    bootstrap_inflation_curve,
)

# a euro-style index: three-month lag, no interpolation (ZCIS convention)
LEVELS = 118.0 * np.exp(0.0025 * np.arange(27))
HICP = InflationIndex(
    "HICPxT", CPIFixings.from_series("HICPxT", "2023-01-01", LEVELS), lag=3, interpolated=False
)
TODAY = dt.date(2025, 4, 1)
QUOTES = {1: 0.0210, 2: 0.0205, 3: 0.0200, 5: 0.0198, 10: 0.0210, 30: 0.0230}
# January and July sales show as negative seasonal increments
SEASON = Seasonality(
    (-0.006, 0.004, 0.003, 0.002, 0.001, 0.0, -0.004, 0.001, 0.002, 0.0, -0.001, 0.003)
)


@pytest.mark.parametrize("seasonality", [None, SEASON])
def test_reprices_every_quote(seasonality: Seasonality | None) -> None:
    curve = bootstrap_inflation_curve(HICP, TODAY, QUOTES, seasonality)
    assert curve.base_month == dt.date(2025, 1, 1)
    for n, k in QUOTES.items():
        assert curve.zero_coupon_rate(TODAY, add_months(TODAY, 12 * n)) == pytest.approx(
            k, abs=1e-14
        )


def test_interpolation_and_seasonality() -> None:
    plain = bootstrap_inflation_curve(HICP, TODAY, QUOTES)
    seasonal = bootstrap_inflation_curve(HICP, TODAY, QUOTES, SEASON)
    base = plain.base_level
    # log-linear in months between the 1y and 2y pillars
    i1, i2 = base * 1.021, base * 1.0205**2
    assert plain.fixing("2026-07-01") == pytest.approx(math.sqrt(i1 * i2), rel=1e-14)
    # seasonality nets to zero over a year, and moves months inside it
    assert sum(SEASON.factors) == pytest.approx(0.0, abs=1e-15)
    assert SEASON.adjustment("2026-01-01", "2025-01-01") == pytest.approx(1.0, abs=1e-15)
    expected = math.exp(sum(SEASON.factors[:6]))  # January -> July
    assert seasonal.fixing("2025-07-01") / plain.fixing("2025-07-01") == pytest.approx(
        expected, rel=1e-14
    )
    assert SEASON.adjustment("2024-07-01", "2025-01-01") == pytest.approx(
        math.exp(-sum(SEASON.factors[6:])), rel=1e-14
    )
    # published months come from the fixings
    assert plain.fixing("2024-03-01") == HICP.fixings["2024-03-01"]
    # flat beyond the last pillar
    m30 = add_months(plain.base_month, 360)
    assert plain.fixing(add_months(m30, 12)) / plain.fixing(m30) == pytest.approx(
        plain.fixing(m30) / plain.fixing(add_months(m30, -12)), rel=1e-12
    )


def test_real_curve_fisher_relation() -> None:
    curve = bootstrap_inflation_curve(HICP, TODAY, QUOTES)
    nominal = FlatCurve(0.03, reference_date=TODAY)
    real = curve.real_curve(nominal, TODAY)
    for n, k in QUOTES.items():
        m = add_months(TODAY, 12 * n)
        # one unit of index-linked notional at M is worth P_n(M) (1 + K)^n today
        assert real.discount(m) == pytest.approx(nominal.discount(m) * (1 + k) ** n, rel=1e-13)
    # real rate ~ nominal minus inflation
    t = real.time(add_months(TODAY, 120))
    assert real.zero_rate(t) == pytest.approx(0.03 - math.log(1.021), abs=1e-3)


def test_interpolated_index_needs_month_start() -> None:
    cpi = InflationIndex("CPI-U", HICP.fixings, lag=3, interpolated=True)
    curve = bootstrap_inflation_curve(cpi, TODAY, {1: 0.025, 2: 0.024})
    assert curve.zero_coupon_rate(TODAY, "2026-04-01") == pytest.approx(0.025, abs=1e-14)
    # mid-month: interpolated between forward months
    mid = curve.reference_cpi("2026-04-16")
    assert curve.fixing("2026-01-01") < mid < curve.fixing("2026-02-01")
    with pytest.raises(ValueError):
        bootstrap_inflation_curve(cpi, "2025-04-15", {1: 0.025})
    with pytest.raises(ValueError):
        Seasonality((0.0,) * 11)
