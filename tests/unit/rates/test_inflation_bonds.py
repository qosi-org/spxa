"""Inflation-linked bonds: Treasury's TIPS example, curve values and the deflation floor."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from spxa.market.curves import FlatCurve
from spxa.rates.inflation import CPIFixings, InflationIndex, bootstrap_inflation_curve
from spxa.rates.inflation_bonds import InflationLinkedBond
from spxa.rates.jarrow_yildirim import JarrowYildirim
from spxa.rates.jy_mc import jy_paths

# TreasuryDirect, "Price Calculations for a Newly Issued TIPS": 3.875% of 15 Jan 2009
CPI_1998 = InflationIndex(
    "CPI-U",
    CPIFixings.from_pairs("CPI-U", [("1998-10-01", 164.0), ("1998-11-01", 164.0)]),
    digits=5,
)
TIPS_2009 = InflationLinkedBond(0.03875, dt.date(2009, 1, 15), dt.date(1999, 1, 15), CPI_1998)


def test_treasury_example() -> None:
    assert len(TIPS_2009.coupon_dates()) == 20 and TIPS_2009.base == 164.0
    assert TIPS_2009.real_price("1999-01-15", 0.03898) == pytest.approx(
        99.811029750363067, abs=1e-12
    )
    assert TIPS_2009.settlement_amount("1999-01-15", 0.03898) == 99.811030
    assert TIPS_2009.accrued("1999-01-15") == 0.0
    assert TIPS_2009.real_yield("1999-01-15", 99.811029750363067) == pytest.approx(
        0.03898, abs=1e-13
    )


def test_accrual_and_index_ratio() -> None:
    cpi = InflationIndex(
        "CPI-U",
        CPIFixings.from_pairs(
            "CPI-U",
            [
                ("1998-10-01", 164.0),
                ("1998-11-01", 164.0),
                ("1999-01-01", 164.3),
                ("1999-02-01", 164.5),
            ],
        ),
        digits=5,
    )
    bond = InflationLinkedBond(0.03875, dt.date(2009, 1, 15), dt.date(1999, 1, 15), cpi)
    settle = dt.date(1999, 4, 20)  # 95 of 181 days into the first period
    assert bond.accrued(settle) == pytest.approx(95 / 181 * 1.9375, abs=1e-14)
    ratio = cpi.index_ratio(settle, "1999-01-15")
    assert ratio == round((164.3 + 19 / 30 * 0.2) / 164.0, 5)
    p, a = round(bond.real_price(settle, 0.035), 6), round(bond.accrued(settle), 6)
    assert bond.settlement_amount(settle, 0.035) == round(
        round(p * ratio, 6) + round(a * ratio, 6), 6
    )
    # zero real yield: price is the undiscounted real cash flows (minus accrued, with simple first-period interest)
    assert bond.real_price(settle, 0.0) == pytest.approx(
        1.9375 * 20 + 100 - 95 / 181 * 1.9375, abs=1e-12
    )
    assert bond.breakeven_inflation(0.045, 0.02) == pytest.approx(
        (1.0225 / 1.01) ** 2 - 1, rel=1e-14
    )


LEVELS = 300.0 * np.exp(0.002 * np.arange(28))
CPI = InflationIndex("CPI-U", CPIFixings.from_series("CPI-U", "2023-01-01", LEVELS), lag=3)
TODAY = dt.date(2025, 4, 1)
NOMINAL = FlatCurve(0.04, reference_date=TODAY)


def _setup(zc: float) -> tuple[InflationLinkedBond, object, JarrowYildirim]:
    curve = bootstrap_inflation_curve(CPI, TODAY, {1: zc, 5: zc, 10: zc})
    bond = InflationLinkedBond(0.00125, dt.date(2035, 4, 1), TODAY, CPI)
    real = curve.real_curve(NOMINAL, TODAY)
    model = JarrowYildirim(
        NOMINAL, real, curve.reference_cpi(TODAY), 0.05, 0.01, 0.05, 0.008, 0.012, 0.6, 0.2, -0.3
    )
    return bond, curve, model


def test_curve_value_and_deflation_floor() -> None:
    bond, curve, model = _setup(0.02)
    pv = bond.present_value(NOMINAL, curve, TODAY)
    real = curve.real_curve(NOMINAL, TODAY)
    expected = sum(0.0625 * float(real.discount(d)) for d in bond.coupon_dates()) + 100 * float(
        real.discount(bond.maturity)
    )
    assert pv["coupons"] + pv["principal"] == pytest.approx(expected, rel=1e-13)
    assert pv["floor"] == 0.0  # 2% inflation: the floor is out of the money on the forward
    with_model = bond.present_value(NOMINAL, curve, TODAY, model)
    assert 0 < with_model["floor"] < 1.0
    # lower expected inflation makes the floor dearer
    low_bond, low_curve, low_model = _setup(0.0025)
    assert (
        low_bond.present_value(NOMINAL, low_curve, TODAY, low_model)["floor"]
        > 3 * with_model["floor"]
    )
    assert [
        round(with_model["floor"], 3),
        round(low_bond.present_value(NOMINAL, low_curve, TODAY, low_model)["floor"], 3),
    ] == [0.455, 3.204]


def test_floor_against_jarrow_yildirim_monte_carlo() -> None:
    bond, curve, model = _setup(0.0025)
    floor = bond.present_value(NOMINAL, curve, TODAY, model)["floor"]
    T = float(NOMINAL.time(bond.maturity))
    paths = jy_paths(model, T, 200, 100_000, seed=4)
    c = paths.column(T)
    pay = 100 * paths.discount[:, c] * np.maximum(1.0 - paths.index[:, c] / bond.base, 0.0)
    assert abs(pay.mean() - floor) < 3 * pay.std() / math.sqrt(pay.size)
