"""
Inflation with spxa.rates.

1. Reference CPI and a TIPS settlement amount from TreasuryDirect's example.
2. An inflation curve bootstrapped from zero-coupon swaps, with seasonality
   estimated from fixing history, and the implied real curve.
3. Jarrow-Yildirim: year-on-year convexity, caplets checked by Monte Carlo
   and a calibration to cap prices.
4. A ten-year linker and the value of its deflation floor.

Run with: python examples/rates_inflation_example.py
"""

import datetime as dt
import math

import numpy as np

from spxa.market import FlatCurve
from spxa.rates import (
    CPIFixings,
    InflationIndex,
    InflationLinkedBond,
    InflationOptionQuote,
    JarrowYildirim,
    Seasonality,
    YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
    bootstrap_inflation_curve,
    calibrate_jarrow_yildirim,
    jy_paths,
)

TODAY = dt.date(2025, 4, 1)


def main() -> None:
    print("1. Reference CPI (CPI-U, three-month lag, daily interpolation)")
    cpi_u = CPIFixings.from_pairs("CPI-U", [("2022-10-01", 298.012), ("2022-11-01", 297.711)])
    print(
        f"   Ref CPI 2023-01-15: {InflationIndex('CPI-U', cpi_u, digits=5).reference_cpi('2023-01-15'):.5f}"
    )
    old = InflationIndex(
        "CPI-U",
        CPIFixings.from_pairs("CPI-U", [("1998-10-01", 164.0), ("1998-11-01", 164.0)]),
        digits=5,
    )
    tips = InflationLinkedBond(0.03875, dt.date(2009, 1, 15), dt.date(1999, 1, 15), old)
    print(
        f"   3.875% TIPS 2009 at a 3.898% real yield: {tips.settlement_amount('1999-01-15', 0.03898):.6f}"
    )

    print("\n2. Curve from zero-coupon swaps")
    rng = np.random.default_rng(1)
    pattern = np.array(
        [-0.006, 0.004, 0.003, 0.002, 0.001, 0.0, -0.004, 0.001, 0.002, 0.0, -0.001, 0.003]
    )
    logs = np.cumsum(
        np.r_[
            math.log(250.0),
            math.log(1.025) / 12
            + np.resize(pattern - pattern.mean(), 147)
            + 2e-4 * rng.standard_normal(147),
        ]
    )
    history = CPIFixings.from_series("CPI", "2013-01-01", np.exp(logs))  # through March 2025
    season = Seasonality.estimate(history, years=8)
    index = InflationIndex("CPI", history, lag=3, interpolated=False)
    quotes = {1: 0.0240, 2: 0.0232, 3: 0.0228, 5: 0.0225, 10: 0.0230, 20: 0.0238}
    curve = bootstrap_inflation_curve(index, TODAY, quotes, season)
    nominal = FlatCurve(0.04, reference_date=TODAY)
    real = curve.real_curve(nominal, TODAY)
    print(f"   seasonal factors (Jan, Jul) {season.factors[0]:+.4f} {season.factors[6]:+.4f}")
    worst = max(
        abs(ZeroCouponInflationSwap(TODAY, n, k).present_value(nominal, curve))
        for n, k in quotes.items()
    )
    print(f"   every ZCIS reprices to 1e-12: {worst < 1e-12}")
    for n in (2, 5, 10):
        t = float(real.time(dt.date(2025 + n, 4, 1)))
        print(f"   {n:>2}y real zero rate {real.zero_rate(t):.4%}")

    print("\n3. Jarrow-Yildirim on these curves")
    jy = JarrowYildirim(
        nominal, real, curve.reference_cpi(TODAY), 0.1, 0.01, 0.05, 0.008, 0.012, 0.6, 0.2, -0.3
    )
    yoy = YearOnYearInflationSwap(TODAY, 10, 0.0)
    plain, convex = yoy.fair_rate(nominal, curve), yoy.fair_rate(nominal, curve, jy)
    print(
        f"   10y YoY rate {convex:.4%} vs {plain:.4%} without convexity ({(convex - plain) * 1e4:+.2f} bp)"
    )
    paths = jy_paths(jy, 5.0, 100, 100_000, record=(4.0,), seed=3)
    c4, c5 = paths.column(4.0), paths.column(5.0)
    pay = paths.discount[:, c5] * np.maximum(paths.index[:, c5] / paths.index[:, c4] - 1.025, 0.0)
    se = pay.std() / math.sqrt(pay.size)
    closed = jy.yoy_caplet(4.0, 5.0, 0.025)
    print(f"   4y-5y 2.5% YoY caplet {closed:.6f}, Monte Carlo {pay.mean():.6f} +/- {se:.6f}")
    caps = [
        InflationOptionQuote(t - 1, t, k, jy.yoy_caplet(t - 1, t, k))
        for t in (2, 3, 5, 7, 10)
        for k in (0.01, 0.03)
    ]
    start = JarrowYildirim(nominal, real, jy.cpi0, 0.1, 0.01, 0.05, 0.003, 0.02, 0.6, 0.2, -0.3)
    fit = calibrate_jarrow_yildirim(start, caps)
    print(
        f"   calibration from (2.0%, 0.3%): sigma_I {fit.parameters['sigma_i']:.4%}, sigma_r {fit.parameters['sigma_r']:.4%}, rmse below 1e-12 {fit.rmse < 1e-12}"
    )

    print("\n4. A new ten-year 0.125% linker")
    bond = InflationLinkedBond(
        0.00125, dt.date(2035, 4, 1), TODAY, InflationIndex("CPI", history, lag=3)
    )
    pv = bond.present_value(nominal, curve, TODAY, jy)
    print(
        f"   coupons {pv['coupons']:.4f}  principal {pv['principal']:.4f}  deflation floor {pv['floor']:.4f}  total {pv['total']:.4f}"
    )


if __name__ == "__main__":
    main()
