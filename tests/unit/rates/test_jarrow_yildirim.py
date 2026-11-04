"""Jarrow–Yildirim: curves, year-on-year convexity and (later) options."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.market.curves import FlatCurve
from spxa.rates.jarrow_yildirim import (
    InflationOptionQuote,
    JarrowYildirim,
    calibrate_jarrow_yildirim,
)
from spxa.rates.jy_mc import jy_paths

NOMINAL, REAL = FlatCurve(0.03), FlatCurve(0.01)
PARAMS = dict(
    a_n=0.1,
    sigma_n=0.01,
    a_r=0.05,
    sigma_r=0.008,
    sigma_i=0.012,
    rho_nr=0.6,
    rho_ni=0.2,
    rho_ri=-0.3,
)
JY = JarrowYildirim(NOMINAL, REAL, 100.0, **PARAMS)


def _mercurio(m: JarrowYildirim, t1: float, t2: float) -> float:
    """Mercurio (2005), eq. (16), written independently of the implementation."""

    def B(a: float, tau: float) -> float:
        return (1 - math.exp(-a * tau)) / a

    an, ar, sn, sr, si = m.a_n, m.a_r, m.sigma_n, m.sigma_r, m.sigma_i
    br, bn = B(ar, t1), B(an, t1)
    k = m.rho_nr * sn / (an + ar)
    return sr * B(ar, t2 - t1) * (br * (m.rho_ri * si - 0.5 * sr * br + k * (1 + ar * bn)) - k * bn)


def test_curves_fix_zero_coupon_swaps() -> None:
    assert JY.forward_index(5.0) == pytest.approx(100.0 * math.exp(0.1), rel=1e-14)
    for T in (1.0, 5.0, 30.0):
        assert JY.zero_coupon_swap_rate(T) == pytest.approx(math.expm1(0.02), rel=1e-13)


def test_yoy_convexity_matches_mercurio() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = dict(
            a_n=rng.uniform(0.01, 0.5),
            sigma_n=rng.uniform(0.0, 0.02),
            a_r=rng.uniform(0.01, 0.5),
            sigma_r=rng.uniform(0.0, 0.02),
            sigma_i=rng.uniform(0.0, 0.03),
            rho_nr=rng.uniform(-0.5, 0.5),
            rho_ni=rng.uniform(-0.3, 0.3),
            rho_ri=rng.uniform(-0.3, 0.3),
        )
        m = JarrowYildirim(NOMINAL, REAL, 100.0, **p)
        t1 = rng.uniform(0.1, 20.0)
        t2 = t1 + rng.uniform(0.25, 2.0)
        assert m.yoy_convexity(t1, t2) == pytest.approx(_mercurio(m, t1, t2), abs=1e-15)
    assert JY.yoy_convexity(0.0, 1.0) == 0.0  # the first period is a zero-coupon swap
    assert [round(JY.yoy_convexity(t, t + 1) * 1e4, 2) for t in (1.0, 4.0, 9.0)] == [
        -0.35,
        -2.23,
        -7.62,
    ]


def test_year_on_year_swaps() -> None:
    flat = JarrowYildirim(NOMINAL, REAL, 100.0, **{**PARAMS, "sigma_r": 0.0})
    years = np.arange(1.0, 11.0)
    # without real-rate volatility each period is a forward ratio e^{0.02}
    assert flat.yoy_swap_rate(years) == pytest.approx(math.expm1(0.02), rel=1e-13)
    assert flat.yoy_swaplet(3.0, 4.0) == pytest.approx(
        math.exp(-0.12) * math.expm1(0.02), rel=1e-13
    )
    # the convexity lowers the YoY rate below the zero-coupon one here (C < 0)
    assert JY.yoy_swap_rate(years) < flat.yoy_swap_rate(years)
    assert round((JY.yoy_swap_rate(years) - flat.yoy_swap_rate(years)) * 1e4, 2) == -2.98
    with pytest.raises(ValueError):
        JY.yoy_convexity(2.0, 1.0)
    with pytest.raises(ValueError):
        JarrowYildirim(
            NOMINAL, REAL, 100.0, **{**PARAMS, "rho_nr": 0.99, "rho_ni": 0.99, "rho_ri": -0.99}
        )


def test_monte_carlo_reprices_curves_and_yoy_convexity() -> None:
    paths = jy_paths(JY, 10.0, 200, 100_000, record=(1.0, 5.0, 9.0), seed=1)
    n = paths.discount.shape[0]
    for T in (1.0, 5.0, 10.0):
        c = paths.column(T)
        d, idx = paths.discount[:, c], paths.index[:, c] / 100.0
        assert abs(d.mean() - math.exp(-0.03 * T)) < 3 * d.std() / math.sqrt(n)
        assert abs((d * idx).mean() - math.exp(-0.01 * T)) < 3 * (d * idx).std() / math.sqrt(n)
    c9, c10 = paths.column(9.0), paths.column(10.0)
    pay = paths.discount[:, c10] * (paths.index[:, c10] / paths.index[:, c9] - 1.0)
    se = pay.std() / math.sqrt(n)
    assert abs(pay.mean() - JY.yoy_swaplet(9.0, 10.0)) < 3 * se
    naive = math.exp(-0.3) * math.expm1(0.02)  # the swaplet without convexity
    assert abs(pay.mean() - naive) > 8 * se
    with pytest.raises(KeyError):
        paths.column(2.5)


def test_caplets_closed_form_parity_and_limits() -> None:
    # put-call parity on both ratios
    for k in (-0.01, 0.0, 0.02, 0.04):
        c, p = JY.yoy_caplet(4.0, 5.0, k), JY.yoy_caplet(4.0, 5.0, k, "put")
        assert c - p == pytest.approx(JY.yoy_swaplet(4.0, 5.0) - k * math.exp(-0.15), abs=1e-15)
        zc, zf = JY.zero_coupon_caplet(10.0, k), JY.zero_coupon_caplet(10.0, k, "put")
        fwd = math.exp(-0.3) * (JY.forward_index(10.0) / 100.0 - (1 + k) ** 10)
        assert zc - zf == pytest.approx(fwd, abs=1e-14)
    # only the index volatility left: Var log I(T2)/I(T1) = sigma_I^2 (T2 - T1)
    index_only = JarrowYildirim(NOMINAL, REAL, 100.0, **{**PARAMS, "sigma_n": 0.0, "sigma_r": 0.0})
    assert index_only.log_ratio_variance(4.0, 5.5) == pytest.approx(0.012**2 * 1.5, rel=1e-13)
    # rate volatility adds to the zero-coupon variance; over ten years it dominates the index's
    assert JY.log_ratio_variance(0.0, 10.0) > 10 * 0.012**2
    assert round(JY.log_ratio_variance(0.0, 10.0), 4) == 0.0184


def test_caplets_against_monte_carlo() -> None:
    paths = jy_paths(JY, 10.0, 200, 100_000, record=(4.0, 5.0, 9.0), seed=2)
    n = paths.discount.shape[0]

    def check(pay: np.ndarray, closed: float) -> None:
        assert abs(pay.mean() - closed) < 3 * pay.std() / math.sqrt(n)

    for t1, t2, k in [(4.0, 5.0, 0.02), (9.0, 10.0, 0.0), (9.0, 10.0, 0.03)]:
        c1, c2 = paths.column(t1), paths.column(t2)
        ratio = paths.index[:, c2] / paths.index[:, c1]
        check(paths.discount[:, c2] * np.maximum(ratio - 1 - k, 0.0), JY.yoy_caplet(t1, t2, k))
    c = paths.column(10.0)
    growth = paths.index[:, c] / 100.0
    check(
        paths.discount[:, c] * np.maximum(1.0 - growth, 0.0),
        JY.zero_coupon_caplet(10.0, 0.0, "put"),
    )
    check(
        paths.discount[:, c] * np.maximum(growth - 1.02**10, 0.0), JY.zero_coupon_caplet(10.0, 0.02)
    )


def test_calibration_recovers_volatilities() -> None:
    quotes = [
        InflationOptionQuote(t - 1, t, k, JY.yoy_caplet(t - 1, t, k))
        for t in (1, 2, 3, 5, 7, 10)
        for k in (0.01, 0.03)
    ]
    floors = [
        InflationOptionQuote(0, t, 0.0, JY.zero_coupon_caplet(t, 0.0, "put"), "put", "zc")
        for t in (5, 10)
    ]
    start = JarrowYildirim(NOMINAL, REAL, 100.0, **{**PARAMS, "sigma_i": 0.02, "sigma_r": 0.003})
    fit = calibrate_jarrow_yildirim(start, quotes + floors)
    assert fit.success and fit.rmse < 1e-14
    assert fit.parameters == pytest.approx({"sigma_i": 0.012, "sigma_r": 0.008}, abs=1e-12)
    # from a single start the search stalls in a local minimum of the real-rate volatility
    stuck = calibrate_jarrow_yildirim(start, quotes + floors, starts=(1.0,))
    assert stuck.rmse > 1e-4 and stuck.parameters["sigma_r"] < 0.002
    # zero-coupon floors alone pin the index volatility when the rate parameters are known
    zc = calibrate_jarrow_yildirim(
        JarrowYildirim(NOMINAL, REAL, 100.0, **{**PARAMS, "sigma_i": 0.02}), floors, ("sigma_i",)
    )
    assert zc.parameters["sigma_i"] == pytest.approx(0.012, abs=1e-12)
    with pytest.raises(ValueError):
        calibrate_jarrow_yildirim(start, floors, ("kappa",))
