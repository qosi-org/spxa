"""Black, Bachelier and shifted-Black formulas."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from spxa.market import (
    bachelier_greeks,
    bachelier_price,
    black_greeks,
    black_price,
    black_scholes_greeks,
    black_scholes_price,
    shifted_black_greeks,
    shifted_black_price,
)
from spxa.pde.reference import bs_greeks, bs_price


def test_black_textbook_value():
    # at the money d1 = -d2 = sigma sqrt(T) / 2, so call and put coincide
    F, K, T, s, df = 20.0, 20.0, 4 / 12, 0.25, math.exp(-0.09 * 4 / 12)
    d1 = 0.5 * s * math.sqrt(T)
    expected = df * (F * norm.cdf(d1) - K * norm.cdf(-d1))
    assert black_price(F, K, T, s, df) == pytest.approx(expected, rel=1e-14)
    assert black_price(F, K, T, s, df, "put") == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("F, K", [(100.0, 80.0), (100.0, 100.0), (100.0, 130.0), (0.03, 0.025)])
def test_put_call_parity(F, K):
    for T, s, df in [(0.25, 0.2, 0.99), (5.0, 0.6, 0.8)]:
        c = black_price(F, K, T, s, df, "call")
        p = black_price(F, K, T, s, df, "put")
        assert c - p == pytest.approx(df * (F - K), abs=1e-12 * max(F, 1))
        cn = bachelier_price(F, K, T, s * F, df, "call")
        pn = bachelier_price(F, K, T, s * F, df, "put")
        assert cn - pn == pytest.approx(df * (F - K), abs=1e-12 * max(F, 1))


def test_black_scholes_matches_pde_reference():
    S, K, T, r, s, q = 100.0, np.array([80.0, 100.0, 120.0]), 1.3, 0.04, 0.27, 0.015
    for opt in ("call", "put"):
        assert black_scholes_price(S, K, T, r, s, q, opt) == pytest.approx(
            bs_price(S, K, T, r, s, q, option=opt), rel=1e-13
        )
        ours = black_scholes_greeks(S, 95.0, T, r, s, q, opt)
        ref = bs_greeks(S, 95.0, T, r, s, q, option=opt)
        for key in ("delta", "gamma", "vega"):
            assert ours[key] == pytest.approx(ref[key], rel=1e-12)


def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h), (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("K", [70.0, 100.0, 140.0])
def test_black_greeks_finite_differences(opt, K):
    F, T, s, df = 100.0, 1.7, 0.31, 0.93
    g = black_greeks(F, K, T, s, df, opt)
    d1, d2 = _fd(lambda x: black_price(x, K, T, s, df, opt), F, 1e-3)
    assert g["delta"] == pytest.approx(d1, rel=1e-7)
    assert g["gamma"] == pytest.approx(d2, rel=1e-4)
    v1, v2 = _fd(lambda x: black_price(F, K, T, x, df, opt), s, 1e-4)
    assert g["vega"] == pytest.approx(v1, rel=1e-7)
    assert g["volga"] == pytest.approx(v2, rel=1e-4, abs=1e-6)
    vanna, _ = _fd(lambda x: black_greeks(F, K, T, x, df, opt)["delta"], s, 1e-5)
    assert g["vanna"] == pytest.approx(vanna, rel=1e-6, abs=1e-9)
    k1, k2 = _fd(lambda x: black_price(F, x, T, s, df, opt), K, 1e-3)
    assert g["dual_delta"] == pytest.approx(k1, rel=1e-7)
    assert g["dual_gamma"] == pytest.approx(k2, rel=1e-4)
    t1, _ = _fd(lambda x: black_price(F, K, x, s, df, opt), T, 1e-5)
    assert g["theta"] == pytest.approx(-t1, rel=1e-7)


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("K", [-0.01, 0.02, 0.05])
def test_bachelier_greeks_finite_differences(opt, K):
    F, T, s, df = 0.02, 2.0, 0.008, 0.96
    g = bachelier_greeks(F, K, T, s, df, opt)
    d1, _ = _fd(lambda x: bachelier_price(x, K, T, s, df, opt), F, 1e-7)
    _, d2 = _fd(lambda x: bachelier_price(x, K, T, s, df, opt), F, 1e-5)
    assert g["delta"] == pytest.approx(d1, rel=1e-7)
    assert g["gamma"] == pytest.approx(d2, rel=1e-4)
    v1, v2 = _fd(lambda x: bachelier_price(F, K, T, x, df, opt), s, 1e-6)
    assert g["vega"] == pytest.approx(v1, rel=1e-7)
    assert g["volga"] == pytest.approx(v2, rel=1e-3, abs=1e-6)
    vanna, _ = _fd(lambda x: bachelier_greeks(F, K, T, x, df, opt)["delta"], s, 1e-7)
    assert g["vanna"] == pytest.approx(vanna, rel=1e-5, abs=1e-6)
    k1, _ = _fd(lambda x: bachelier_price(F, x, T, s, df, opt), K, 1e-7)
    assert g["dual_delta"] == pytest.approx(k1, rel=1e-7)
    t1, _ = _fd(lambda x: bachelier_price(F, K, x, s, df, opt), T, 1e-6)
    assert g["theta"] == pytest.approx(-t1, rel=1e-6)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_black_scholes_spot_greeks(opt):
    S, K, T, r, s, q = 100.0, 105.0, 0.8, 0.05, 0.22, 0.02
    g = black_scholes_greeks(S, K, T, r, s, q, opt)

    def price(**kw):
        return black_scholes_price(**{**dict(S=S, K=K, T=T, r=r, sigma=s, q=q), **kw}, option=opt)

    d1, d2 = _fd(lambda x: price(S=x), S, 1e-2)
    assert g["delta"] == pytest.approx(d1, rel=1e-7) and g["gamma"] == pytest.approx(d2, rel=1e-4)
    assert g["theta"] == pytest.approx(-_fd(lambda x: price(T=x), T, 1e-5)[0], rel=1e-7)
    assert g["rho"] == pytest.approx(_fd(lambda x: price(r=x), r, 1e-6)[0], rel=1e-7)
    assert g["epsilon"] == pytest.approx(_fd(lambda x: price(q=x), q, 1e-6)[0], rel=1e-7)
    assert g["vega"] == pytest.approx(_fd(lambda x: price(sigma=x), s, 1e-6)[0], rel=1e-7)


def test_bachelier_atm_and_small_vol_limit():
    F, T, sn, df = 0.03, 3.0, 0.01, 0.9
    assert bachelier_price(F, F, T, sn, df) == pytest.approx(df * sn * math.sqrt(T / (2 * math.pi)))
    # lognormal and normal prices agree to O(sigma^3) near the money with sigma_N = sigma F
    F, K, T, s = 100.0, 101.0, 0.5, 0.01
    assert black_price(F, K, T, s) == pytest.approx(
        bachelier_price(F, K, T, s * math.sqrt(F * K)), rel=1e-4
    )


def test_shifted_black():
    F, K, T, s, shift = -0.002, 0.001, 2.0, 0.15, 0.03
    assert shifted_black_price(F, K, T, s, shift) == pytest.approx(
        black_price(F + shift, K + shift, T, s)
    )
    g = shifted_black_greeks(F, K, T, s, shift, 0.97, "put")
    d1, _ = _fd(lambda x: shifted_black_price(x, K, T, s, shift, 0.97, "put"), F, 1e-6)
    assert g["delta"] == pytest.approx(d1, rel=1e-6)
    with pytest.raises(ValueError, match="positive forward"):
        black_price(F, K, T, s)


def test_degenerate_cases_and_broadcasting():
    assert black_price(100.0, 90.0, 0.0, 0.2, 0.95) == pytest.approx(9.5)
    assert black_price(100.0, 110.0, 1.0, 0.0, 0.95, "put") == pytest.approx(9.5)
    assert black_price(100.0, 0.0, 1.0, 0.2, 0.95) == pytest.approx(95.0)
    assert black_price(100.0, 0.0, 1.0, 0.2, 0.95, "put") == 0.0
    assert bachelier_price(0.01, 0.02, 0.0, 0.01, 1.0, "put") == pytest.approx(0.01)
    g = black_greeks(100.0, 90.0, 0.0, 0.2)
    assert g["delta"] == 1.0 and g["gamma"] == 0.0 and g["vega"] == 0.0
    K = np.array([[80.0, 100.0, 120.0]])
    T = np.array([[0.5], [1.0]])
    out = black_price(100.0, K, T, 0.2, 1.0, np.array([True, False, True]))
    assert out.shape == (2, 3)
    assert out[1, 1] == pytest.approx(black_price(100.0, 100.0, 1.0, 0.2, 1.0, "put"))
    assert black_price(100.0, 100.0, 1.0, 0.2, 1.0, np.array(["c", "p"])).shape == (2,)
    assert black_price(100.0, 100.0, 1.0, 0.2, 1.0, -1) == black_price(
        100.0, 100.0, 1.0, 0.2, 1.0, "put"
    )
    for bad in ("straddle", 0):
        with pytest.raises(ValueError):
            black_price(100.0, 100.0, 1.0, 0.2, 1.0, bad)
    with pytest.raises(ValueError):
        black_price(100.0, -1.0, 1.0, 0.2)
    with pytest.raises(ValueError):
        bachelier_price(0.0, 0.0, 1.0, -0.01)


def test_extreme_strikes_are_finite():
    K = np.array([1e-8, 1e-3, 1e3, 1e8])
    c = black_price(1.0, K, 1.0, 0.3)
    assert np.all(np.isfinite(c)) and np.all(c >= 0)
    assert c[0] == pytest.approx(1.0 - 1e-8, abs=1e-15) and c[-1] == pytest.approx(0.0, abs=1e-300)
