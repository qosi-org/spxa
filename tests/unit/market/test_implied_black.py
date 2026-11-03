"""Black implied volatility: round trips against 50-digit reference prices."""

import math

import mpmath as mp
import numpy as np
import pytest

from spxa.market import black_price, implied_black_volatility, normalised_black_call

mp.mp.dps = 50


def _b_mp(x, s):
    x, s = mp.mpf(x), mp.mpf(s)
    return mp.e ** (x / 2) * mp.ncdf(x / s + s / 2) - mp.e ** (-x / 2) * mp.ncdf(x / s - s / 2)


def _black_mp(F, K, T, sigma, option):
    """Black price in 50-digit arithmetic; puts via the symmetry p(x) = b(-x) (no cancellation)."""
    x = mp.log(mp.mpf(F) / mp.mpf(K))
    s = mp.mpf(sigma) * mp.sqrt(T)
    return mp.sqrt(mp.mpf(F) * K) * _b_mp(x if option == "call" else -x, s)


@pytest.mark.parametrize("x", [-40.0, -12.0, -3.0, -0.5, -1e-3, -1e-9, 0.0, 0.7, 9.0])
def test_normalised_call_against_mpmath(x):
    for s in (1e-5, 1e-3, 0.03, 0.2, 1.0, 3.0, 10.0):
        ref = _b_mp(x, s)
        if ref < mp.mpf("1e-290"):
            continue
        got = normalised_black_call(x, s)
        assert got == pytest.approx(float(ref), rel=5e-13), (x, s)


def test_normalised_call_vectorised_and_limits():
    out = normalised_black_call(np.array([-1.0, 0.0, 1.0]), np.array([[0.0], [0.5]]))
    assert out.shape == (2, 3)
    assert out[0] == pytest.approx([0.0, 0.0, math.exp(0.5) - math.exp(-0.5)])
    assert normalised_black_call(-2.0, 400.0) == pytest.approx(math.exp(-1.0))
    with pytest.raises(ValueError):
        normalised_black_call(0.0, -1.0)


def _otm_cases():
    for x in np.linspace(-15.0, 15.0, 31):
        for s in (1e-4, 1e-3, 0.01, 0.1, 0.4, 1.0, 2.5, 6.0, 15.0):
            yield float(x), s


def test_round_trip_otm():
    """Out-of-the-money prices recover sigma to ~machine precision."""
    worst, count = 0.0, 0
    for x, s in _otm_cases():
        F, K, T = math.exp(x), 1.0, 1.0
        opt = "call" if x <= 0 else "put"
        price = _black_mp(F, K, T, s, opt)
        cap = F if opt == "call" else K
        # skip prices that are not representable, or numerically equal to the upper bound
        if price < mp.mpf("1e-300") or (cap - price) < mp.mpf("1e-10") * cap:
            continue
        iv = implied_black_volatility(float(price), F, K, T, 1.0, opt)
        worst = max(worst, abs(iv / s - 1.0))
        count += 1
    assert count > 100
    assert worst < 1e-13


def test_round_trip_in_the_money():
    """ITM prices: accuracy limited only by the rounding of the premium itself."""
    for x in (-6.0, -1.0, 1.0, 6.0):
        for s in (0.05, 0.3, 1.5):
            F, K, T = 100.0 * math.exp(x), 100.0, 2.0
            sigma = s / math.sqrt(T)
            opt = "call" if x > 0 else "put"
            price = float(_black_mp(F, K, T, sigma, opt))
            intrinsic = max(F - K, 0.0) if opt == "call" else max(K - F, 0.0)
            if (price - intrinsic) < 1e-6 * price:
                continue
            # error bound: eps * price / (vega * sigma)
            iv = implied_black_volatility(price, F, K, T, 1.0, opt)
            assert iv == pytest.approx(sigma, rel=1e-9)


def test_discounting_and_expiry_scaling():
    F, K, T, df, sigma = 0.035, 0.03, 7.5, 0.82, 0.37
    price = black_price(F, K, T, sigma, df, "call")
    assert implied_black_volatility(price, F, K, T, df, "call") == pytest.approx(sigma, rel=1e-13)
    assert implied_black_volatility(
        black_price(F, K, T, sigma, df, "put"), F, K, T, df, "put"
    ) == pytest.approx(sigma, rel=1e-12)


def test_at_the_money_closed_form():
    s = 0.6
    beta = math.erf(s / (2 * math.sqrt(2)))
    assert implied_black_volatility(beta, 1.0, 1.0, 1.0) == pytest.approx(s, rel=1e-15)


def test_vectorised():
    K = np.array([80.0, 100.0, 125.0])
    T = np.array([[0.25], [3.0]])
    sig = np.array([[0.4, 0.2, 0.25], [0.18, 0.21, 0.3]])
    opts = np.array([False, True, True])
    prices = black_price(100.0, K, T, sig, 0.97, opts)
    iv = implied_black_volatility(prices, 100.0, K, T, 0.97, opts)
    assert iv.shape == (2, 3)
    assert iv == pytest.approx(sig, rel=1e-12)


def test_bounds_and_errors():
    F, K, T = 100.0, 90.0, 1.0
    assert implied_black_volatility(10.0, F, K, T) == 0.0  # exactly intrinsic
    assert math.isnan(implied_black_volatility(9.0, F, K, T))  # below intrinsic
    assert math.isnan(implied_black_volatility(100.5, F, K, T))  # above the forward
    assert math.isnan(implied_black_volatility(91.0, F, K, T, option="put"))  # above the strike
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_black_volatility(9.0, F, K, T, errors="raise")
    with pytest.raises(ValueError):
        implied_black_volatility(10.0, F, K, T, errors="ignore")
    with pytest.raises(ValueError):
        implied_black_volatility(10.0, F, -K, T)
    with pytest.raises(ValueError):
        implied_black_volatility(10.0, F, K, 0.0)
