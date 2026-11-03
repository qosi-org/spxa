"""Bachelier (normal) implied volatility against 50-digit reference prices."""

import math

import mpmath as mp
import numpy as np
import pytest

from spxa.market import bachelier_price, implied_normal_volatility

mp.mp.dps = 50


def _bachelier_mp(F, K, T, sigma_n, option):
    """Bachelier price in 50-digit arithmetic, OTM part from psi(d) = d Phi(d) + phi(d)."""
    sd = mp.mpf(sigma_n) * mp.sqrt(T)
    a = abs(mp.mpf(F) - mp.mpf(K))
    d = -a / sd
    otm = sd * (d * mp.ncdf(d) + mp.npdf(d))
    eta = 1 if option == "call" else -1
    return otm + max(eta * (mp.mpf(F) - mp.mpf(K)), 0)


def test_round_trip_otm():
    worst, n = 0.0, 0
    for moneyness in np.linspace(-38.0, 38.0, 39):  # (F - K) in units of sd
        for sd in (1e-6, 1e-3, 0.02, 1.0, 50.0):
            F, K = 0.01 + moneyness * sd, 0.01
            opt = "call" if F <= K else "put"
            price = _bachelier_mp(F, K, 1.0, sd, opt)
            if price < mp.mpf("1e-300"):
                continue
            iv = implied_normal_volatility(float(price), F, K, 1.0, 1.0, opt)
            worst = max(worst, abs(iv / sd - 1.0))
            n += 1
    assert n > 150
    assert worst < 5e-14


def test_round_trip_itm_negative_rates_and_discounting():
    for F, K, T, s, opt in [
        (-0.004, -0.001, 2.0, 0.006, "put"),
        (0.03, 0.01, 10.0, 0.009, "call"),
        (-0.002, 0.002, 0.5, 0.004, "call"),
    ]:
        price = bachelier_price(F, K, T, s, 0.9, opt)
        assert implied_normal_volatility(price, F, K, T, 0.9, opt) == pytest.approx(s, rel=1e-12)


def test_atm_closed_form():
    sd = 0.0123
    price = sd / math.sqrt(2 * math.pi)
    assert implied_normal_volatility(price, 0.02, 0.02, 1.0) == pytest.approx(sd, rel=1e-15)
    assert implied_normal_volatility(price * 0.8, 0.02, 0.02, 4.0, 0.8) == pytest.approx(
        sd / 2, rel=1e-15
    )


def test_vectorised_and_bounds():
    K = np.array([-0.01, 0.0, 0.02, 0.05])
    sig = np.array([0.007, 0.008, 0.009, 0.011])
    prices = bachelier_price(0.015, K, 3.0, sig, 0.95, np.array([False, True, True, True]))
    iv = implied_normal_volatility(prices, 0.015, K, 3.0, 0.95, np.array([False, True, True, True]))
    assert iv == pytest.approx(sig, rel=1e-12)
    assert implied_normal_volatility(1.0, 3.0, 2.0, 1.0) == 0.0  # exactly intrinsic
    assert math.isnan(implied_normal_volatility(0.5, 3.0, 2.0, 1.0))
    with pytest.raises(ValueError, match="intrinsic"):
        implied_normal_volatility(0.5, 3.0, 2.0, 1.0, errors="raise")
    with pytest.raises(ValueError):
        implied_normal_volatility(0.01, 0.03, 0.02, -1.0)
