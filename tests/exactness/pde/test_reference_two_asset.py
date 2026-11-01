"""Exactness tests for the two-asset closed-form references."""

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from spxa.pde.reference import bs_price, margrabe_price, spread_option_price

pytestmark = pytest.mark.exactness


def spread_by_cubature(S1, S2, K, T, r, s1, s2, rho, q1=0.0, q2=0.0):
    """Independent reference: nested adaptive quadrature of the discounted payoff.

    The inner integral over the idiosyncratic factor starts at the exercise
    point, located by root finding, so the integrand is smooth.
    """
    sT, c = np.sqrt(T), np.sqrt(1 - rho**2)

    def inner(z):
        x2 = np.log(S2) + (r - q2 - 0.5 * s2**2) * T + s2 * sT * z
        m1 = np.log(S1) + (r - q1 - 0.5 * s1**2) * T + s1 * sT * rho * z
        level = np.exp(x2) + K
        if level <= 0.0:
            w0 = -12.0
        else:
            w0 = (np.log(level) - m1) / (s1 * sT * c)
        if w0 >= 12.0:
            return 0.0

        def g(w):
            return (np.exp(m1 + s1 * sT * c * w) - level) * norm.pdf(w)

        return integrate.quad(g, max(w0, -12.0), 12.0, epsabs=1e-13, epsrel=1e-12)[0]

    val = integrate.quad(lambda z: inner(z) * norm.pdf(z), -12.0, 12.0, epsabs=1e-12, limit=200)[0]
    return np.exp(-r * T) * val


@pytest.mark.parametrize("K, rho", [(0.0, 0.4), (5.0, -0.3), (12.0, 0.8)])
def test_spread_against_cubature(K, rho):
    args = (100.0, 95.0, K, 0.7, 0.04, 0.3, 0.2, rho, 0.01, 0.02)
    assert spread_option_price(*args) == pytest.approx(spread_by_cubature(*args), abs=1e-7)


def test_margrabe_identities():
    a = margrabe_price(100.0, 90.0, 1.0, 0.2, 0.3, 0.4)
    b = margrabe_price(90.0, 100.0, 1.0, 0.3, 0.2, 0.4)
    assert a - b == pytest.approx(10.0, abs=1e-12)  # exchange parity
    assert spread_option_price(100.0, 90.0, 0.0, 1.0, 0.07, 0.2, 0.3, 0.4) == pytest.approx(
        a, abs=1e-10
    )
    # asset 2 deterministic in the limit: Black–Scholes with strike S2 e^{(r - q2) T}
    r, q2 = 0.03, 0.01
    ref = bs_price(100.0, 90.0 * np.exp((r - q2) * 1.0), 1.0, r, 0.2, 0.0)
    assert margrabe_price(100.0, 90.0, 1.0, 0.2, 1e-9, 0.0, 0.0, q2) == pytest.approx(ref, rel=1e-8)


def test_spread_limits():
    fwd = spread_option_price(100.0, 90.0, -1e4, 1.0, 0.05, 0.2, 0.3, 0.4, q1=0.01)
    assert fwd == pytest.approx(100 * np.exp(-0.01) - 90 + 1e4 * np.exp(-0.05), rel=1e-12)
    assert spread_option_price(100.0, 90.0, 1e4, 1.0, 0.05, 0.2, 0.3, 0.4) == pytest.approx(
        0.0, abs=1e-12
    )
