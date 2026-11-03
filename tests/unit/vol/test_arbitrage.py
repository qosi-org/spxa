"""Butterfly and calendar diagnostics for total-variance slices."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.vol import (
    SVIRaw,
    check_butterfly,
    check_calendar,
    durrleman_g,
    normalised_call_from_total_variance,
    risk_neutral_density,
)

GOOD = SVIRaw(0.02, 0.1, -0.4, 0.05, 0.2)
# Axel Vogt's slice, Gatheral & Jacquier (2014) Example 3.1: butterfly arbitrage
VOGT = SVIRaw(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


def test_flat_variance_gives_g_equal_one() -> None:
    # constant w: g = 1 exactly (density is the lognormal one)
    k = np.linspace(-2, 2, 9)
    assert np.allclose(durrleman_g(k, 0.04, 0.0, 0.0), 1.0, rtol=0, atol=0)
    with pytest.raises(ValueError):
        durrleman_g(0.0, 0.0, 0.0, 0.0)


def test_density_matches_second_strike_derivative() -> None:
    # with F = 1 and K = e^k: p(k) = K d^2c/dK^2 * K
    k = np.linspace(-1.0, 1.0, 21)
    K = np.exp(k)
    h = 1e-4

    def c(KK: np.ndarray) -> np.ndarray:
        return normalised_call_from_total_variance(np.log(KK), GOOD(np.log(KK)))

    d2 = (c(K * (1 + h)) - 2 * c(K) + c(K * (1 - h))) / (K * h) ** 2
    assert np.max(np.abs(risk_neutral_density(GOOD, k) - d2 * K)) < 1e-6


def test_density_is_a_martingale_density() -> None:
    mass = quad(lambda x: risk_neutral_density(GOOD, x), -12, 12, limit=200)[0]
    mean = quad(lambda x: np.exp(x) * risk_neutral_density(GOOD, x), -12, 12, limit=200)[0]
    assert mass == pytest.approx(1.0, abs=1e-8)
    assert mean == pytest.approx(1.0, abs=1e-8)


def test_good_slice_passes() -> None:
    res = check_butterfly(GOOD)
    assert res and res.arbitrage_free and res.min_g > 0 and res.negative_intervals == []


def test_vogt_slice_has_butterfly_arbitrage() -> None:
    res = check_butterfly(VOGT, np.linspace(-1.5, 1.5, 3001))
    assert not res
    assert res.min_g < 0
    assert len(res.negative_intervals) >= 1
    lo, hi = res.negative_intervals[0]
    assert lo <= res.argmin <= hi
    assert risk_neutral_density(VOGT, res.argmin) < 0


def test_callable_slices_use_finite_differences() -> None:
    k = np.linspace(-1, 1, 41)
    f = GOOD.total_variance
    exact = check_butterfly(GOOD, k)
    numeric = check_butterfly(lambda x: f(x), k)
    assert numeric.min_g == pytest.approx(exact.min_g, abs=1e-6)
    assert risk_neutral_density(lambda x: f(x), 0.1) == pytest.approx(
        risk_neutral_density(GOOD, 0.1), rel=1e-6
    )


def test_negative_variance_is_flagged() -> None:
    res = check_butterfly(lambda x: 0.01 - 0.02 * x * x, np.linspace(-1, 1, 11))
    assert not res and res.min_g == -np.inf


def test_calendar() -> None:
    s1 = SVIRaw(0.01, 0.1, -0.4, 0.0, 0.2)
    s2 = SVIRaw(0.02, 0.1, -0.4, 0.0, 0.2)
    assert check_calendar([s1, s2]).arbitrage_free
    res = check_calendar([s2, s1, s2])
    assert not res and res.between == (0, 1)
    assert res.worst == pytest.approx(0.01, abs=1e-15)
    assert check_calendar([s1]).arbitrage_free
    # crossing slices: steeper short-dated wing overtakes the longer expiry
    steep = SVIRaw(0.0, 0.5, 0.0, 0.0, 0.1)
    res = check_calendar([steep, s2])
    assert not res and abs(res.at_k) == pytest.approx(3.0, abs=1e-12)
    assert check_calendar([s2, s2])  # equal slices are allowed
