"""Tests for the raw SVI slice."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.vol import SVIRaw

P = SVIRaw(a=0.02, b=0.1, rho=-0.4, m=0.05, sigma=0.2)
K = np.linspace(-2.0, 2.0, 41)


def test_formula() -> None:
    x = K - P.m
    ref = P.a + P.b * (P.rho * x + np.sqrt(x * x + P.sigma**2))
    assert np.allclose(P.total_variance(K), ref, rtol=1e-15, atol=0)
    assert P(0.05) == pytest.approx(P.a + P.b * P.sigma, abs=1e-16)
    assert isinstance(P(0.0), float)


def test_derivatives_against_finite_differences() -> None:
    h = 1e-5
    fd1 = (P(K + h) - P(K - h)) / (2 * h)
    fd2 = (P(K + h) - 2 * P(K) + P(K - h)) / h**2
    assert np.max(np.abs(P.dw(K) - fd1)) < 1e-10
    assert np.max(np.abs(P.d2w(K) - fd2)) < 1e-5
    assert np.all(P.d2w(K) > 0)  # SVI is convex in k


def test_minimum_and_wings() -> None:
    assert P.dw(P.argmin) == pytest.approx(0.0, abs=1e-15)
    assert P(P.argmin) == pytest.approx(P.min_variance, abs=1e-15)
    grid = np.linspace(-3, 3, 20001)
    assert P(grid).min() >= P.min_variance - 1e-15
    big = 1e6
    assert P(big) / big == pytest.approx(P.right_slope, rel=1e-6)
    assert P(-big) / big == pytest.approx(P.left_slope, rel=1e-6)
    assert P.left_slope == pytest.approx(0.14, abs=1e-16)
    assert P.right_slope == pytest.approx(0.06, abs=1e-16)


def test_lee_bound() -> None:
    assert P.satisfies_lee_bound()
    assert not SVIRaw(0.0, 1.5, 0.5, 0.0, 0.1).satisfies_lee_bound()
    assert SVIRaw(0.0, 1.0, 1.0 - 1e-12, 0.0, 0.1).satisfies_lee_bound()


def test_implied_vol() -> None:
    T = 0.5
    assert np.allclose(P.implied_vol(K, T) ** 2 * T, P(K), rtol=1e-14)
    with pytest.raises(ValueError):
        P.implied_vol(0.0, 0.0)


def test_array_round_trip() -> None:
    q = SVIRaw.from_array(P.to_array())
    assert q == P


@pytest.mark.parametrize(
    "params,msg",
    [
        ((0.0, -0.1, 0.0, 0.0, 0.1), "b must"),
        ((0.0, 0.1, 1.0, 0.0, 0.1), "rho"),
        ((0.0, 0.1, 0.0, 0.0, 0.0), "sigma"),
        ((-0.05, 0.1, 0.0, 0.0, 0.1), "negative minimum"),
        ((math.nan, 0.1, 0.0, 0.0, 0.1), "finite"),
    ],
)
def test_validation(params: tuple, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        SVIRaw(*params)


def test_zero_minimum_allowed() -> None:
    s = SVIRaw(-0.1 * 0.2 * math.sqrt(1 - 0.3**2), 0.1, 0.3, 0.0, 0.2)
    assert s.min_variance == pytest.approx(0.0, abs=1e-16)
