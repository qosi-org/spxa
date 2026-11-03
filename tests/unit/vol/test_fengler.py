"""Fengler's arbitrage-free cubic-spline smoothing of call prices."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.black import black_price
from spxa.vol import SSVI, PowerLawPhi
from spxa.vol.fengler import fengler_smooth, spline_matrices

F, D, T = 100.0, 0.98, 0.5
SS = SSVI(-0.6, PowerLawPhi(1.0, 0.4), lambda t: 0.04 * t)
K = np.linspace(60.0, 150.0, 25)
SIG = SS.implied_vol(np.log(K / F), T)
C = black_price(F, K, T, SIG, D, "call")


def test_spline_matrices_and_interpolation() -> None:
    u = np.array([0.0, 0.3, 1.0, 1.4, 2.5])
    Q, R = spline_matrices(u)
    assert Q.shape == (5, 3) and R.shape == (3, 3)
    assert np.allclose(R, R.T) and np.all(np.linalg.eigvalsh(R) > 0)
    # a straight line is a spline with zero second derivatives: Q^T g = 0
    assert Q.T @ (2 - 3 * u) == pytest.approx(0.0, abs=1e-14)
    with pytest.raises(ValueError):
        spline_matrices(np.array([0.0, 1.0]))


def test_interpolates_clean_prices_in_the_small_lambda_limit() -> None:
    sm = fengler_smooth(K, C, F, D, T, smoothing=1e-12)
    assert sm.price(K) == pytest.approx(C, abs=2e-7 * D * F)
    assert sm.implied_vol(K[3:-3]) == pytest.approx(SIG[3:-3], abs=2e-4)
    # C^2 spline: Q^T g = R gamma, and g, g', g'' continuous at the knots
    Q, R = spline_matrices(sm.knots)
    assert Q.T @ sm.values == pytest.approx(R @ sm.gammas[1:-1], abs=1e-9)
    eps = 1e-9
    for nu in (0, 1, 2):
        left = sm.normalised(sm.knots[1:-1] - eps, nu)
        right = sm.normalised(sm.knots[1:-1] + eps, nu)
        assert left == pytest.approx(right, abs=1e-6)
    # the density integrates to the slope change
    u = np.linspace(sm.knots[0], sm.knots[-1], 20001)
    mass = np.trapezoid(sm.density(u * F), u * F)
    assert mass == pytest.approx(
        sm.normalised(sm.knots[-1], 1) - sm.normalised(sm.knots[0], 1), abs=1e-6
    )


def test_noisy_prices_become_arbitrage_free() -> None:
    rng = np.random.default_rng(11)
    noisy = C + 0.05 * rng.standard_normal(K.size)
    u = K / F
    c = noisy / (D * F)
    slopes = np.diff(c) / np.diff(u)
    assert np.any(np.diff(slopes) < 0)  # butterfly arbitrage in the input
    sm = fengler_smooth(K, noisy, F, D, T, smoothing=1e-6)
    assert np.all(sm.gammas >= 0)
    grid = np.linspace(K[0], K[-1], 2001)
    p = sm.price(grid)
    assert np.all(np.diff(p) <= 1e-12)  # calls decrease in strike
    assert np.all(np.diff(p, 2) >= -1e-12)  # and are convex
    assert np.all(p >= np.maximum(D * (F - grid), 0) - 1e-12) and np.all(p <= D * F + 1e-12)
    d1 = sm.normalised(grid / F, 1)
    assert np.all(d1 >= -1 - 1e-12) and np.all(d1 <= 1e-12)
    # still close to the clean prices: noise is mostly removed
    assert np.sqrt(np.mean((sm.price(K) - C) ** 2)) < 0.05


def test_smoothing_trades_fit_for_roughness() -> None:
    rng = np.random.default_rng(5)
    noisy = C + 0.03 * rng.standard_normal(K.size)
    fits = [fengler_smooth(K, noisy, F, D, T, smoothing=lam) for lam in (1e-8, 1e-6, 1e-4)]
    err = [np.sum((f.price(K) - noisy) ** 2) for f in fits]
    rough = [f.roughness for f in fits]
    assert err[0] < err[1] < err[2]
    assert rough[0] > rough[1] > rough[2]


def test_validation() -> None:
    sm = fengler_smooth(K, C, F, D)
    with pytest.raises(ValueError):
        sm.price(K[-1] + 1.0)
    with pytest.raises(ValueError):
        sm.implied_vol(100.0)
    with pytest.raises(ValueError):
        fengler_smooth(K[:2], C[:2], F, D)
    with pytest.raises(ValueError):
        fengler_smooth(K, C, -1.0, D)
