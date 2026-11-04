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


# ------------------------------------------------------------- multi-expiry
def test_eval_matrix_matches_spline() -> None:
    from spxa.vol.fengler import spline_eval_matrix

    sm = fengler_smooth(K, C, F, D, T)
    pts = np.linspace(sm.knots[0], sm.knots[-1], 97)
    z = np.concatenate([sm.values, sm.gammas[1:-1]])
    assert spline_eval_matrix(sm.knots, pts) @ z == pytest.approx(sm.normalised(pts), abs=1e-14)
    assert spline_eval_matrix(sm.knots, pts, 1) @ z == pytest.approx(
        sm.normalised(pts, 1), abs=1e-13
    )
    with pytest.raises(ValueError):
        spline_eval_matrix(sm.knots, pts, 2)


def _surface_inputs(noise: float, seed: int) -> tuple:
    from spxa.market.curves import EquityForward

    fwd = EquityForward(100.0, 0.03, dividend_yield=0.01)
    Ts = [0.25, 0.5, 1.0]
    rng = np.random.default_rng(seed)
    strikes, calls, Fs, Ds = [], [], [], []
    for t in Ts:
        Ft, Dt = fwd.forward(t), fwd.discount(t)
        Kt = Ft * np.linspace(0.7, 1.4, 21)
        sig = SS.implied_vol(np.log(Kt / Ft), t)
        c = black_price(Ft, Kt, t, sig, Dt, "call") + noise * rng.standard_normal(Kt.size)
        strikes.append(Kt), calls.append(c), Fs.append(Ft), Ds.append(Dt)
    return Ts, strikes, calls, Fs, Ds


def test_surface_removes_calendar_arbitrage() -> None:
    from spxa.vol.fengler import fengler_smooth as one
    from spxa.vol.fengler import fengler_surface

    Ts, strikes, calls, Fs, Ds = _surface_inputs(0.02, 2)
    # stale upside quotes at 6M: priced with one year of variance above 1.15 F
    k1 = strikes[1]
    up = k1 > 1.15 * Fs[1]
    sig = SS.implied_vol(np.log(k1[up] / Fs[1]), 1.05)
    calls[1] = calls[1].copy()
    calls[1][up] = black_price(Fs[1], k1[up], 1.05, sig, Ds[1], "call")
    # independent fits cross in time there; the joint fit does not
    indep = [one(k, c, f, d, t) for t, k, c, f, d in zip(Ts, strikes, calls, Fs, Ds)]
    u = np.linspace(0.7, 1.4, 401)
    cross = max(float(np.max(a.normalised(u) - b.normalised(u))) for a, b in zip(indep, indep[1:]))
    assert cross > 1e-4
    surf = fengler_surface(Ts, strikes, calls, Fs, Ds)
    # Bernstein rows make the condition hold between sample points too
    assert surf.calendar_violation(20001) < 1e-12
    for s in surf.slices:
        assert np.all(s.gammas >= 0)
    assert surf.slice(0.5).expiry == 0.5
    with pytest.raises(ValueError):
        surf.slice(0.7)


def test_surface_leaves_clean_data_alone() -> None:
    from spxa.vol.fengler import fengler_surface

    Ts, strikes, calls, Fs, Ds = _surface_inputs(0.0, 0)
    surf = fengler_surface(Ts, strikes, calls, Fs, Ds, smoothing=1e-12)
    for s, k, c in zip(surf.slices, strikes, calls):
        assert s.price(k) == pytest.approx(c, abs=5e-5)
    with pytest.raises(ValueError):
        fengler_surface([1.0, 0.5], strikes[:2], calls[:2], Fs[:2], Ds[:2])
