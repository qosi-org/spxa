"""Gatheral–Jacquier price interpolation of slices in time."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.vol import (
    SSVI,
    PowerLawPhi,
    PriceInterpolatedSurface,
    normalised_call_from_total_variance,
)

SURF = SSVI(
    rho=-0.65, phi=PowerLawPhi(eta=0.9, gamma=0.35), theta=lambda t: 0.035 * t + 0.002 * t * t
)
T = [0.25, 0.5, 1.0, 2.0]
SLICES = [SURF.slice(t) for t in T]
PIS = PriceInterpolatedSurface(T, SLICES)
K = np.linspace(-1.5, 1.0, 251)


def test_nodes_are_reproduced() -> None:
    for t, s in zip(T, SLICES):
        assert np.array_equal(PIS.total_variance(K, t), s.total_variance(K))
        c = normalised_call_from_total_variance(K, s.total_variance(K))
        assert np.allclose(PIS.call_price(K, t), c, rtol=0, atol=1e-15)


def test_weights_and_theta() -> None:
    assert PIS.theta(0.0) == 0.0
    th = PIS.thetas
    assert PIS.theta(0.75) == pytest.approx(0.5 * (th[1] + th[2]), abs=1e-16)
    i, a = PIS.bucket(0.75)
    assert i == 2
    ref = (np.sqrt(th[2]) - np.sqrt(PIS.theta(0.75))) / (np.sqrt(th[2]) - np.sqrt(th[1]))
    assert a == pytest.approx(ref, abs=1e-15)
    with pytest.raises(ValueError, match="extrapolation"):
        PIS.theta(2.5)


def test_atm_total_variance_tracks_theta() -> None:
    # the ATM price (s - s^3/24 + ...)/sqrt(2 pi) in s = sqrt(theta) is interpolated
    # linearly in s; the cubic's interpolation error gives the shortfall in closed form
    nodes = np.sqrt(np.concatenate([[0.0], PIS.thetas]))
    for t in (0.1, 0.3, 0.7, 1.4, 1.9):
        i, _ = PIS.bucket(t)
        s, lo, hi = np.sqrt(PIS.theta(t)), nodes[i], nodes[i + 1]
        lead = s * (s - lo) * (hi - s) * (lo + s + hi) / 12
        short = PIS.theta(t) - PIS.total_variance(0.0, t)
        assert short > 0
        assert short == pytest.approx(lead, rel=hi**2 / 20)


def test_no_butterfly_arbitrage_between_nodes() -> None:
    for t in (0.1, 0.4, 0.8, 1.6):
        c = PIS.call_price(K, t)
        Kx = np.exp(K)
        slopes = np.diff(c) / np.diff(Kx)
        assert np.all(slopes <= 1e-15) and np.all(slopes >= -1 - 1e-12)  # -1 <= dC/dK <= 0
        assert np.all(np.diff(slopes) >= -1e-13)  # convex in strike
        # mixture density integrates to the continuous mass
        i, a = PIS.bucket(t)
        mass = quad(lambda x: PIS.density(x, t), -12, 12, points=[0.0], limit=400)[0]
        assert mass == pytest.approx(1.0 - (a if i == 0 else 0.0), abs=1e-9)


def test_no_calendar_arbitrage() -> None:
    ts = np.linspace(0.05, 2.0, 40)
    prices = np.array([PIS.call_price(K, t) for t in ts])
    assert np.all(np.diff(prices, axis=0) >= -1e-15)
    ws = np.array([PIS.total_variance(K, t) for t in ts[1:]])
    assert np.all(np.diff(ws, axis=0) >= -1e-12)


def test_implied_vols_from_otm_prices() -> None:
    # deep wings are inverted from OTM options, so they stay accurate
    t = 0.75
    w = PIS.total_variance(np.array([-1.5, 1.0]), t)
    lo, hi = (
        SLICES[1].total_variance(np.array([-1.5, 1.0])),
        SLICES[2].total_variance(np.array([-1.5, 1.0])),
    )
    assert np.all(w > lo) and np.all(w < hi)
    assert PIS.implied_vol(0.0, t) == pytest.approx(np.sqrt(PIS.total_variance(0.0, t) / t))
    with pytest.raises(ValueError):
        PIS.implied_vol(0.0, 0.0)


def test_validation() -> None:
    with pytest.raises(ValueError, match="increasing"):
        PriceInterpolatedSurface([0.5, 0.25], SLICES[:2])
    with pytest.raises(ValueError, match="one slice"):
        PriceInterpolatedSurface([0.5], SLICES[:2])
    with pytest.raises(ValueError, match="ATM"):
        PriceInterpolatedSurface([0.5, 1.0], [SLICES[2], SLICES[1]])
