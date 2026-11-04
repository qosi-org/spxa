"""American and barrier options under local volatility."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.pde import BlackScholesPDE
from spxa.pde.reference import barrier_price
from spxa.vol import LocalVolSurface, VolSurface
from spxa.vol.local_vol import LocalVolGrid
from tests.unit.vol.test_local_vol_grid import FWD, LV, SURF, TS, YS


def test_flat_surface_matches_black_scholes() -> None:
    flat = LocalVolSurface(VolSurface.flat(0.25, FWD))
    am = flat.american(100.0, 1.0, 100.0, 0.03, 0.01, "put")
    ref = BlackScholesPDE(0.03, 0.25, 0.01).american(100.0, 1.0, 100.0, "put")
    assert am.price == pytest.approx(ref.price, rel=1e-8)
    do = flat.barrier(100.0, 1.0, 100.0, 85.0, 0.03, 0.01)
    exact = barrier_price(100.0, 100.0, 85.0, 1.0, 0.03, 0.25, 0.01, "call", "down-and-out")
    assert do.price == pytest.approx(exact, rel=2e-3)


def test_american_options() -> None:
    # without dividends an American call is never exercised early
    am = LV.american(100.0, 1.0, 100.0, 0.03, 0.0, "call")
    eu = LV.black_scholes_pde(0.03, 0.0).european(100.0, 1.0, 100.0, "call")
    assert am.price == pytest.approx(eu.price, rel=1e-6)
    put = LV.american(100.0, 1.0, 100.0, 0.03, 0.01, "put")
    premium = put.price - float(SURF.price(100.0, 1.0, "put"))
    assert 0.05 < premium < 0.5


def test_barrier_parity_and_monte_carlo() -> None:
    do = LV.barrier(100.0, 1.0, 100.0, 85.0, 0.03, 0.01, "down-and-out")
    di = LV.barrier(100.0, 1.0, 100.0, 85.0, 0.03, 0.01, "down-and-in")
    assert do.price + di.price == pytest.approx(float(SURF.price(100.0, 1.0)), rel=1e-4)
    # Monte Carlo on the tabulated local vol, with a Brownian-bridge crossing probability per step
    grid = LocalVolGrid.from_surface(LV, YS, TS)
    rng = np.random.default_rng(5)
    n, m, H = 100, 100_000, 85.0
    dt = 1.0 / n
    x = np.full(2 * m, np.log(100.0))
    alive = np.ones(2 * m, bool)
    for i in range(n):
        z = rng.standard_normal(m)
        z = np.concatenate([z, -z])
        s = grid(np.exp(x), i * dt)
        xn = x + (0.03 - 0.01 - 0.5 * s * s) * dt + s * np.sqrt(dt) * z
        a, b = x - np.log(H), xn - np.log(H)
        cross = np.where((a > 0) & (b > 0), np.exp(-2 * a * b / (s * s * dt)), 1.0)
        alive &= rng.random(2 * m) > cross
        x = xn
    v = np.exp(-0.03) * np.maximum(np.exp(x) - 100.0, 0.0) * alive
    assert abs(v.mean() - do.price) < 4 * v.std() / np.sqrt(v.size)
