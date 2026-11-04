"""Andreasen–Huge implicit steps: arbitrage-free by construction."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.black import black_price
from spxa.market.curves import EquityForward
from spxa.vol.andreasen_huge import (
    AndreasenHugeSurface,
    andreasen_huge_grid,
    piecewise_constant_vol,
)
from spxa.vol.surface import VolSurface

RNG = np.random.default_rng(20110301)
EXPIRIES = [0.1, 0.25, 0.5, 1.0, 2.0]
NODES = [np.linspace(-0.6, 0.4, 7) * np.sqrt(t) for t in EXPIRIES]
LEVELS = [RNG.uniform(0.1, 0.5, 7) for _ in EXPIRIES]
AH = AndreasenHugeSurface(EXPIRIES, NODES, LEVELS)


def test_grid_and_piecewise_levels() -> None:
    x = andreasen_huge_grid(3.0, 101, 0.3)
    y = np.log(x)
    assert y[0] == pytest.approx(-3.0) and y[-1] == pytest.approx(3.0)
    assert np.all(np.diff(y) > 0)
    dy = np.diff(y)
    assert dy[50] < dy[0] / 5  # dense at the money
    lev = piecewise_constant_vol(
        [-1.0, 0.0, 2.0], [0.3, 0.2, 0.1], [-5.0, -0.51, -0.49, 0.99, 1.01, 9.0]
    )
    assert lev.tolist() == [0.3, 0.3, 0.2, 0.2, 0.1, 0.1]
    with pytest.raises(ValueError):
        piecewise_constant_vol([0.0, 0.0], [0.1, 0.2], 0.0)


def _check_slice(x: np.ndarray, c: np.ndarray) -> None:
    assert np.all(c >= np.maximum(1.0 - x, 0.0) - 1e-14)
    assert np.all(c <= 1.0 + 1e-14)
    assert np.all(np.diff(c) <= 1e-14)  # decreasing in strike
    slopes = np.diff(c) / np.diff(x)
    assert np.all(np.diff(slopes) >= -1e-11)  # convex


def test_no_butterfly_or_calendar_arbitrage_on_the_grid() -> None:
    x = AH.x
    times = [0.0, 0.05, 0.1, 0.17, 0.25, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    prev = None
    for t in times:
        c = AH.grid_calls(t)
        _check_slice(x, c)
        if prev is not None:
            assert np.all(c >= prev - 1e-14)
        prev = c
    assert np.array_equal(AH.grid_calls(1.0), AH.calls[4])


def test_many_flat_steps_converge_to_black() -> None:
    k = np.linspace(-0.4, 0.3, 8)
    target = black_price(1.0, np.exp(k), 1.0, 0.25, 1.0, "call")
    errs = []
    for n in (50, 100, 200):
        t = np.linspace(1.0 / n, 1.0, n)
        ah = AndreasenHugeSurface(t, [[0.0]] * n, [[0.25]] * n, grid=andreasen_huge_grid(4.0, 801))
        errs.append(np.max(np.abs(ah.normalised_call(k, 1.0) - target)))
    # backward Euler in time: first order
    assert errs[1] / errs[0] == pytest.approx(0.5, abs=0.05)
    assert errs[2] / errs[1] == pytest.approx(0.5, abs=0.05)
    assert errs[2] < 2e-4


def test_total_variance_round_trip() -> None:
    fwd = EquityForward(100.0, 0.03, dividend_yield=0.01)
    surf = VolSurface(AH, fwd)
    K = np.array([80.0, 95.0, 100.0, 110.0, 125.0])
    for T in (0.25, 0.7, 2.0):
        F, D = float(fwd.forward(T)), float(fwd.discount(T))
        c = AH.normalised_call(np.log(K / F), T)
        assert surf.price(K, T) == pytest.approx(D * F * c, rel=1e-10)
    assert AH.total_variance(0.0, 0.0) == 0.0


def test_validation() -> None:
    with pytest.raises(ValueError):
        AndreasenHugeSurface([0.5, 0.25], [[0.0], [0.0]], [[0.2], [0.2]])
    with pytest.raises(ValueError):
        AndreasenHugeSurface([0.5], [[0.0]], [[-0.2]])
    with pytest.raises(ValueError):
        AndreasenHugeSurface([0.5], [[0.0], [0.0]], [[0.2]])
    with pytest.raises(ValueError):
        AH.grid_calls(-1.0)
