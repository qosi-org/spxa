"""Monte Carlo under local volatility reprices the surface it came from."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.black import black_scholes_price
from spxa.vol.local_vol import LocalVolGrid
from spxa.vol.local_vol_mc import local_vol_paths
from tests.unit.vol.test_local_vol_grid import LV, SURF, TS, YS


@pytest.fixture(scope="module")
def paths():  # type: ignore[no-untyped-def]
    grid = LocalVolGrid.from_surface(LV, YS, TS)
    return local_vol_paths(grid, 100.0, 0.03, 0.01, 1.0, 100, 200_000, seed=3, record=[0.5, 1.0])


def test_constant_vol_is_exact() -> None:
    p = local_vol_paths(lambda s, t: 0.25 + 0.0 * s, 100.0, 0.03, 0.01, 2.0, 8, 100_000, seed=11)
    x = np.log(p.at(2.0))
    mean, var = np.log(100.0) + (0.03 - 0.01 - 0.5 * 0.0625) * 2.0, 0.0625 * 2.0
    assert x.mean() == pytest.approx(mean, abs=4 * np.sqrt(var / x.size))
    assert x.var() == pytest.approx(var, rel=4 * np.sqrt(2.0 / x.size))
    for K in (80.0, 100.0, 130.0):
        m, se = p.price(lambda s: np.maximum(s - K, 0.0), 2.0)
        assert abs(m - black_scholes_price(100.0, K, 2.0, 0.03, 0.25, 0.01, "call")) < 4 * se


def test_forward_is_a_martingale(paths) -> None:  # type: ignore[no-untyped-def]
    for t in (0.5, 1.0):
        v = np.exp(-(0.03 - 0.01) * t) * paths.at(t)
        assert abs(v.mean() - 100.0) < 4 * v.std() / np.sqrt(v.size)


def test_vanillas_within_standard_errors(paths) -> None:  # type: ignore[no-untyped-def]
    for T in (0.5, 1.0):
        for K in (80.0, 100.0, 120.0):
            for opt in ("call", "put"):
                eta = 1.0 if opt == "call" else -1.0
                m, se = paths.price(lambda s: np.maximum(eta * (s - K), 0.0), T)
                assert abs(m - float(SURF.price(K, T, opt))) < 4 * se


def test_record_times() -> None:
    p = local_vol_paths(lambda s, t: 0.2, 100.0, 0.0, 0.0, 1.0, 4, 10, record=[0.25, 0.75])
    assert p.times.tolist() == [0.0, 0.25, 0.75]
    assert p.paths.shape == (10, 3)
    with pytest.raises(ValueError):
        local_vol_paths(lambda s, t: 0.2, 100.0, 0.0, 0.0, 1.0, 4, 10, record=[0.3])
    with pytest.raises(ValueError):
        p.at(0.5)


def test_forward_start_options(paths) -> None:  # type: ignore[no-untyped-def]
    k = np.array([0.0, 0.9, 1.0, 1.1])
    c, se = paths.forward_start(0.5, 1.0, k)
    p, _ = paths.forward_start(0.5, 1.0, k, "put")
    ratio = paths.at(1.0) / paths.at(0.5)
    # pathwise parity, and E[S_2/S_1] = exp((r - q)(T2 - T1))
    assert c - p == pytest.approx(np.exp(-0.03) * (ratio.mean() - k), abs=1e-13)
    assert abs(c[0] - np.exp(-0.03) * np.exp(0.02 * 0.5)) < 4 * se[0]
    assert c[1] > c[2] > c[3] > 0


def test_forward_start_with_constant_vol() -> None:
    p = local_vol_paths(
        lambda s, t: 0.25 + 0.0 * s, 100.0, 0.03, 0.01, 1.5, 6, 200_000, seed=12, record=[0.5, 1.5]
    )
    k = np.array([0.9, 1.0, 1.2])
    c, se = p.forward_start(0.5, 1.5, k)
    # the return over [T1, T2] is lognormal and independent of S_T1
    exact = np.exp(-0.03 * 0.5) * black_scholes_price(1.0, k, 1.0, 0.03, 0.25, 0.01, "call")
    assert np.all(np.abs(c - exact) < 4 * se)
