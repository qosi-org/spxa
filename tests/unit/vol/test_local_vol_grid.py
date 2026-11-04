"""Tabulated local volatility."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.curves import EquityForward
from spxa.pde import DupirePDE
from spxa.vol import SSVI, HestonLikePhi, LocalVolSurface, VolSurface
from spxa.vol.local_vol import LocalVolGrid

FWD = EquityForward(100.0, 0.03, dividend_yield=0.01)
# a Heston-like phi keeps the local vol bounded as t -> 0
SS = SSVI(-0.6, HestonLikePhi(2.0), lambda t: 0.04 * t + 0.003 * t * t)
SURF = VolSurface(SS, FWD)
LV = LocalVolSurface(SURF)
YS = np.linspace(-2.0, 1.2, 161)
TS = np.linspace(0.01, 1.0, 100)


@pytest.fixture(scope="module")
def grid() -> LocalVolGrid:
    return LocalVolGrid.from_surface(LV, YS, TS)


def test_nodes_bilinear_and_flat_extrapolation() -> None:
    ys, ts = np.array([-1.0, 0.0, 1.0]), np.array([0.5, 1.0])
    var = np.array([[0.09, 0.04, 0.01], [0.16, 0.09, 0.04]])
    g = LocalVolGrid(ys, ts, var, FWD.forward)
    assert g.local_variance_k(ys, 0.5) == pytest.approx(var[0], abs=1e-15)
    assert g.local_variance_k(0.5, 0.75) == pytest.approx(
        0.25 * (0.04 + 0.01 + 0.09 + 0.04), abs=1e-15
    )
    assert g.local_variance_k(-3.0, 0.1) == pytest.approx(0.09, abs=1e-15)
    assert g.local_variance_k(3.0, 2.0) == pytest.approx(0.04, abs=1e-15)
    F = float(FWD.forward(0.5))
    assert g(F, 0.5) == pytest.approx(0.2, abs=1e-15)
    assert g(0.0, 0.5) == pytest.approx(0.3, abs=1e-15)


def test_missing_values_floor_and_cap() -> None:
    ys, ts = np.array([-1.0, 0.0, 1.0, 2.0]), np.array([1.0])
    g = LocalVolGrid(ys, ts, [[np.nan, 0.04, np.nan, 25.0]], FWD.forward, floor=0.05, cap=2.0)
    assert g.variance[0].tolist() == pytest.approx([0.04, 0.04, 0.04, 4.0])
    g = LocalVolGrid(ys, ts, [[1e-6, 0.04, 0.04, 0.04]], FWD.forward, floor=0.05)
    assert g.variance[0, 0] == pytest.approx(0.0025)
    with pytest.raises(ValueError):
        LocalVolGrid(ys, ts, [[np.nan] * 4], FWD.forward)
    with pytest.raises(ValueError):
        LocalVolGrid(ys, ts, [[0.04] * 3], FWD.forward)


def test_close_to_the_exact_local_vol(grid: LocalVolGrid) -> None:
    S = np.array([60.0, 80.0, 100.0, 120.0, 150.0])
    for t in (0.037, 0.3, 0.77):
        assert grid(S, t) == pytest.approx(LV(S, t), abs=1e-3)


def test_pde_with_the_grid_reprices_the_surface(grid: LocalVolGrid) -> None:
    res = DupirePDE(0.03, grid, 0.01).solve(100.0, 1.0, maturities=[0.5], n_space=801, n_time=400)
    K = np.array([80.0, 100.0, 120.0])
    for T in (0.5, 1.0):
        assert res.call(K, T) == pytest.approx(SURF.price(K, T), abs=5e-4)
