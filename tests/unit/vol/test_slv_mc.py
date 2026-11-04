"""SLV Monte Carlo."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.pde.reference import barrier_price, bs_price
from spxa.vol.heston import Heston
from spxa.vol.slv import LeverageSurface, slv_grid
from spxa.vol.slv_fp import calibrate_leverage
from spxa.vol.slv_mc import slv_paths
from tests.unit.vol.test_slv_fp import _ssvi_local_vol

M = Heston(0.04, 1.5, 0.04, 0.5, -0.7)


def test_black_scholes_limit_with_bridge() -> None:
    flat = M.scaled(0.0)  # v stays at 0.04: sigma = 20%
    exact = barrier_price(1.0, 1.0, 0.85, 1.0, 0.0, 0.2)
    kw = dict(seed=1, lower=np.log(0.85))
    pv, se = slv_paths(flat, LeverageSurface.constant(), 1.0, 20, 200_000, **kw).price(
        lambda S: np.maximum(S - 1.0, 0.0), 1.0, knock_out=True
    )
    assert abs(pv - exact) < 3 * se
    naive, _ = slv_paths(
        flat, LeverageSurface.constant(), 1.0, 20, 200_000, bridge=False, **kw
    ).price(lambda S: np.maximum(S - 1.0, 0.0), 1.0, knock_out=True)
    assert naive - exact > 5 * se  # discrete monitoring misses crossings between steps


def test_heston_limit_and_martingale() -> None:
    p = slv_paths(M, LeverageSurface.constant(), 1.0, 200, 100_000, record=[0.5], seed=2)
    fwd, se = p.price(lambda S: S, 0.5)
    assert abs(fwd - 1.0) < 3 * se
    for K in (0.8, 1.0, 1.15):
        pv, se = p.price(lambda S: np.maximum(S - K, 0.0), 1.0)
        assert abs(pv - M.price(K, 1.0)) < 3 * se + 1e-4


def test_calibrated_leverage_reprices_vanillas() -> None:
    ss, lv = _ssvi_local_vol()
    m = M.scaled(0.5)
    cal = calibrate_leverage(lv, m, 1.0, slv_grid(m, 1.0, 121, 61, vol=0.2), 100)
    p = slv_paths(m, cal.leverage, 1.0, 100, 100_000, seed=3)
    for k in (-0.2, 0.0, 0.15):
        K = float(np.exp(k))
        vol = float(ss.implied_vol(k, 1.0))
        target = float(bs_price(1.0, K, 1.0, 0.0, vol, option="put" if k < 0 else "call"))
        kind = (lambda S: np.maximum(K - S, 0.0)) if k < 0 else (lambda S: np.maximum(S - K, 0.0))
        pv, se = p.price(kind, 1.0)
        vega = float(bs_price(1.0, K, 1.0, 0.0, vol + 1e-4) - bs_price(1.0, K, 1.0, 0.0, vol)) * 1e4
        assert abs(pv - target) < 3 * se + 5e-4 * vega


def test_validation_and_records() -> None:
    p = slv_paths(M, LeverageSurface.constant(), 0.5, 4, 10, record=[0.3], seed=0)
    assert list(p.times) == [0.3, 0.5] and p.y.shape == (2, 10)
    with pytest.raises(ValueError):
        p.at(0.25)
    with pytest.raises(ValueError):
        slv_paths(M, LeverageSurface.constant(), 0.5, 0, 10)
