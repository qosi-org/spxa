"""Particle calibration of the SLV leverage."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.pde.reference import bs_price
from spxa.vol.heston import Heston
from spxa.vol.slv import slv_grid
from spxa.vol.slv_fp import calibrate_leverage
from spxa.vol.slv_mc import slv_paths
from spxa.vol.slv_particles import calibrate_leverage_particles, kernel_conditional_mean
from tests.unit.vol.test_slv_fp import _ssvi_local_vol

M = Heston(0.04, 1.5, 0.04, 0.5, -0.7).scaled(0.5)


def test_kernel_regression() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 20_000)
    nodes = np.array([-0.5, 0.0, 0.5, 30.0])
    np.testing.assert_allclose(
        kernel_conditional_mean(x, np.full_like(x, 0.3), nodes, 0.05, chunk=7000), 0.3
    )
    est = kernel_conditional_mean(x, 2 * x + 1, nodes, 0.02)
    np.testing.assert_allclose(est[:3], [0.0, 1.0, 2.0], atol=0.01)
    assert est[3] == est[2]  # no particles near 30: nearest informative node


def test_particles_agree_with_fokker_planck_and_reprice() -> None:
    ss, lv = _ssvi_local_vol()
    fp = calibrate_leverage(lv, M, 1.0, slv_grid(M, 1.0, 121, 61, vol=0.2), 100)
    pc = calibrate_leverage_particles(lv, M, 1.0, 100, 50_000, seed=5)
    assert pc.bandwidths[0] == 0.0 and np.all(np.diff(pc.bandwidths[1:]) > 0)
    ys = np.linspace(-0.3, 0.2, 6)
    for t in (0.25, 0.5, 0.99):
        np.testing.assert_allclose(pc.leverage(ys, t), fp.leverage(ys, t), rtol=0.025)
    p = slv_paths(M, pc.leverage, 1.0, 100, 100_000, seed=9)
    for k in (-0.2, 0.0, 0.15):
        K, vol = float(np.exp(k)), float(ss.implied_vol(k, 1.0))
        opt = "put" if k < 0 else "call"
        target = float(bs_price(1.0, K, 1.0, 0.0, vol, option=opt))
        vega = (float(bs_price(1.0, K, 1.0, 0.0, vol + 1e-4, option=opt)) - target) * 1e4
        pv, se = p.price(
            (lambda S: np.maximum(K - S, 0.0)) if k < 0 else (lambda S: np.maximum(S - K, 0.0)), 1.0
        )
        assert abs(pv - target) < 3 * se + 5e-4 * vega
    assert pc.leverage.values.shape == (100, 81)
    with pytest.raises(ValueError):
        calibrate_leverage_particles(lv, M, 1.0, 2, 1, nodes=np.array([0.0, 0.0]))
