"""Calibrated SLV models and the mixing fraction."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.pde import BlackScholesPDE
from spxa.vol.heston import Heston
from spxa.vol.slv_model import calibrate_slv
from tests.unit.vol.test_slv_fp import _ssvi_local_vol

H = Heston(0.04, 1.5, 0.04, 0.5, -0.7)
R, Q = 0.03, 0.01


def test_mixing_fraction_moves_barriers_not_vanillas() -> None:
    ss, lv = _ssvi_local_vol()
    k = np.array([-0.3, -0.15, 0.0, 0.1, 0.2])
    up_and_out = []
    for eta in (0.0, 0.5, 1.0):
        slv = calibrate_slv(lv, H, eta, 1.0, n_time=50, record=(1.0,))
        assert slv.model.xi == pytest.approx(max(eta, 1e-12) * H.xi)
        assert np.abs(slv.smile(k, 1.0) - ss.implied_vol(k, 1.0)).max() < 6e-4
        pde = slv.pde(R, Q, n_x=81, n_v=31, n_time=40)
        up_and_out.append(pde.barrier(100.0, 1.0, "call", 100.0, upper=130.0))
        if eta == 0.0:
            # eta = 0 is local volatility: compare with the one-factor LV PDE
            lv_do = float(
                BlackScholesPDE(R, lv, Q).barrier(100.0, 1.0, 100.0, 85.0, "down-and-out", "call")
            )
            assert pde.barrier(100.0, 1.0, "call", 100.0, lower=85.0) == pytest.approx(
                lv_do, rel=2e-3
            )
    assert up_and_out[0] < up_and_out[1] < up_and_out[2]
    assert up_and_out[2] / up_and_out[0] - 1 > 0.05
    p = slv.paths(1.0, 10, 100, seed=0)
    assert p.y.shape == (1, 100)


def test_validation() -> None:
    _, lv = _ssvi_local_vol()
    with pytest.raises(ValueError):
        calibrate_slv(lv, H, 1.5, 1.0)
