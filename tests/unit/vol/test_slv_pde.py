"""SLV backward PDE."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.pde.reference import barrier_price
from spxa.vol.heston import Heston
from spxa.vol.slv import slv_grid
from spxa.vol.slv_fp import calibrate_leverage
from spxa.vol.slv_mc import slv_paths
from spxa.vol.slv_pde import SLVPDE
from tests.unit.vol.test_slv_fp import _ssvi_local_vol

M = Heston(0.04, 1.5, 0.04, 0.5, -0.7)
R, Q = 0.03, 0.01


def test_heston_limit_with_carry() -> None:
    pde = SLVPDE(M, r=R, q=Q)
    F, df = 100 * math.exp(R - Q), math.exp(-R)
    for K, opt in ((100.0, "call"), (100.0, "put"), (80.0, "put"), (120.0, "call")):
        ref = M.price(K, 1.0, F=F, df=df, option=opt)
        assert pde.european(K, 1.0, opt, spot=100.0) == pytest.approx(ref, abs=4e-3)


def test_black_scholes_down_and_out() -> None:
    exact = barrier_price(100.0, 100.0, 85.0, 1.0, R, 0.2, Q)
    errs = [
        abs(
            SLVPDE(M.scaled(0.0), r=R, q=Q, n_x=n, n_v=41).barrier(
                100.0, 1.0, "call", 100.0, lower=85.0
            )
            - exact
        )
        for n in (161, 321)
    ]
    assert errs[1] < 4e-4 and errs[1] < 0.25 * errs[0]


def test_calibrated_barrier_matches_simulation() -> None:
    _, lv = _ssvi_local_vol()
    m = M.scaled(0.5)
    cal = calibrate_leverage(lv, m, 1.0, slv_grid(m, 1.0, 121, 61, vol=0.2), 100)
    pde = SLVPDE(m, cal.leverage, n_x=101, n_v=41, n_time=50).barrier(1.0, 1.0, "call", lower=0.85)
    pv, se = slv_paths(m, cal.leverage, 1.0, 100, 200_000, seed=4, lower=np.log(0.85)).price(
        lambda S: np.maximum(S - 1.0, 0.0), 1.0, knock_out=True
    )
    assert abs(pde - pv) < 3 * se


def test_validation() -> None:
    with pytest.raises(ValueError):
        SLVPDE(M).barrier(1.0, 1.0)
    with pytest.raises(ValueError):
        SLVPDE(M).barrier(1.0, 1.0, lower=1.1)
    up = SLVPDE(M, n_x=101, n_v=41, n_time=50)
    assert 0 < up.price(lambda S: np.maximum(S - 1.0, 0.0), 1.0, upper=1.3) < up.european(1.0, 1.0)
