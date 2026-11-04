"""Local volatility reproduces its implied surface through the spxa.pde solvers."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.curves import EquityForward
from spxa.pde import vanilla
from spxa.vol import SSVI, LocalVolSurface, PowerLawPhi, VolSurface

SS = SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t)
FWD = EquityForward(100.0, 0.03, dividend_yield=0.01)
SURF = VolSurface(SS, FWD)
LV = LocalVolSurface(SURF)
K = np.array([70.0, 85.0, 100.0, 115.0, 130.0])


@pytest.fixture(scope="module")
def dupire() -> tuple:
    pde = LV.dupire_pde(0.03, 0.01)
    coarse = pde.solve(100.0, 1.0, maturities=[0.25, 0.5], n_space=601, n_time=400)
    fine = pde.solve(100.0, 1.0, maturities=[0.25, 0.5], n_space=1201, n_time=800)
    return coarse, fine


def test_forward_pde_reprices_the_surface(dupire: tuple) -> None:
    coarse, fine = dupire
    for T, tol in ((0.25, 1e-4), (0.5, 3e-5), (1.0, 1e-5)):
        err_c = np.abs(coarse.implied_vol(K, T) - SURF.implied_vol(K, T))
        err_f = np.abs(fine.implied_vol(K, T) - SURF.implied_vol(K, T))
        assert np.max(err_f) < tol
        # second-order convergence: halving both steps divides the error by about four
        assert np.all(err_f < 0.4 * err_c)


def test_backward_pde_converges_to_the_surface(dupire: tuple) -> None:
    _, fine = dupire
    pde = LV.black_scholes_pde(0.03, 0.01)
    for k in (85.0, 100.0):
        target = float(SURF.price(k, 1.0))
        assert fine.call(k, 1.0) == pytest.approx(target, rel=2e-5)
        errs = [
            abs(
                pde.solve(vanilla(k, "call"), 1.0, center=100.0, n_space=n, n_time=m).value(100.0)
                - target
            )
            for n, m in ((401, 200), (801, 400))
        ]
        # the payoff kink and the steep short-dated local vol slow the backward solve down
        assert errs[1] < 0.6 * errs[0] and errs[1] < 5e-4 * target


def test_clipping_for_wide_grids() -> None:
    S = np.array([0.0, 1e-6, 1.0, 1e4])
    sig = LV(S, 0.5)
    assert np.all(np.isfinite(sig)) and np.all(sig > 0)
    lo, hi = LV.y_range
    F = float(FWD.forward(0.5))
    assert sig[0] == pytest.approx(LV.local_vol_k(lo, 0.5))
    assert sig[-1] == pytest.approx(LV.local_vol_k(hi, 0.5))
    assert LV(F, 0.0) == pytest.approx(
        LV.local_vol_k(np.log(F / float(FWD.forward(LV.t_min))), LV.t_min)
    )
