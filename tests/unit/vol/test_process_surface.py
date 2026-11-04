"""Implied-volatility surfaces of zoo processes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.market.implied import implied_black_volatility
from spxa.pde.reference import merton_price
from spxa.vol import ProcessSurface, process_vol_surface
from spxa.zoo.levy.brownian import BrownianMotion
from spxa.zoo.levy.merton import MertonJumpDiffusion

MERTON = MertonJumpDiffusion(sigma=0.15, rate=0.5, jump_mean=-0.1, jump_std=0.15)


def test_brownian_motion_gives_a_flat_surface() -> None:
    out = BrownianMotion(sigma=0.25).to_vol_surface(100.0, 0.03, [0.5, 1.0], dividend_yield=0.01)
    assert isinstance(out, ProcessSurface)
    assert out.max_vol_error < 1e-12
    for v in out.model_vols:
        assert np.allclose(v, 0.25, atol=1e-10)
    assert float(out.surface.implied_vol(120.0, 1.0)) == pytest.approx(0.25, abs=1e-10)


def test_merton_model_vols_are_exact_and_svi_is_close() -> None:
    out = process_vol_surface(MERTON, 100.0, 0.03, [0.25, 0.5, 1.0, 2.0], dividend_yield=0.01)
    fwd = out.surface.forward_curve
    for T, k, v in zip(out.expiries, out.log_moneyness, out.model_vols):
        F, D = float(fwd.forward(T)), float(fwd.discount(T))
        for kj, vj in zip(k[::5], v[::5]):
            K = F * math.exp(kj)
            option = "put" if kj < 0 else "call"
            p = merton_price(100.0, K, T, 0.03, 0.15, 0.5, -0.1, 0.15, q=0.01, option=option)
            assert vj == pytest.approx(
                float(implied_black_volatility(p, F, K, T, D, option)), abs=1e-8
            )
    assert out.max_vol_error < 2e-3  # SVI misfit of the Merton smile, 12 bp at most
    # negative jumps: downside skew that flattens with maturity
    skew = [(v[0] - v[-1]) / (k[-1] - k[0]) for k, v in zip(out.log_moneyness, out.model_vols)]
    assert all(s > 0 for s in skew) and skew[0] > skew[-1]


def test_validation() -> None:
    with pytest.raises(ValueError):
        process_vol_surface(MERTON, 100.0, 0.03, [1.0, 0.5])
    with pytest.raises(ValueError):
        process_vol_surface(MERTON, 100.0, 0.03, [1.0], n_strikes=3)
