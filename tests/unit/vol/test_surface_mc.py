"""Monte Carlo checks of surface prices against sampled terminal laws."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.curves import EquityForward
from spxa.vol import SSVI, PowerLawPhi, SVIRaw
from spxa.vol.surface import VolSurface

FWD = EquityForward(100.0, 0.03, dividend_yield=0.015)
SURF = VolSurface(SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t), FWD)
N = 400_000


def _check(surf: VolSurface, T: float, seed: int) -> None:
    S = surf.sample_terminal(T, N, rng=seed)
    F, D = surf.forward(T), surf.discount(T)
    # martingale: E[S_T] = F_T
    assert abs(S.mean() - F) < 4 * S.std() / np.sqrt(N)
    for K in F * np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]):
        pay = D * np.maximum(S - K, 0.0)
        se = pay.std() / np.sqrt(N)
        assert abs(pay.mean() - surf.price(K, T)) < 4 * se + 1e-4 * D * F
        dig = D * (S > K)
        sed = dig.std() / np.sqrt(N)
        h = 1e-3 * K
        digital = -(surf.price(K + h, T) - surf.price(K - h, T)) / (2 * h)
        assert abs(dig.mean() - digital) < 4 * sed + 1e-4


@pytest.mark.parametrize("T", [0.25, 1.0, 3.0])
def test_ssvi_surface(T: float) -> None:
    _check(SURF, T, seed=int(T * 100))


def test_price_interpolated_surface_between_slices() -> None:
    slices = [SVIRaw(0.002, 0.05, -0.5, 0.02, 0.15), SVIRaw(0.012, 0.1, -0.45, 0.03, 0.2)]
    surf = VolSurface.from_slices([0.25, 1.0], slices, FWD)
    _check(surf, 0.6, seed=7)  # a mixture law: still a martingale with matching prices
