"""Gyöngy: the local-vol diffusion has the marginals implied by the surface."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.vol import LocalVolSurface, VolSurface
from tests.unit.vol.test_local_vol_grid import FWD, LV, SURF


def _breeden_litzenberger(surf: VolSurface, y: np.ndarray, T: float) -> np.ndarray:
    F, D = float(surf.forward(T)), float(surf.discount(T))
    K = F * np.exp(y)
    h = 1e-3 * K
    d2c = (surf.price(K + h, T) - 2 * surf.price(K, T) + surf.price(K - h, T)) / h**2
    return d2c / D * K  # density of y = log(K/F)


def test_fokker_planck_density_matches_breeden_litzenberger() -> None:
    y, p = LV.density(0.5, 100.0, 0.03, 0.01)
    assert np.trapezoid(p, y) == pytest.approx(1.0, abs=1e-10)
    yy = np.linspace(-0.5, 0.4, 10)
    bl = _breeden_litzenberger(SURF, yy, 0.5)
    assert np.interp(yy, y, p) == pytest.approx(bl, abs=2e-4)
    # the forward is a martingale under the density
    assert np.trapezoid(p * np.exp(y), y) == pytest.approx(1.0, abs=1e-6)


def test_flat_surface_gives_the_gaussian() -> None:
    lv = LocalVolSurface(VolSurface.flat(0.2, FWD))
    y, p = lv.density(1.0, 100.0, 0.03, 0.01, n=401, n_time=200)
    gauss = np.exp(-((y + 0.02) ** 2) / (2 * 0.04)) / np.sqrt(2 * np.pi * 0.04)
    assert np.max(np.abs(p - gauss)) < 5e-4  # peak density 2.0
