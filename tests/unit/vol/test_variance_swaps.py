"""Variance, gamma and volatility swaps."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

import spxa.vol as v
from spxa.market import EquityForward
from spxa.vol.variance_swaps import (
    fair_variance,
    gamma_swap_strike,
    variance_swap_mtm,
    variance_swap_strike,
)


class Flat:
    def __init__(self, vol: float) -> None:
        self.vol = vol

    def total_variance(self, k: np.ndarray, T: float) -> np.ndarray:
        return np.full(np.shape(k), self.vol**2 * T)


SSVI_SURFACE = v.VolSurface(
    v.SSVI(-0.7, v.PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t),
    EquityForward(100.0, 0.03, dividend_yield=0.01),
)


def test_flat_smile_gives_sigma_squared() -> None:
    for vol, T in [(0.2, 1.0), (0.05, 0.1), (0.6, 5.0)]:
        assert variance_swap_strike(Flat(vol), T) == pytest.approx(vol * vol, rel=1e-12)
        assert gamma_swap_strike(Flat(vol), T) == pytest.approx(vol * vol, rel=1e-12)


def test_replication_recovers_model_expected_variance() -> None:
    # Heston through the H = 1/2 rough Heston pricer (vectorised over strikes)
    h = v.RoughHeston(0.04, 0.06, 1.5, 0.5, -0.7, 0.5)

    def otm(k: np.ndarray) -> np.ndarray:
        return np.where(
            k >= 1, h.price(k, 1.0, n_steps=400), h.price(k, 1.0, option="put", n_steps=400)
        )

    expected = v.Heston(0.04, 1.5, 0.06, 0.5, -0.7).expected_total_variance(1.0)
    assert fair_variance(otm, 1.0) == pytest.approx(expected, abs=1e-6)
    r = v.RoughHeston(0.02, 0.05, 0.3, 0.3, -0.7, 0.1)
    for T in (0.25, 1.0):
        rough = lambda k, T=T: np.where(k >= 1, r.price(k, T), r.price(k, T, option="put"))  # noqa: E731
        ev = quad(lambda t: float(r.expected_variance(t)), 0, T, epsabs=1e-13)[0] / T
        assert fair_variance(rough, T) == pytest.approx(ev, abs=2e-5)


def test_skew_orders_variance_gamma_and_atm() -> None:
    # negative skew: the 1/k^2 weight loads the expensive puts, 1/k less so
    kvar, kgam = variance_swap_strike(SSVI_SURFACE, 1.0), gamma_swap_strike(SSVI_SURFACE, 1.0)
    atm = float(SSVI_SURFACE.total_variance(np.zeros(1), 1.0)[0])
    assert kvar > kgam > atm
    assert (round(kvar, 4), round(kgam, 4), round(atm, 4)) == (0.0569, 0.0494, 0.043)


def test_mark_to_market() -> None:
    # halfway, realised 0.05 vs remaining fair 0.03: expected 0.04 = strike, so worth zero
    assert variance_swap_mtm(0.05, 0.5, 1.0, 0.04, 0.03, 0.98) == pytest.approx(0.0, abs=1e-17)
    assert variance_swap_mtm(0.06, 0.25, 1.0, 0.04, 0.04, 0.98, notional=100.0) == pytest.approx(
        100.0 * 0.98 * 0.005, rel=1e-14
    )
    with pytest.raises(ValueError):
        variance_swap_mtm(0.05, 1.0, 1.0, 0.04, 0.03, 0.98)
    with pytest.raises(ValueError):
        fair_variance(lambda k: k, 0.0)
