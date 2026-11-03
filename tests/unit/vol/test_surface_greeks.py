"""Surface Greeks under sticky-strike and sticky-moneyness dynamics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.market.black import black_greeks
from spxa.market.curves import EquityForward
from spxa.vol import SSVI, PowerLawPhi
from spxa.vol.surface import VolSurface

FWD = EquityForward(100.0, 0.03, dividend_yield=0.01)
SS = SSVI(-0.7, PowerLawPhi(1.0, 0.4), lambda t: 0.04 * t)
SURF = VolSurface(SS, FWD)
T = 0.5
K = np.array([80.0, 95.0, 100.0, 110.0, 125.0])


def test_flat_surface_dynamics_agree_with_black() -> None:
    flat = VolSurface.flat(0.2, FWD)
    F, D = FWD.forward(T), FWD.discount(T)
    ref = black_greeks(F, K, T, 0.2, D, "call")
    for dyn in ("sticky_strike", "sticky_moneyness"):
        g = flat.greeks(K, T, dynamics=dyn)
        assert g["delta"] == pytest.approx(ref["delta"], abs=5e-8)  # O(h^2) bumps
        assert g["gamma"] == pytest.approx(ref["gamma"], rel=1e-5)
        assert g["vega"] == pytest.approx(ref["vega"], rel=1e-7)


def test_sticky_strike_is_black_at_surface_vol() -> None:
    F, D = FWD.forward(T), FWD.discount(T)
    sig = SURF.implied_vol(K, T)
    ref = black_greeks(F, K, T, sig, D, "put")
    g = SURF.greeks(K, T, "put")
    assert g["price"] == pytest.approx(SURF.price(K, T, "put"), rel=1e-15)
    assert g["delta"] == pytest.approx(ref["delta"], abs=5e-8)  # O(h^2) bumps
    assert g["gamma"] == pytest.approx(ref["gamma"], rel=1e-5)


def test_sticky_moneyness_delta_formula() -> None:
    F, D = FWD.forward(T), FWD.discount(T)
    k = np.log(K / F)
    sig = SS.implied_vol(k, T)
    h = 1e-6
    sig_k = (SS.implied_vol(k + h, T) - SS.implied_vol(k - h, T)) / (2 * h)
    ref = black_greeks(F, K, T, sig, D, "call")
    expected = ref["delta"] - ref["vega"] * sig_k / F
    g = SURF.greeks(K, T, dynamics="sticky_moneyness")
    assert g["delta"] == pytest.approx(expected, abs=1e-7)
    # negative skew: when F rises, k = log(K/F) falls and the vol at K rises,
    # so sticky-moneyness deltas exceed sticky-strike ones
    ss = SURF.greeks(K, T)
    assert np.all(g["delta"] > ss["delta"])
    # gamma is the derivative of that delta
    eps = 1e-3 * F

    def delta_at(Fb: float) -> np.ndarray:
        s = SS.implied_vol(np.log(K / Fb), T)
        sk = (SS.implied_vol(np.log(K / Fb) + h, T) - SS.implied_vol(np.log(K / Fb) - h, T)) / (
            2 * h
        )
        gg = black_greeks(Fb, K, T, s, D, "call")
        return gg["delta"] - gg["vega"] * sk / Fb

    gamma = (delta_at(F + eps) - delta_at(F - eps)) / (2 * eps)
    assert g["gamma"] == pytest.approx(gamma, rel=2e-3)


def test_validation_and_scalar() -> None:
    g = SURF.greeks(100.0, T)
    assert isinstance(g["delta"], float)
    assert g["delta"] == pytest.approx(SURF.greeks(np.array([100.0]), T)["delta"][0], rel=1e-15)
    with pytest.raises(ValueError):
        SURF.greeks(100.0, T, dynamics="sticky_delta")
    with pytest.raises(ValueError):
        SURF.greeks(100.0, 0.0)
    assert math.isfinite(SURF.greeks(60.0, 2.0)["vega"])
