"""VolSurface: strike-space surfaces on a forward curve."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.market.black import black_price, black_scholes_price
from spxa.market.curves import EquityForward, FlatCurve
from spxa.vol import SSVI, PowerLawPhi, PriceInterpolatedSurface
from spxa.vol.surface import VolSurface

FWD = EquityForward(100.0, 0.03, dividend_yield=0.01)
SS = SSVI(-0.6, PowerLawPhi(1.0, 0.4), lambda t: 0.04 * t)


def test_flat_is_black_scholes() -> None:
    s = VolSurface.flat(0.2, FWD)
    K = np.array([80.0, 100.0, 120.0])
    for T in (0.25, 1.0, 3.0):
        ref = black_scholes_price(100.0, K, T, 0.03, 0.2, q=0.01)
        assert s.price(K, T) == pytest.approx(ref, rel=1e-14)
        put = black_scholes_price(100.0, K, T, 0.03, 0.2, q=0.01, option="put")
        assert s.price(K, T, "put") == pytest.approx(put, rel=1e-14)
    assert s.implied_vol(90.0, 2.0) == pytest.approx(0.2, abs=1e-16)
    assert s.atm_vol(np.array([0.5, 1.0])) == pytest.approx([0.2, 0.2], abs=1e-16)


def test_from_spot_matches_forward() -> None:
    s = VolSurface.from_spot(SS, 100.0, 0.03, 0.01)
    assert s.forward(2.0) == pytest.approx(100 * math.exp(0.04), rel=1e-14)
    assert s.discount(2.0) == pytest.approx(math.exp(-0.06), rel=1e-14)
    curve = VolSurface(SS, EquityForward(100.0, FlatCurve(0.03), FlatCurve(0.01)))
    assert curve.price(110.0, 1.5) == pytest.approx(s.price(110.0, 1.5), rel=1e-14)


def test_ssvi_backed_surface() -> None:
    s = VolSurface(SS, FWD)
    T = 0.75
    K = np.array([70.0, 90.0, 100.0, 115.0, 140.0])
    k = np.log(K / FWD.forward(T))
    assert s.log_moneyness(K, T) == pytest.approx(k, abs=1e-15)
    assert s.strike(k, T) == pytest.approx(K, rel=1e-15)
    assert s.implied_vol(K, T) == pytest.approx(SS.implied_vol(k, T), rel=1e-14)
    ref = black_price(FWD.forward(T), K, T, SS.implied_vol(k, T), FWD.discount(T))
    assert s.price(K, T) == pytest.approx(ref, rel=1e-14)
    # put-call parity: C - P = P(0,T)(F - K)
    par = s.price(K, T) - s.price(K, T, "put")
    assert par == pytest.approx(FWD.discount(T) * (FWD.forward(T) - K), abs=1e-12)
    assert s.smile(T, K) == pytest.approx(s.implied_vol(K, T), abs=0)


def test_atm_skew_matches_ssvi() -> None:
    s = VolSurface(SS, FWD)
    for T in (0.1, 1.0, 4.0):
        th = 0.04 * T
        exact = -0.6 * SS.phi(th) * math.sqrt(th) / (2 * math.sqrt(T))
        assert s.atm_skew(T) == pytest.approx(exact, rel=1e-6)
        assert s.atm_vol(T) == pytest.approx(0.2, rel=1e-14)


def test_broadcasting_and_zero_expiry() -> None:
    s = VolSurface(SS, FWD)
    K = np.array([[90.0], [110.0]])
    T = np.array([0.0, 0.5, 2.0])
    p = s.price(K, T)
    assert p.shape == (2, 3)
    assert p[0, 0] == pytest.approx(10.0, abs=1e-14) and p[1, 0] == 0.0  # intrinsic at T=0
    for i in range(2):
        for j in (1, 2):
            assert p[i, j] == pytest.approx(s.price(float(K[i, 0]), float(T[j])), rel=1e-15)
    assert s.total_variance(np.array([0.1, -0.1]), 0.0).tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        s.implied_vol(100.0, 0.0)
    with pytest.raises(ValueError):
        s.price(100.0, -1.0)
    with pytest.raises(ValueError):
        s.atm_skew(0.0)


def test_from_slices() -> None:
    T = [0.5, 1.0, 2.0]
    slices = [SS.slice(t) for t in T]
    s = VolSurface.from_slices(T, slices, FWD)
    k = np.linspace(-0.5, 0.3, 9)
    for t, sl in zip(T, slices):
        assert s.total_variance(k, t) == pytest.approx(sl(k), rel=1e-14)
    pis = PriceInterpolatedSurface(T, slices)
    assert s.total_variance(k, 0.75) == pytest.approx(pis.total_variance(k, 0.75), rel=1e-15)
    assert s.atm_vol(0.75) == pytest.approx(0.2, abs=1e-4)
    with pytest.raises(ValueError):
        s.implied_vol_k(0.0, 3.0)  # no extrapolation beyond the last slice


def test_validation_and_repr() -> None:
    with pytest.raises(TypeError):
        VolSurface(object(), FWD)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VolSurface.flat(-0.1, FWD)
    assert repr(VolSurface(SS, FWD)) == "VolSurface(model=SSVI)"
