"""Total-variance interpolation in time."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.curves import EquityForward
from spxa.vol import SSVI, PowerLawPhi, SVIRaw, VarianceInterpolatedSurface
from spxa.vol.surface import VolSurface

TS = [0.25, 0.5, 1.0, 2.0]
K = np.linspace(-0.8, 0.6, 15)
BASE = SVIRaw(0.01, 0.1, -0.4, 0.05, 0.2)


def _scaled(t: float):  # w(k, t) = t * S(k): linear in t at fixed k
    return lambda k: t * BASE.total_variance(k)


def test_nodes_and_fixed_k_linearity() -> None:
    surf = VarianceInterpolatedSurface(TS, [_scaled(t) for t in TS])
    for t in TS:
        assert surf.total_variance(K, t) == pytest.approx(t * BASE.total_variance(K), rel=1e-14)
    for t in (0.1, 0.3, 0.75, 1.6, 3.0, 5.0):  # inside, short end and long end
        assert surf.total_variance(K, t) == pytest.approx(t * BASE.total_variance(K), rel=1e-13)
    assert surf.implied_vol(0.0, 0.75) == pytest.approx(
        np.sqrt(BASE.total_variance(0.0)), rel=1e-13
    )


def test_normalised_mode_is_exact_for_constant_phi_sqrt_theta() -> None:
    ss = SSVI(-0.6, lambda th: 0.5 / np.sqrt(th), lambda t: 0.04 * t + 0.01 * t * t)
    slices = [lambda k, t=t: ss.total_variance(k, t) for t in TS]
    norm = VarianceInterpolatedSurface(TS, slices, "normalised")
    fixed = VarianceInterpolatedSurface(TS, slices, "fixed_k")
    for t in (0.1, 0.37, 0.8, 1.5, 2.6):
        th = norm.theta(t)
        ref = ss._w(K, th, 0.5 / np.sqrt(th), -0.6)  # SSVI at the interpolated ATM variance
        assert norm.total_variance(K, t) == pytest.approx(ref, rel=1e-12)
    # fixed-k mode pins the smile in k instead, which differs off the nodes
    assert np.max(np.abs(fixed.total_variance(K, 0.37) - norm.total_variance(K, 0.37))) > 1e-5


def test_calendar_order_and_surface_wiring() -> None:
    ss = SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t)
    slices = [lambda k, t=t: ss.total_variance(k, t) for t in TS]
    surf = VarianceInterpolatedSurface(TS, slices)
    grid = np.linspace(0.05, 4.0, 200)
    w = np.array([surf.total_variance(K, t) for t in grid])
    assert np.all(np.diff(w, axis=0) > 0)  # calendar-free, extrapolation included
    fwd = EquityForward(100.0, 0.02)
    vs = VolSurface.from_slices(TS, slices, fwd, time_interpolation="normalised")
    assert vs.implied_vol(100.0, 3.0) > 0  # extrapolates, unlike price interpolation
    with pytest.raises(ValueError):
        VolSurface.from_slices(TS, slices, fwd).implied_vol(100.0, 3.0)


def test_single_slice_and_validation() -> None:
    one = VarianceInterpolatedSurface([1.0], [BASE])
    assert one.total_variance(K, 2.5) == pytest.approx(2.5 * BASE.total_variance(K), rel=1e-14)
    assert one.theta(3.0) == pytest.approx(3 * BASE.total_variance(0.0), rel=1e-14)
    with pytest.raises(ValueError):
        VarianceInterpolatedSurface(TS, [BASE] * 4)  # ATM variance not increasing
    with pytest.raises(ValueError):
        VarianceInterpolatedSurface([1.0], [BASE], mode="linear")
    with pytest.raises(ValueError):
        one.total_variance(0.0, 0.0)
