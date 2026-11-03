"""Forward implied variance and the calendar condition."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.curves import EquityForward
from spxa.vol import SSVI, PowerLawPhi, SVIRaw
from spxa.vol.surface import VolSurface

FWD = EquityForward(100.0, 0.03, dividend_yield=0.015)


def theta(t: float) -> float:
    return 0.04 * t + 0.003 * t * t


SURF = VolSurface(SSVI(-0.7, PowerLawPhi(1.1, 0.4), theta), FWD)


def test_atm_forward_variance_closed_form() -> None:
    # ATM w = theta(T): (theta(T2) - theta(T1))/(T2 - T1) = 0.04 + 0.003 (T1 + T2)
    T2 = np.array([0.75, 1.0, 2.0, 5.0])
    fv = SURF.forward_variance(0.5, T2)
    assert fv == pytest.approx(0.04 + 0.003 * (0.5 + T2), rel=1e-13)
    assert SURF.forward_vol(0.5, 1.0) == pytest.approx(np.sqrt(0.04 + 0.0045), rel=1e-13)
    assert SURF.forward_variance(0.0, 2.0, 0.1) == pytest.approx(
        SURF.implied_vol_k(0.1, 2.0) ** 2, rel=1e-14
    )


def test_flat_and_additivity() -> None:
    flat = VolSurface.flat(0.23, FWD)
    assert flat.forward_variance(0.3, 1.7, [-1.0, 0.0, 1.0]) == pytest.approx(0.23**2, rel=1e-13)
    # variance is additive in time: weighted forward variances recombine
    T = [0.25, 0.8, 1.5, 3.0]
    k = np.array([-0.4, 0.0, 0.3])
    grid = SURF.forward_variance_grid(T, k)
    total = SURF.total_variance(k, T[0]) + (np.diff(T)[:, None] * grid).sum(axis=0)
    assert total == pytest.approx(SURF.total_variance(k, T[-1]), rel=1e-13)
    assert np.all(grid > 0)  # SSVI with these parameters is calendar-free


def test_calendar_arbitrage_detected() -> None:
    a = SVIRaw(0.02, 0.1, -0.5, 0.0, 0.2)  # short slice
    b = SVIRaw(0.04, 0.02, -0.5, 0.0, 0.2)  # longer slice with flatter wings
    surf = VolSurface.from_slices([0.5, 1.0], [a, b], FWD)
    grid = surf.forward_variance_grid([0.5, 1.0], [-1.5, 0.0, 1.5])
    assert grid[0, 1] > 0 and grid[0, 0] < 0 and grid[0, 2] < 0
    with pytest.raises(ValueError, match="calendar"):
        surf.forward_vol(0.5, 1.0, 1.5)


def test_validation() -> None:
    with pytest.raises(ValueError):
        SURF.forward_variance(1.0, 1.0)
    with pytest.raises(ValueError):
        SURF.forward_variance(-0.1, 1.0)
    with pytest.raises(ValueError):
        SURF.forward_variance_grid([1.0, 0.5], 0.0)
    assert isinstance(SURF.forward_variance(0.5, 1.0), float)
