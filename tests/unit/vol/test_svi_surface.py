"""SVI surfaces free of butterfly and calendar arbitrage."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.vol import (
    SSVI,
    PowerLawPhi,
    SVIRaw,
    calibrate_svi,
    calibrate_svi_butterfly_free,
    calibrate_svi_surface,
    check_butterfly,
    check_calendar,
)

K = np.linspace(-0.6, 0.4, 21)
WIDE = np.linspace(-10.0, 10.0, 20001)
# the 1Y slice sits above the 6M one at the money but has a flatter left wing:
# the quotes themselves contain calendar arbitrage for k < -0.45
SHORT = SVIRaw(0.01, 0.1, -0.5, 0.0, 0.15)
LONG = SVIRaw(0.03, 0.06, -0.3, 0.05, 0.2)


def test_crossing_quotes_are_repaired() -> None:
    assert LONG(-0.6) < SHORT(-0.6) and LONG(0.0) > SHORT(0.0)
    independent = [calibrate_svi(K, s(K)).params for s in (SHORT, LONG)]
    assert not check_calendar(independent, WIDE)
    fit = calibrate_svi_surface([0.5, 1.0], [K, K], [SHORT(K), LONG(K)])
    assert fit.success
    assert check_calendar(fit.slices, WIDE)
    assert all(check_butterfly(s, WIDE) for s in fit.slices)
    # wings ordered, so the slices cannot cross beyond the grid either
    s0, s1 = fit.slices
    assert s1.left_slope >= s0.left_slope - 1e-12 and s1.right_slope >= s0.right_slope - 1e-12
    # the first slice is untouched; the second pays for the repair, but only a little
    assert fit.rmse[0] < 1e-9
    assert 1e-4 < fit.rmse[1] < 0.01
    assert fit.total_variance(0.0, 1) == pytest.approx(s1(0.0), abs=0)


def test_consistent_quotes_are_reproduced() -> None:
    # slices of an arbitrage-free SSVI surface need no repair
    surf = SSVI(rho=-0.6, phi=PowerLawPhi(eta=1.0, gamma=0.4), theta=lambda t: 0.04 * t)
    T = [0.25, 0.5, 1.0, 2.0]
    ws = [surf.total_variance(K, t) for t in T]
    fit = calibrate_svi_surface(T, [K] * 4, ws)
    assert fit.success and np.all(fit.rmse < 1e-7)
    assert check_calendar(fit.slices, WIDE)
    for s, t in zip(fit.slices, T):
        assert np.max(np.abs(s(K) - surf.total_variance(K, t))) < 5e-7


def test_minimum_gap() -> None:
    fit = calibrate_svi_surface([0.5, 1.0], [K, K], [SHORT(K), LONG(K)], min_gap=1e-4)
    grid = np.linspace(-1.8, 1.6, 601)  # the default grid: data +- twice its width
    gap = fit.slices[1](grid) - fit.slices[0](grid)
    assert gap.min() >= 1e-4 - 1e-9


def test_lower_slice_constraint_directly() -> None:
    lower = SHORT
    fit = calibrate_svi_butterfly_free(K, LONG(K), lower=lower)
    assert fit.success
    assert check_calendar([lower, fit.params], WIDE)


def test_validation() -> None:
    with pytest.raises(ValueError, match="increasing"):
        calibrate_svi_surface([1.0, 0.5], [K, K], [SHORT(K), LONG(K)])
    with pytest.raises(ValueError, match="one strike"):
        calibrate_svi_surface([0.5, 1.0], [K], [SHORT(K)])
