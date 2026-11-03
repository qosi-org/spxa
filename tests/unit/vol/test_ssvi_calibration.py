"""SSVI surface calibration."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.vol import (
    SSVI,
    ATMVarianceCurve,
    HestonLikePhi,
    PowerLawPhi,
    calibrate_ssvi,
    check_butterfly,
    check_calendar,
    isotonic_increasing,
)

T = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
TH = np.array([0.004, 0.009, 0.017, 0.033, 0.064])
KS = [np.linspace(-1.2 * np.sqrt(t), 0.8 * np.sqrt(t), 15) for t in T]


def _data(s: SSVI) -> list[np.ndarray]:
    return [np.asarray(s.total_variance(k, t)) for k, t in zip(KS, T)]


def test_isotonic() -> None:
    assert isotonic_increasing([1, 3, 2, 4]) == pytest.approx([1, 2.5, 2.5, 4], abs=1e-15)
    assert isotonic_increasing([3, 2, 1]) == pytest.approx([2, 2, 2], abs=1e-15)
    assert isotonic_increasing([1, 3, 2], weights=[1, 3, 1]) == pytest.approx(
        [1, 2.75, 2.75], abs=1e-15
    )
    y = np.array([0.1, 0.2, 0.3])
    assert np.array_equal(isotonic_increasing(y), y)


def test_power_law_recovery() -> None:
    true = SSVI(-0.7, PowerLawPhi(1.1, 0.4), ATMVarianceCurve(T, TH))
    fit = calibrate_ssvi(T, KS, _data(true))
    assert fit.success and fit.rmse < 1e-7
    assert fit.thetas == pytest.approx(TH, rel=1e-9)
    assert fit.surface.rho == pytest.approx(-0.7, abs=1e-5)
    assert fit.surface.phi.eta == pytest.approx(1.1, rel=1e-5)
    assert fit.surface.phi.gamma == pytest.approx(0.4, rel=1e-5)


def test_heston_like_recovery() -> None:
    true = SSVI(-0.6, HestonLikePhi(2.0), ATMVarianceCurve(T, TH))
    fit = calibrate_ssvi(T, KS, _data(true), phi="heston")
    assert fit.success and fit.rmse < 1e-7
    assert fit.surface.rho == pytest.approx(-0.6, abs=1e-6)
    assert fit.surface.phi.lam == pytest.approx(2.0, rel=1e-6)


def test_no_arbitrage_constraint_binds() -> None:
    # eta (1 + |rho|) = 2.72 > 2 violates Remark 4.4
    steep = SSVI(-0.7, PowerLawPhi(1.6, 0.4), ATMVarianceCurve(T, TH))
    free = calibrate_ssvi(T, KS, _data(steep), no_arbitrage=False)
    assert free.surface.phi.eta == pytest.approx(1.6, rel=1e-5)
    fit = calibrate_ssvi(T, KS, _data(steep))
    p = fit.surface.phi
    assert p.eta * (1 + abs(fit.surface.rho)) <= 2 + 1e-9 and p.gamma <= 0.5
    assert fit.rmse > 1e-3
    assert fit.surface.is_arbitrage_free(np.geomspace(1e-4, 1.0, 100))
    k = np.linspace(-2, 2, 801)
    slices = [fit.surface.slice(t) for t in np.linspace(0.05, 3.0, 30)]
    assert all(check_butterfly(s, k) for s in slices)
    assert check_calendar(slices, k)


def test_noisy_atm_made_monotone() -> None:
    true = SSVI(-0.7, PowerLawPhi(1.1, 0.4), ATMVarianceCurve(T, TH))
    ws = _data(true)
    ws[2] = ws[2] * 0.4  # a stale slice with too little variance
    fit = calibrate_ssvi(T, KS, ws)
    assert np.all(np.diff(fit.thetas) >= 0)
    assert fit.thetas[1] == fit.thetas[2]


def test_validation() -> None:
    true = SSVI(-0.7, PowerLawPhi(1.1, 0.4), ATMVarianceCurve(T, TH))
    ws = _data(true)
    with pytest.raises(ValueError, match="one"):
        calibrate_ssvi(T[:3], KS, ws)
    with pytest.raises(ValueError, match="increasing"):
        calibrate_ssvi(T[::-1], KS, ws)
    with pytest.raises(ValueError, match="phi"):
        calibrate_ssvi(T, KS, ws, phi="sabr")
