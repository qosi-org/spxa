"""Calibrating SABR smiles."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.sabr import SABR, alpha_from_atm, calibrate_sabr

F, T = 0.03, 2.0
K = F * np.exp(np.linspace(-0.6, 0.6, 13))
TRUE = SABR(0.035, 0.5, -0.25, 0.45)


def test_alpha_from_atm() -> None:
    atm = TRUE.lognormal_vol(F, F, T)
    assert alpha_from_atm(atm, F, T, 0.5, -0.25, 0.45) == pytest.approx(0.035, rel=1e-13)
    # beta = 1 leaves a quadratic, beta = 1 and rho = 0 a linear equation
    m = SABR(0.2, 1.0, 0.0, 0.3)
    assert alpha_from_atm(m.lognormal_vol(F, F, T), F, T, 1.0, 0.0, 0.3) == pytest.approx(
        0.2, rel=1e-14
    )
    with pytest.raises(ValueError):
        alpha_from_atm(-0.1, F, T, 0.5, 0.0, 0.3)


def test_recovers_parameters_from_black_vols() -> None:
    vols = TRUE.lognormal_vol(F, K, T)
    fit = calibrate_sabr(F, K, vols, T, beta=0.5)
    assert fit.rmse < 1e-12
    assert (fit.model.alpha, fit.model.rho, fit.model.nu) == pytest.approx(
        (0.035, -0.25, 0.45), rel=1e-8
    )
    pinned = calibrate_sabr(F, K, vols, T, beta=0.5, atm_vol=TRUE.lognormal_vol(F, F, T))
    assert pinned.model.nu == pytest.approx(0.45, rel=1e-8)


def test_recovers_parameters_from_normal_vols_and_obloj() -> None:
    sh = SABR(0.02, 0.3, 0.1, 0.6, shift=0.01)
    Kn = np.linspace(-0.005, 0.03, 12)
    fit = calibrate_sabr(
        0.005, Kn, sh.normal_vol(0.005, Kn, 5.0), 5.0, beta=0.3, shift=0.01, vol_type="normal"
    )
    assert (fit.model.alpha, fit.model.rho, fit.model.nu) == pytest.approx(
        (0.02, 0.1, 0.6), rel=1e-7
    )
    ob = TRUE.lognormal_vol(F, K, T, "obloj")
    fit = calibrate_sabr(F, K, ob, T, beta=0.5, method="obloj")
    assert fit.model.nu == pytest.approx(0.45, rel=1e-8)


def test_noisy_smile_and_atm_pinning() -> None:
    rng = np.random.default_rng(7)
    vols = TRUE.lognormal_vol(F, K, T) + rng.normal(0.0, 5e-4, K.size)
    atm = float(vols[6])
    free = calibrate_sabr(F, K, vols, T, beta=0.5)
    pinned = calibrate_sabr(F, K, vols, T, beta=0.5, atm_vol=atm)
    assert free.rmse < 6e-4
    assert pinned.errors[6] == pytest.approx(0.0, abs=1e-12)
    assert pinned.rmse >= free.rmse - 1e-15
    with pytest.raises(ValueError):
        calibrate_sabr(F, K, vols, T, beta=0.5, vol_type="normal", atm_vol=atm)
