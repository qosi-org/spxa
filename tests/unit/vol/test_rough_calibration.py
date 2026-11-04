"""Rough Heston calibration."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.vol.rough_calibration import calibrate_rough_heston
from spxa.vol.rough_heston import RoughHeston

TRUE = RoughHeston(0.03, 0.05, 0.3, 0.4, -0.65, 0.08)
TS = [0.1, 0.5, 1.5]
KS = [np.exp(np.linspace(-2, 2, 9) * 0.2 * np.sqrt(T)) for T in TS]
VOLS = [TRUE.implied_vol(K, T, n_steps=100) for T, K in zip(TS, KS)]


def test_recovers_parameters_from_its_own_smiles() -> None:
    fit = calibrate_rough_heston(TS, KS, VOLS, RoughHeston(0.04, 0.04, 0.3, 0.3, -0.5, 0.2))
    m = fit.model
    got = [m.v0, m.theta, m.nu, m.rho, m.hurst]
    np.testing.assert_allclose(got, [0.03, 0.05, 0.4, -0.65, 0.08], rtol=2e-3)
    assert m.lam == 0.3 and fit.rmse < 1e-5


def test_fixed_hurst_and_validation() -> None:
    # with H pinned at 1/2 (classical Heston) the short-dated skew cannot be matched
    heston = calibrate_rough_heston(
        TS,
        KS,
        VOLS,
        RoughHeston(0.04, 0.04, 0.3, 0.3, -0.5, 0.5),
        free=("v0", "theta", "nu", "rho"),
    )
    assert heston.model.hurst == 0.5 and heston.rmse > 20 * 1e-5
    with pytest.raises(ValueError):
        calibrate_rough_heston(TS, KS, VOLS, TRUE, free=("kappa",))
    with pytest.raises(ValueError):
        calibrate_rough_heston(TS, KS, VOLS[:2] + [VOLS[2][:3]], TRUE)
