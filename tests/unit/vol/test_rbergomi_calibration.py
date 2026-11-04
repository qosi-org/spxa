"""Rough Bergomi calibration with common random numbers."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.vol.rbergomi import rbergomi_paths
from spxa.vol.rbergomi_calibration import calibrate_rbergomi

T = 0.5
K = np.exp(np.linspace(-0.25, 0.15, 9))
START = dict(hurst=0.2, eta=1.0, rho=-0.5, xi0=0.03)


def test_exact_recovery_with_the_same_sample() -> None:
    target = rbergomi_paths(0.08, 1.6, -0.85, 0.035, T, 50, 10_000, seed=0).implied_vol(K)
    fit = calibrate_rbergomi(T, K, target, START, free=("hurst", "eta", "rho", "xi0"))
    np.testing.assert_allclose(
        [fit.hurst, fit.eta, fit.rho, fit.xi0], [0.08, 1.6, -0.85, 0.035], rtol=1e-6
    )
    assert fit.rmse < 1e-10


def test_independent_sample_fits_to_monte_carlo_error() -> None:
    # quotes from an independent, larger simulation: the residual is the MC noise of the fixed sample
    target = rbergomi_paths(0.08, 1.6, -0.85, 0.035, T, 50, 100_000, seed=9).implied_vol(K)
    fit = calibrate_rbergomi(
        T, K, target, dict(START, hurst=0.08, xi0=0.035), free=("eta", "rho"), seed=1
    )
    assert fit.rmse < 0.004
    assert abs(fit.rho + 0.85) < 0.05 and abs(fit.eta - 1.6) < 0.1
    with pytest.raises(ValueError):
        calibrate_rbergomi(T, K, target, dict(hurst=0.1, eta=1.0, rho=-0.5))
    with pytest.raises(ValueError):
        calibrate_rbergomi(T, K[:3], target, START)
