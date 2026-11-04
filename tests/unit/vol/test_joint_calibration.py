"""Joint SPX/VIX calibration of Heston."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.vol import Heston
from spxa.vol.joint_calibration import SmileQuote, VIXQuote, calibrate_heston_joint
from spxa.vol.vix import HestonVIX

TRUE = Heston(0.03, 2.0, 0.05, 0.8, -0.7)
START = Heston(0.04, 1.0, 0.04, 0.5, -0.5)


def quotes() -> tuple[list[SmileQuote], list[VIXQuote]]:
    spx = []
    for T in (0.25, 1.0):
        K = np.exp(np.linspace(-2, 2, 7) * 0.2 * np.sqrt(T))
        spx.append(SmileQuote(T, K, TRUE.implied_vol(K, T)))
    hv = HestonVIX(TRUE)
    vix = []
    for T in (0.1, 0.5):
        f = hv.futures(T)
        K = f * np.array([0.85, 1.0, 1.2, 1.5])
        vix.append(VIXQuote(T, f, K, hv.implied_vols(T, K)))
    return spx, vix


def test_joint_fit_recovers_heston() -> None:
    spx, vix = quotes()
    fit = calibrate_heston_joint(spx, vix, START)
    got = [fit.model.v0, fit.model.kappa, fit.model.theta, fit.model.xi, fit.model.rho]
    np.testing.assert_allclose(got, [0.03, 2.0, 0.05, 0.8, -0.7], rtol=1e-7)
    assert max(fit.rmse("spx"), fit.rmse("futures"), fit.rmse("vix")) < 1e-10
    with pytest.raises(ValueError):
        calibrate_heston_joint([], [], START)


def test_spx_vix_tension_with_an_upward_vix_smile() -> None:
    spx, vix = quotes()
    # market-like VIX smiles: same futures and ATM vol, slope +0.4 per unit log-moneyness
    market = [
        VIXQuote(q.T, q.futures, q.strikes, q.vols[1] + 0.4 * np.log(q.strikes / q.futures))
        for q in vix
    ]
    spx_only = calibrate_heston_joint(spx, market, START, weights=(1.0, 0.0, 0.0))
    assert spx_only.rmse("spx") < 1e-10 and spx_only.rmse("futures") < 1e-10
    assert spx_only.rmse("vix") > 0.15  # Heston's VIX smile slopes down
    joint = calibrate_heston_joint(spx, market, START)
    # fitting the VIX smile costs SPX and futures accuracy: 2 vol points and 2 VIX points
    assert joint.rmse("vix") < spx_only.rmse("vix") / 4
    assert joint.rmse("spx") > 0.015 and joint.rmse("futures") > 0.015
    assert joint.model.xi > 1.5 and not joint.model.feller
