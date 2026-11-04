"""LMM calibration to caplets and swaptions."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.lmm import AbcdVol, LiborMarketModel, rebonato_correlation
from spxa.rates.lmm_calibration import calibrate_lmm, fit_abcd
from tests.unit.rates.test_lmm import CURVE, TIMES

TRUE = AbcdVol(0.05, 0.09, 0.44, 0.11)
FIX = TIMES[:-1]


def test_abcd_recovered_from_its_own_caplets() -> None:
    m = LiborMarketModel(CURVE, TIMES, TRUE)
    fit = fit_abcd(FIX, [m.caplet_vol(i) for i in range(m.n)])
    np.testing.assert_allclose([fit.a, fit.b, fit.c, fit.d], [0.05, 0.09, 0.44, 0.11], rtol=1e-6)


def test_two_stage_calibration() -> None:
    # caplets with a non-homogeneous bump (needs k_i != 1) and correlation (beta, rho_inf) = (0.15, 0.35)
    bump = 1 + 0.05 * np.sin(np.arange(20))
    truth = LiborMarketModel(
        CURVE, TIMES, TRUE, k=bump, correlation=rebonato_correlation(FIX, 0.15, 0.35)
    )
    caps = [truth.caplet_vol(i) for i in range(truth.n)]
    grid = [(2, 6), (2, 12), (4, 10), (6, 20), (10, 20), (2, 20)]
    first = calibrate_lmm(CURVE, TIMES, caps, grid, [truth.swaption_vol(a, b) for a, b in grid])
    # stage two is exact when the swaptions come from the stage-one volatilities
    rho = rebonato_correlation(FIX, 0.15, 0.35)
    target = LiborMarketModel(CURVE, TIMES, first.model.vol, k=first.model.k, correlation=rho)
    swpt = [target.swaption_vol(a, b) for a, b in grid]
    cal = calibrate_lmm(CURVE, TIMES, caps, grid, swpt)
    assert first.max_swaption_error < 2e-3  # the bumped truth is not time-homogeneous
    np.testing.assert_allclose([cal.model.caplet_vol(i) for i in range(20)], caps, rtol=1e-12)
    assert cal.beta == pytest.approx(0.15, rel=1e-6) and cal.rho_inf == pytest.approx(
        0.35, rel=1e-6
    )
    assert cal.max_swaption_error < 1e-10
    with pytest.raises(ValueError):
        calibrate_lmm(CURVE, np.r_[0.0, TIMES], [0.2] * 21, grid, swpt)
