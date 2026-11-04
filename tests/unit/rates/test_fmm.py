"""Forward market model for backward-looking rates."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.rates.fmm import ForwardMarketModel, fmm_paths
from spxa.rates.lmm import AbcdVol, LiborMarketModel
from tests.unit.rates.test_lmm import CURVE, LMM, TIMES, VOL

FMM = ForwardMarketModel(CURVE, TIMES, VOL, correlation=LMM.correlation)


def test_flat_vol_caplet_variance() -> None:
    flat = ForwardMarketModel(CURVE, TIMES, AbcdVol(0.0, 0.0, 1.0, 0.2))
    i = 6
    T, tau = TIMES[i], TIMES[i + 1] - TIMES[i]
    assert flat.caplet_variance(i) == pytest.approx(0.04 * (T + tau / 3), rel=1e-13)
    # the backward-looking caplet is worth more than the forward-looking one
    lmm = LiborMarketModel(CURVE, TIMES, AbcdVol(0.0, 0.0, 1.0, 0.2))
    assert flat.caplet(i, 0.03) > lmm.caplet(i, 0.03)


def test_covariance_with_decay_against_quadrature() -> None:
    i, j = 4, 5
    ref = quad(
        lambda t: FMM.sigma(t)[i] * FMM.sigma(t)[j],
        0.0,
        TIMES[j + 1],
        points=[TIMES[i], TIMES[j]],
        epsabs=0,
        epsrel=1e-12,
    )[0]
    assert FMM.covariance(0.0, TIMES[j + 1])[i, j] == pytest.approx(
        ref * LMM.correlation[i, j], rel=1e-10
    )
    # before any accrual starts the FMM and the LMM agree
    np.testing.assert_allclose(FMM.covariance(0.0, 0.9), LMM.covariance(0.0, 0.9), rtol=1e-12)
    assert FMM.sigma(TIMES[3] + 0.25)[3] == pytest.approx(0.5 * VOL(0.0))


def test_simulated_caplets_and_bonds() -> None:
    p = fmm_paths(FMM, 20_000, 2, seed=3)
    for k in (1, 10, 20):
        d = p.deflator(k)
        assert abs(d.mean() - CURVE.discount(TIMES[k])) < 3 * d.std(ddof=1) / np.sqrt(d.size)
    for i, K in ((2, 0.03), (12, 0.035)):
        pv, se = p.price(FMM.tau[i] * np.maximum(p.fixings[:, i] - K, 0.0), i + 1)
        assert abs(pv - FMM.caplet(i, K)) < 3 * se
