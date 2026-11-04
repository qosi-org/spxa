"""LIBOR market model: parametrisation and caplets."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.market import FlatCurve
from spxa.market.black import black_price
from spxa.rates.lmm import (
    AbcdVol,
    LiborMarketModel,
    exponential_correlation,
    rebonato_correlation,
    reduce_rank,
)

CURVE = FlatCurve(0.03)
TIMES = np.arange(1.0, 11.5, 0.5)  # 1Y to 11Y semi-annual: 20 forwards
VOL = AbcdVol(0.05, 0.09, 0.44, 0.11)
LMM = LiborMarketModel(CURVE, TIMES, VOL, correlation=rebonato_correlation(TIMES[:-1], 0.1, 0.4))


def test_abcd_covariance_in_closed_form() -> None:
    for Ti, Tj, t0, t1 in ((5.0, 5.0, 0.0, 5.0), (3.0, 7.0, 0.5, 2.5), (10.0, 4.0, 1.0, 4.0)):
        ref = quad(lambda t: VOL(Ti - t) * VOL(Tj - t), t0, t1, epsabs=0, epsrel=1e-13)[0]
        assert VOL.covariance(Ti, Tj, t0, t1) == pytest.approx(ref, rel=1e-12)
    # the hump of the abcd curve sits at tau = 1/c - a/b
    taus = np.linspace(0, 10, 100001)
    assert taus[np.argmax(VOL(taus))] == pytest.approx(1 / 0.44 - 0.05 / 0.09, abs=1e-4)


def test_forwards_reprice_the_curve() -> None:
    P = np.cumprod(1 / (1 + LMM.tau * LMM.forwards)) * CURVE.discount(TIMES[0])
    np.testing.assert_allclose(P, [CURVE.discount(t) for t in TIMES[1:]], rtol=1e-14)
    assert LMM.n == 20 and np.all(LMM.sigma(1.2)[:1] == 0) and LMM.sigma(1.2)[1] > 0


def test_caplets_are_black_with_integrated_variance() -> None:
    i, K = 7, 0.035
    T = TIMES[i]
    var = quad(lambda t: VOL(T - t) ** 2, 0, T, epsabs=0, epsrel=1e-13)[0]
    ref = black_price(
        LMM.forwards[i], K, T, math.sqrt(var / T), CURVE.discount(TIMES[i + 1]) * 0.5, "call"
    )
    assert LMM.caplet(i, K) == pytest.approx(ref, rel=1e-12)
    floor = LMM.caplet(i, K, option="put")
    assert LMM.caplet(i, K) - floor == pytest.approx(
        CURVE.discount(TIMES[i + 1]) * 0.5 * (LMM.forwards[i] - K), rel=1e-12
    )
    shifted = LiborMarketModel(CURVE, TIMES, VOL, shift=0.02)
    assert shifted.caplet(i, -0.01) > 0 and shifted.caplet_vol(i) == pytest.approx(
        LMM.caplet_vol(i)
    )


def test_covariance_matrix_and_correlations() -> None:
    C = LMM.covariance(0.0, 1.0)
    assert np.allclose(C, C.T) and np.all(np.linalg.eigvalsh(C) > -1e-14)
    s = np.sqrt(np.diag(C))
    np.testing.assert_allclose(C / np.outer(s, s), LMM.correlation, atol=0.05)
    rho = exponential_correlation([1.0, 2.0, 4.0], 0.2)
    assert rho[0, 2] == pytest.approx(math.exp(-0.6))
    B, low = reduce_rank(rebonato_correlation(TIMES[:-1], 0.1, 0.4), 3)
    assert B.shape == (20, 3) and np.allclose(np.diag(low), 1.0)
    full = rebonato_correlation(TIMES[:-1], 0.1, 0.4)
    errs = [np.abs(reduce_rank(full, k)[1] - full).max() for k in (2, 3, 5)]
    assert errs[0] > errs[1] > errs[2] and errs[1] < 0.1
    assert np.abs(reduce_rank(full, 1)[1] - 1.0).max() < 1e-12


def test_validation() -> None:
    with pytest.raises(ValueError):
        LiborMarketModel(CURVE, [1.0, 1.0], VOL)
    with pytest.raises(ValueError):
        LiborMarketModel(CURVE, TIMES, VOL, k=[1.0])
    with pytest.raises(ValueError):
        LiborMarketModel(
            CURVE, [1.0, 2.0, 3.0], VOL, correlation=np.array([[1.0, 0.5], [0.4, 1.0]])
        )
