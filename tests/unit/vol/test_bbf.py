"""Short-expiry implied volatility as the harmonic mean of local volatility."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.sabr import sabr_lognormal_vol
from spxa.vol import LocalVolSurface, VolSurface
from spxa.vol.local_vol import bbf_implied_vol
from tests.unit.vol.test_local_vol import FWD, LinearSmile
from tests.unit.vol.test_local_vol_grid import LV, SURF

K = np.linspace(-0.5, 0.5, 11)


def test_constant_and_linear_local_vol() -> None:
    assert bbf_implied_vol(lambda y: 0.3 + 0 * y, K) == pytest.approx(np.full(11, 0.3), rel=1e-14)
    # sigma_loc = a + b y gives b k / log(1 + b k / a)
    a, b = 0.2, -0.15
    Kz = np.where(K == 0, 1.0, K)
    expect = np.where(K == 0, a, b * Kz / np.log1p(b * Kz / a))
    assert bbf_implied_vol(lambda y: a + b * y, K) == pytest.approx(expect, rel=1e-13)
    assert bbf_implied_vol(lambda y: a + b * y, 0.3) == pytest.approx(expect[8], rel=1e-13)


def test_round_trip_through_dupire() -> None:
    # a time-independent smile: the Dupire local vol at small T maps back to the smile
    lv = LocalVolSurface(VolSurface(LinearSmile(), FWD))
    gaps = []
    for T in (1e-2, 1e-3):
        got = bbf_implied_vol(lambda y: lv.local_vol_k(y, T), K)
        gaps.append(np.max(np.abs(got - (0.2 - 0.1 * K))))
    assert gaps[0] < 1e-8 and gaps[1] < gaps[0] / 50
    # a Heston-like SSVI surface: first order in T
    errs = [
        np.max(
            np.abs(bbf_implied_vol(lambda y: LV.local_vol_k(y, T), K) - SURF.implied_vol_k(K, T))
        )
        for T in (1e-2, 1e-3)
    ]
    assert errs[0] < 1e-4
    assert errs[0] / errs[1] == pytest.approx(10.0, rel=0.05)


def test_cev_limit_is_oblojs_formula() -> None:
    F, alpha, beta = 0.03, 0.035, 0.5
    got = bbf_implied_vol(lambda y: alpha * (F * np.exp(y)) ** (beta - 1), K)
    expect = sabr_lognormal_vol(F, F * np.exp(K), 0.0, alpha, beta, 0.0, 0.0, method="obloj")
    assert got == pytest.approx(expect, rel=1e-13)
