"""Implied-volatility bumps and bucketed vega."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.black import black_greeks
from spxa.market.curves import EquityForward
from spxa.vol import SSVI, PowerLawPhi
from spxa.vol.bumps import bucketed_vega, expiry_buckets, parallel_bump, skew_bump
from spxa.vol.surface import VolSurface

FWD = EquityForward(100.0, 0.03, dividend_yield=0.01)
SURF = VolSurface(SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t), FWD)
K = np.array([80.0, 100.0, 120.0])


def test_parallel_and_skew_shapes() -> None:
    up = SURF.bumped(0.01)
    assert up.implied_vol(K, 0.7) == pytest.approx(SURF.implied_vol(K, 0.7) + 0.01, abs=1e-14)
    assert SURF.bumped(parallel_bump(0.01)).implied_vol(K, 2.0) == pytest.approx(
        SURF.implied_vol(K, 2.0) + 0.01, abs=1e-14
    )
    sk = SURF.bumped(skew_bump(0.05))
    assert sk.atm_vol(1.0) == pytest.approx(SURF.atm_vol(1.0), abs=1e-14)  # pivots at the money
    assert sk.atm_skew(1.0) == pytest.approx(SURF.atm_skew(1.0) + 0.05, abs=1e-7)
    with pytest.raises(ValueError, match="negative"):
        SURF.bumped(-1.0).implied_vol(100.0, 1.0)


def test_buckets_partition_unity() -> None:
    pillars = [0.25, 0.5, 1.0, 2.0]
    bs = expiry_buckets(pillars, 1.0)
    for T in (0.1, 0.25, 0.4, 0.75, 1.0, 1.7, 3.0):
        vals = [float(b(np.zeros(1), T)[0]) for b in bs]
        assert sum(vals) == pytest.approx(1.0, abs=1e-15) and min(vals) >= 0
    assert float(bs[2](np.zeros(1), 1.0)[0]) == 1.0 and float(bs[2](np.zeros(1), 0.75)[0]) == 0.5
    with pytest.raises(ValueError):
        expiry_buckets([1.0, 0.5], 1.0)


def test_bucketed_vega_sums_to_parallel_and_matches_black() -> None:
    h = 1e-4
    pillars = [0.25, 0.5, 1.0, 2.0]
    T = 0.75  # halfway between the 0.5 and 1.0 pillars

    def book(s: VolSurface) -> float:
        return float(np.sum(s.price(K, T)))

    vb = bucketed_vega(book, SURF, expiry_buckets(pillars, h), h)
    vp = bucketed_vega(book, SURF, [parallel_bump(h)], h)[0]
    assert vb.sum() == pytest.approx(vp, rel=1e-7)  # equal up to O(h^2) bump curvature
    assert vb[1] == pytest.approx(vb[2], rel=1e-12) and vb[1] == pytest.approx(0.5 * vp, rel=1e-7)
    assert vb[0] == 0.0 and vb[3] == 0.0
    F, D = FWD.forward(T), FWD.discount(T)
    ref = black_greeks(F, K, T, SURF.implied_vol(K, T), D, "call")["vega"].sum()
    assert vp == pytest.approx(ref, rel=1e-7)
