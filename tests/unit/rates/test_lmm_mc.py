"""LIBOR market model Monte Carlo."""

from __future__ import annotations

import numpy as np

from spxa.rates.lmm import LiborMarketModel
from spxa.rates.lmm_mc import lmm_paths
from tests.unit.rates.test_lmm import CURVE, LMM, TIMES, VOL


def test_deflated_bonds_are_martingales() -> None:
    p = lmm_paths(LMM, 40_000, 1, seed=1)
    for k in (1, 5, 10, 15, 20):
        d = p.deflator(k)
        assert abs(d.mean() - CURVE.discount(TIMES[k])) < 3 * d.std(ddof=1) / np.sqrt(d.size)
    # P(T_k, T_e) / N(T_k) is a martingale too
    pv, se = p.price(p.bond(5, 12), 5)
    assert abs(pv - CURVE.discount(TIMES[12])) < 3 * se


def test_caplets_match_black() -> None:
    p = lmm_paths(LMM, 40_000, 2, seed=2)
    for i, K in ((3, 0.03), (10, 0.04), (19, 0.025), (19, 0.05)):
        pv, se = p.price(LMM.tau[i] * np.maximum(p.fixings[:, i] - K, 0.0), i + 1)
        assert abs(pv - LMM.caplet(i, K)) < 3 * se


def test_predictor_corrector_with_coarse_steps() -> None:
    # high vol and one step per period: the predictor-corrector drift keeps bonds unbiased
    hot = LiborMarketModel(
        CURVE, TIMES, type(VOL)(0.3, 0.2, 0.5, 0.25), correlation=LMM.correlation
    )
    p = lmm_paths(hot, 40_000, 1, seed=4)
    for k in (10, 20):
        d = p.deflator(k)
        assert abs(d.mean() - CURVE.discount(TIMES[k])) < 3 * d.std(ddof=1) / np.sqrt(d.size)
    assert p.forwards.shape == (20, 40_000, 20)
