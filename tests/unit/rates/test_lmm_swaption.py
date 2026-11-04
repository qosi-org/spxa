"""Rebonato swaption approximation in the LMM."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.lmm import LiborMarketModel
from spxa.rates.lmm_mc import lmm_paths
from tests.unit.rates.test_lmm import CURVE, LMM, TIMES, VOL


def test_single_period_swaption_is_a_caplet() -> None:
    assert LMM.swaption(6, 7, 0.032) == pytest.approx(LMM.caplet(6, 0.032), rel=1e-12)
    assert LMM.swaption_vol(6, 7) == pytest.approx(LMM.caplet_vol(6), rel=1e-12)


def test_parity_and_decorrelation() -> None:
    w, S, A = LMM.swap_weights(2, 12)
    assert w.sum() == pytest.approx(1.0) and S == pytest.approx(LMM.forwards[2:12] @ w)
    payer, receiver = LMM.swaption(2, 12, 0.03), LMM.swaption(2, 12, 0.03, payer=False)
    assert payer - receiver == pytest.approx(A * (S - 0.03), rel=1e-12)
    flat = LiborMarketModel(CURVE, TIMES, VOL)  # perfect correlation
    caplet_vols = [flat.caplet_vol(i) for i in range(2, 12)]
    assert min(caplet_vols) < flat.swaption_vol(2, 12) < max(caplet_vols)
    assert LMM.swaption_vol(2, 12) < flat.swaption_vol(2, 12)


def test_rebonato_against_simulation() -> None:
    p = lmm_paths(LMM, 30_000, 2, seed=7)
    alpha, beta = 2, 12
    L = p.forwards[alpha][:, alpha:beta]
    bonds = np.cumprod(1.0 / (1.0 + LMM.tau[alpha:beta] * L), axis=1)
    annuity = bonds @ LMM.tau[alpha:beta]
    swap = (1.0 - bonds[:, -1]) / annuity
    for K in (0.025, LMM.swap_weights(alpha, beta)[1], 0.04):
        pv, se = p.price(annuity * np.maximum(swap - K, 0.0), alpha)
        assert abs(pv - LMM.swaption(alpha, beta, K)) < 3 * se + 0.01 * pv
