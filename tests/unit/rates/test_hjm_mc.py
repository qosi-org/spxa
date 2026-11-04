"""Gaussian HJM Monte Carlo."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.hjm import GaussianHJM, exponential_vol, hump_vol
from spxa.rates.hjm_mc import hjm_paths
from tests.unit.rates.test_hull_white import CURVE

MATS = np.linspace(0.0, 10.0, 81)
HJM = GaussianHJM(CURVE, [hump_vol(0.004, 0.006, 0.4), exponential_vol(0.006, 0.03)])
PATHS = hjm_paths(
    HJM, MATS, 100_000, record=[0.0, 2.0, 5.0], keep=[2.0, 5.0, 6.0, 7.0, 10.0], seed=4
)


def test_deflated_bonds_are_martingales() -> None:
    for t in (2.0, 5.0):
        for T in (t, 6.0, 10.0):
            z = PATHS.deflated(t, T)
            se = z.std(ddof=1) / np.sqrt(z.size)
            assert abs(z.mean() - CURVE.discount(T)) < 4 * se


def test_bond_option_matches_closed_form() -> None:
    from spxa.market.black import black_price

    T, S = 2.0, 7.0
    K = float(CURVE.discount(S) / CURVE.discount(T))
    pv, se = PATHS.price(np.maximum(PATHS.zero_bond(T, S) - K, 0.0), T)
    # Z(T, S) and Z(T, T) are jointly lognormal with the left-point variance, so the
    # scheme prices exactly Black at that variance (Margrabe with numeraire Z(T, T))
    left = sum(
        float(np.sum((HJM.Sigma(t, S) - HJM.Sigma(t, T)) ** 2)) * 0.125
        for t in MATS[MATS < T - 1e-12]
    )
    pT, pS = float(CURVE.discount(T)), float(CURVE.discount(S))
    assert abs(pv - black_price(pS / pT, K, 1.0, np.sqrt(left), pT, "call")) < 4 * se
    # and the left-point variance converges to the exact one: 0.8% low at dt = 1/8
    exact = HJM.forward_bond_variance(T, T, S)
    assert -0.01 < left / exact - 1 < 0
    assert abs(pv - HJM.zcb_option(T, S, K)) < 4 * se + 0.005 * pv


def test_validation() -> None:
    with pytest.raises(ValueError):
        hjm_paths(HJM, [0.5, 1.0], 10)
    with pytest.raises(ValueError):
        hjm_paths(HJM, MATS, 10, record=[0.3])
    with pytest.raises(ValueError):
        PATHS.deflated(1.0, 2.0)
