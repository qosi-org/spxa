"""Cheyette PDE."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.cheyette import Cheyette
from spxa.rates.cheyette_mc import cheyette_paths
from spxa.rates.cheyette_pde import CheyettePDE
from spxa.rates.hw_pde import HullWhitePDE
from spxa.rates.model_cashflows import swap_cashflows
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_capfloor import DISC, PROJ, REF

HW = Cheyette(DISC, 0.05, 0.01)
SKEWED = Cheyette(DISC, 0.05, 0.01, beta=0.3)


def test_zero_bonds_converge() -> None:
    err = [
        abs(CheyettePDE(SKEWED, 161, 41, n).zero_bond(5.0) / DISC.discount(5.0) - 1)
        for n in (50, 100, 200)
    ]
    assert err[2] < 2e-5 and err[1] < 0.6 * err[0] and err[2] < 0.6 * err[1]


def test_hull_white_limit_and_refinement() -> None:
    sw = swaption(REF, "2Y", "5Y", 0.034, notional=1e4)
    ref = HW.to_hull_white().swaption(sw, PROJ)
    coarse = abs(CheyettePDE(HW, 161, 41, 100).swaption(sw, PROJ) / ref - 1)
    fine = abs(CheyettePDE(HW, 321, 161, 100).swaption(sw, PROJ) / ref - 1)
    assert coarse < 2e-3 and fine < 0.5 * coarse


def test_american_bond_put_against_hull_white_pde() -> None:
    T, S = 2.0, 6.0
    K = float(DISC.discount(S))
    pde = CheyettePDE(HW, 161, 41, 200)

    def put(t: float):  # type: ignore[no-untyped-def]
        return lambda X, Y: np.maximum(K - HW.discount_bond(t, S, X, Y), 0.0)

    eu = pde.solve(put(T), T)
    am = pde.solve(put(T), T, exercise={t: put(t) for t in np.linspace(0, T, 201)[1:-1]})
    hw = HW.to_hull_white()
    assert eu == pytest.approx(hw.zcb_option(T, S, K, "put"), rel=5e-3)
    assert am == pytest.approx(
        HullWhitePDE(hw, 301, 400).zcb_option(T, S, K, "put", american=True), rel=5e-3
    )
    assert am > 4 * eu


def test_skewed_swaptions_match_simulation() -> None:
    T = float(DISC.time(swaption(REF, "2Y", "5Y", 0.03).expiry))
    p = cheyette_paths(SKEWED, T, 80, 200_000, record=[T], seed=3)
    pde = CheyettePDE(SKEWED, 161, 41, 100)
    for K in (0.025, 0.0346, 0.045):
        sw = swaption(REF, "2Y", "5Y", K, notional=1e4)
        cf = swap_cashflows(sw.swap, DISC, PROJ)
        V = sum(
            a * SKEWED.discount_bond(T, t, p.x[0], p.y[0]) for t, a in zip(cf.times, cf.amounts)
        )
        pv, se = p.price(np.maximum(V, 0.0), T)
        assert abs(pde.swaption(sw, PROJ) - pv) < 4 * se + 0.003 * pv
