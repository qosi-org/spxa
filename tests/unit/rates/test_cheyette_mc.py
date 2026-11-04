"""Cheyette Monte Carlo."""

from __future__ import annotations

import numpy as np

from spxa.rates.cheyette import Cheyette
from spxa.rates.cheyette_mc import cheyette_paths
from spxa.rates.model_cashflows import swap_cashflows
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_capfloor import DISC, PROJ, REF

SKEWED = Cheyette(DISC, 0.05, 0.01, beta=0.3)
PATHS = cheyette_paths(SKEWED, 5.0, 100, 100_000, record=[2.0, 5.0], seed=8)
SW = swaption(REF, "2Y", "5Y", 0.034, notional=1e4)
TX = float(DISC.time(SW.expiry))


def test_deflated_bonds_reprice_the_curve() -> None:
    for t in (2.0, 5.0):
        for T in (t, 7.0, 15.0):
            v = PATHS.deflator[PATHS._row(t)] * PATHS.zero_bond(t, T)
            se = v.std(ddof=1) / np.sqrt(v.size)
            assert abs(v.mean() - DISC.discount(T)) < 4 * se + 2e-5


def _payer(paths, model, sw) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    T = float(DISC.time(sw.expiry))
    cf = swap_cashflows(sw.swap, DISC, PROJ)
    k = paths._row(T)
    V = sum(
        a * model.discount_bond(T, t, paths.x[k], paths.y[k]) for t, a in zip(cf.times, cf.amounts)
    )
    return paths.price(np.maximum(V, 0.0), T)


def test_beta_zero_matches_hull_white_swaptions() -> None:
    m = Cheyette(DISC, 0.05, 0.01)
    p = cheyette_paths(m, TX, 40, 100_000, record=[TX], seed=2)
    pv, se = _payer(p, m, SW)
    assert abs(pv - m.to_hull_white().swaption(SW, PROJ)) < 4 * se


def test_skew_shifts_value_to_high_strikes() -> None:
    # with beta > 0 vol rises with rates: OTM payers gain relative to OTM receivers
    hw = Cheyette(DISC, 0.05, 0.01)
    p0 = cheyette_paths(hw, TX, 40, 50_000, record=[TX], seed=1)
    p1 = cheyette_paths(Cheyette(DISC, 0.05, 0.01, beta=0.3), TX, 40, 50_000, record=[TX], seed=1)
    hi = swaption(REF, "2Y", "5Y", 0.045, notional=1e4)
    lo = swaption(REF, "2Y", "5Y", 0.025, notional=1e4, payer=False)
    r0 = _payer(p0, hw, hi)[0] / _payer(p0, hw, lo)[0]
    r1 = _payer(p1, p1.model, hi)[0] / _payer(p1, p1.model, lo)[0]
    assert r1 > 2 * r0  # 2.10 vs 0.91 at these seeds
