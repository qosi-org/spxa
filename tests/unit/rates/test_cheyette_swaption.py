"""Cheyette swaptions by displaced-diffusion projection."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.cheyette import Cheyette
from spxa.rates.cheyette_mc import cheyette_paths
from spxa.rates.cheyette_swaption import cheyette_swaption, swap_rate_dynamics
from spxa.rates.model_cashflows import swap_cashflows
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_capfloor import DISC, PROJ, REF

STRIKES = (0.025, 0.0346, 0.045)


def test_hull_white_limit() -> None:
    m = Cheyette(DISC, 0.05, 0.01)
    for K in STRIKES:
        sw = swaption(REF, "2Y", "5Y", K, notional=1e4)
        assert cheyette_swaption(m, sw, PROJ) == pytest.approx(
            m.to_hull_white().swaption(sw, PROJ), rel=1e-3
        )


def test_skewed_model_against_simulation() -> None:
    m = Cheyette(DISC, 0.05, 0.01, beta=0.3)
    T = float(DISC.time(swaption(REF, "2Y", "5Y", 0.03).expiry))
    p = cheyette_paths(m, T, 80, 200_000, record=[T], seed=3)
    for K in STRIKES:
        sw = swaption(REF, "2Y", "5Y", K, notional=1e4)
        cf = swap_cashflows(sw.swap, DISC, PROJ)
        V = sum(a * m.discount_bond(T, t, p.x[0], p.y[0]) for t, a in zip(cf.times, cf.amounts))
        pv, se = p.price(np.maximum(V, 0.0), T)
        assert abs(cheyette_swaption(m, sw, PROJ) - pv) < 4 * se + 0.01 * pv


def test_skew_grows_with_beta_and_parity() -> None:
    sw = swaption(REF, "2Y", "5Y", 0.03, notional=1e4)
    skews = [
        swap_rate_dynamics(Cheyette(DISC, 0.05, 0.01, beta=b), sw, PROJ).skew
        for b in (0.0, 0.15, 0.3)
    ]
    assert skews[0] < 0.1 and np.all(np.diff(skews) > 0) and skews[2] > 1.0
    d = swap_rate_dynamics(Cheyette(DISC, 0.05, 0.01, beta=0.3), sw, PROJ)
    assert d.S0 == pytest.approx(sw.forward_rate(DISC, PROJ), rel=1e-12)
    assert d.annuity == pytest.approx(sw.annuity(DISC, PROJ), rel=1e-12)
    assert d.price(0.03) - d.price(0.03, payer=False) == pytest.approx(
        d.annuity * (d.S0 - 0.03), rel=1e-10
    )
