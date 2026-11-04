"""Black-Karasinski Monte Carlo against its tree."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.bk_mc import black_karasinski_paths
from tests.unit.rates.test_black_karasinski import BK, CURVE

TREE = BK.tree(6.0, 120)
PATHS = black_karasinski_paths(TREE, 100_000, seed=11)


def test_reprices_the_curve() -> None:
    for m in (10, 40, 80, 120):
        d = PATHS.discount[m]
        se = d.std(ddof=1) / np.sqrt(d.size)
        assert abs(d.mean() - CURVE.discount(m * TREE.dt)) < 4 * se + 1e-5


def test_bond_option_matches_tree() -> None:
    m = 40  # t = 2
    S = 6.0
    F = float(CURVE.discount(S) / CURVE.discount(2.0))
    pv, se = PATHS.price(np.maximum(PATHS.zero_bond(m, S) - 0.97 * F, 0.0), m)
    assert abs(pv - TREE.zcb_option(2.0, S, 0.97 * F)) < 4 * se + 1e-5


def test_rates_are_lognormal() -> None:
    r = PATHS.short_rate(60)
    assert np.all(r > 0)
    lr = np.log(r)
    var = BK.sigma**2 * (1 - np.exp(-2 * BK.a * 3.0)) / (2 * BK.a)
    assert lr.var() == pytest.approx(var, rel=0.02)
