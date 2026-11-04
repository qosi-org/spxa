"""Hull-White trinomial tree."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.hull_white import HullWhite
from spxa.rates.hw_tree import HullWhiteTree
from tests.unit.rates.test_hull_white import CURVE, HW, HW_PW


def test_probabilities_and_moments() -> None:
    tree = HullWhiteTree(HW, 10.0, 50)
    p = tree._p
    assert np.all(p > 0) and np.allclose(p.sum(axis=0), 1.0, atol=1e-15)
    j = np.arange(-tree.jmax, tree.jmax + 1)
    dest = j + tree._shift
    mean = sum(p[s] * (dest + k) for k, s in zip((1, 0, -1), range(3))) * tree.dx
    M = np.expm1(-HW.a * tree.dt)
    assert mean == pytest.approx(j * tree.dx * (1 + M), abs=1e-15)
    var = sum(p[s] * ((dest + k) * tree.dx - mean) ** 2 for k, s in zip((1, 0, -1), range(3)))
    assert var == pytest.approx(np.full(j.size, HW.sigma[0] ** 2 * tree.dt), rel=1e-12)


def test_reprices_every_zero_bond() -> None:
    tree = HullWhiteTree(HW, 10.0, 80)
    for m in range(1, 81):
        assert tree.discount_bonds(0, m * tree.dt)[0] == pytest.approx(
            CURVE.discount(m * tree.dt), rel=1e-13
        )
        assert tree.arrow_debreu(m).sum() == pytest.approx(CURVE.discount(m * tree.dt), rel=1e-13)


def test_bond_options_converge_to_closed_form() -> None:
    T, S = 2.0, 6.0
    K = float(CURVE.discount(S) / CURVE.discount(T))
    exact = HW.zcb_option(T, S, K)
    # an ATM kink between nodes converges slowly and not monotonically
    errs = [abs(HullWhiteTree(HW, S, n).zcb_option(T, S, K) - exact) for n in (30, 960)]
    assert errs[1] < 1e-3 * exact and errs[1] < errs[0] / 10
    itm = HW.zcb_option(T, S, 0.9 * K)
    assert HullWhiteTree(HW, S, 240).zcb_option(T, S, 0.9 * K) == pytest.approx(itm, rel=1e-4)
    put = HullWhiteTree(HW, S, 120)
    eu = put.zcb_option(T, S, K, "put")
    am = put.zcb_option(T, S, K, "put", american=True)
    assert am >= eu - 1e-15
    assert eu == pytest.approx(HW.zcb_option(T, S, K, "put"), rel=1e-2)


def test_validation() -> None:
    with pytest.raises(ValueError):
        HullWhiteTree(HW_PW, 5.0, 10)
    tree = HullWhiteTree(HullWhite(CURVE, 0.1, 0.01), 5.0, 10)
    with pytest.raises(ValueError):
        tree.step(0.33)
