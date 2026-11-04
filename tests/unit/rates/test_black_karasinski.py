"""Black-Karasinski trinomial tree."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.black_karasinski import BlackKarasinski
from tests.unit.rates.test_hull_white import CURVE

BK = BlackKarasinski(CURVE, 0.1, 0.25)


def test_reprices_bonds_and_keeps_rates_positive() -> None:
    tree = BK.tree(10.0, 100)
    for m in (1, 10, 50, 100):
        assert tree.discount_bonds(0, m * tree.dt)[0] == pytest.approx(
            CURVE.discount(m * tree.dt), rel=1e-13
        )
    assert all(np.all(tree.rates(m) > 0) for m in range(100))
    # the node rates are lognormally spaced
    r = tree.rates(40)
    assert np.diff(np.log(r)) == pytest.approx(np.full(r.size - 1, tree.dx), rel=1e-12)


def test_bond_options_converge() -> None:
    T, S = 2.0, 6.0
    F = float(CURVE.discount(S) / CURVE.discount(T))
    v = [BK.tree(S, n).zcb_option(T, S, 0.97 * F) for n in (60, 120, 240, 480)]
    d = np.abs(np.diff(v))
    assert d[2] < 0.7 * d[1] < 0.7 * 0.7 * d[0]
    assert d[2] / v[3] < 1e-3
    eu = BK.tree(S, 120).zcb_option(T, S, F, "put")
    am = BK.tree(S, 120).zcb_option(T, S, F, "put", american=True)
    assert am > eu


def test_zero_vol_limit_is_the_forward_intrinsic() -> None:
    T, S = 2.0, 6.0
    F = float(CURVE.discount(S) / CURVE.discount(T))
    tree = BlackKarasinski(CURVE, 0.1, 1e-4).tree(S, 120)
    assert tree.zcb_option(T, S, 0.99 * F) == pytest.approx(CURVE.discount(T) * 0.01 * F, rel=1e-10)


def test_validation() -> None:
    with pytest.raises(ValueError):
        BlackKarasinski(CURVE, 0.0, 0.2)
    with pytest.raises(ValueError):
        BK.tree(-1.0, 10)
