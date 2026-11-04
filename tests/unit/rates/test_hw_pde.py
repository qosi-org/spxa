"""Hull-White finite differences."""

from __future__ import annotations

import pytest

from spxa.rates.hw_pde import HullWhitePDE
from spxa.rates.hw_tree import HullWhiteTree
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_capfloor import PROJ
from tests.unit.rates.test_hull_white import CURVE, HW, HW_PW


def test_zero_bonds_converge_at_second_order() -> None:
    errs = [
        abs(HullWhitePDE(HW_PW, n, m).zero_bond(10.0) - CURVE.discount(10.0))
        for n, m in ((101, 50), (201, 100), (401, 200))
    ]
    assert errs[-1] < 2e-5
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.25)
    assert errs[1] / errs[2] == pytest.approx(4.0, rel=0.25)


def test_bond_options_and_swaptions() -> None:
    pde = HullWhitePDE(HW_PW, 301, 300)
    T, S = 2.0, 7.0
    K = float(CURVE.discount(S) / CURVE.discount(T))
    for opt in ("call", "put"):
        assert pde.zcb_option(T, S, K, opt) == pytest.approx(
            HW_PW.zcb_option(T, S, K, opt), rel=2e-4
        )
    sw = swaption("2024-03-15", "2Y", "5Y", 0.035, notional=1e6)
    assert pde.swaption(sw, PROJ) == pytest.approx(HW_PW.swaption(sw, PROJ), rel=2e-4)


def test_american_bond_put_agrees_with_the_tree() -> None:
    # struck at today's bond price: the bond accretes towards par, so early exercise pays at high rates
    T, S = 2.0, 6.0
    K = float(CURVE.discount(S))
    am = HullWhitePDE(HW, 301, 400).zcb_option(T, S, K, "put", american=True)
    assert am > 4 * HW.zcb_option(T, S, K, "put")
    errs = [
        abs(HullWhiteTree(HW, S, n).zcb_option(T, S, K, "put", american=True) / am - 1)
        for n in (150, 300)
    ]
    assert errs[1] < 3e-3 and errs[1] < 0.6 * errs[0]  # the tree converges at first order
