"""Bermudan swaptions on the Hull–White PDE and tree."""

from __future__ import annotations

import pytest

from spxa.rates.bermudan import bermudan_from_times
from spxa.rates.bermudan_engines import bermudan_pde, bermudan_tree
from spxa.rates.cheyette import Cheyette
from spxa.rates.cheyette_pde import CheyettePDE
from spxa.rates.hull_white import HullWhite
from spxa.rates.hw_pde import HullWhitePDE
from spxa.rates.hw_tree import HullWhiteTree
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_bermudan import DISC, PROJ, REF, _ber

HW = HullWhite(DISC, 0.05, 0.01)
PDE = HullWhitePDE(HW, 401, 400)


def test_single_exercise_is_the_european() -> None:
    ber = _ber()
    eu = swaption(REF, "1Y", "5Y", 0.035, notional=1e4)
    assert bermudan_pde(PDE, ber.restrict([0])) == pytest.approx(HW.swaption(eu, PROJ), rel=1e-4)


def test_bermudan_bounds() -> None:
    ber = _ber()
    europeans = [bermudan_pde(PDE, ber.restrict([k])) for k in range(ber.n_exercise)]
    value = bermudan_pde(PDE, ber)
    assert max(europeans) < value < sum(europeans)
    # adding exercise dates never lowers the value
    assert bermudan_pde(PDE, ber.restrict([1, 3])) < value


@pytest.mark.parametrize("payer", [True, False])
def test_tree_agrees_with_pde(payer: bool) -> None:
    ber = bermudan_from_times([1.0, 2.0, 3.0, 4.0], 5.0, 0.03, notional=1e4, payer=payer)
    tree = bermudan_tree(HullWhiteTree(HW, 5.0, 1000), ber)
    assert tree == pytest.approx(bermudan_pde(PDE, ber), rel=5e-4)
    with pytest.raises(ValueError):
        bermudan_tree(HullWhiteTree(HW, 4.0, 400), ber)
    with pytest.raises(TypeError):
        bermudan_pde(object(), ber)


def test_cheyette_pde_bermudan() -> None:
    ber = _ber()
    hw_value = bermudan_pde(PDE, ber)
    flat = CheyettePDE(Cheyette(DISC, 0.05, 0.01, beta=0.0), 161, 41, 200)
    assert bermudan_pde(flat, ber) == pytest.approx(hw_value, rel=1e-3)
    eu = swaption(REF, "1Y", "5Y", 0.035, notional=1e4)
    assert bermudan_pde(flat, ber.restrict([0])) == pytest.approx(HW.swaption(eu, PROJ), rel=2e-3)
    skewed = CheyettePDE(Cheyette(DISC, 0.05, 0.01, beta=0.3), 161, 41, 200)
    europeans = [bermudan_pde(skewed, ber.restrict([k])) for k in range(ber.n_exercise)]
    assert max(europeans) < bermudan_pde(skewed, ber) < sum(europeans)
