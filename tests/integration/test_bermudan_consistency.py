"""Bermudan swaptions priced consistently across models and engines."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates import (
    Cheyette,
    CheyettePDE,
    HullWhitePDE,
    HullWhiteTree,
    andersen_broadie,
    bermudan_from_times,
    bermudan_pde,
    bermudan_swaption,
    bermudan_tree,
    calibrate_hull_white,
    cheyette_exercise_problem,
    coterminal_swaptions,
    hull_white_exercise_problem,
    lsm,
    market_prices,
    model_price,
    swaption,
)
from tests.unit.rates.test_bermudan import DISC, PROJ, REF

SWAP = swaption(REF, "1Y", "6Y", 0.035, notional=1e4).swap  # 1Y forward 6Y payer
BER = bermudan_swaption(SWAP, DISC, PROJ, lockout=1)  # 7Y-non-call-2
VOLS = np.array([92.0, 89.0, 86.0, 83.0, 80.0]) * 1e-4


def _bracket(problem, value: float) -> None:
    res = lsm(problem, 100_000, seed=1, n_train=50_000)
    dual = andersen_broadie(res, 500, 300, seed=2)
    assert dual.lower - 3 * dual.lower_stderr < value < dual.upper + 3 * dual.stderr
    assert dual.gap < 0.002 * value


def test_coterminal_calibration_and_bermudan() -> None:
    insts = coterminal_swaptions(SWAP, DISC, lockout=1)
    assert len(insts) == BER.n_exercise == 5
    np.testing.assert_allclose([DISC.time(s.expiry) for s in insts], BER.exercise_times, atol=1e-14)
    targets = market_prices(insts, VOLS, DISC, PROJ, "normal")
    cal = calibrate_hull_white(DISC, insts, targets, PROJ, a=0.03)
    hw = cal.model
    np.testing.assert_allclose([model_price(hw, s, PROJ) for s in insts], targets, rtol=1e-10)
    pde = HullWhitePDE(hw, 401, 400)
    # the PDE reprices the calibration set through the Bermudan's own underlyings
    for k in range(BER.n_exercise):
        assert bermudan_pde(pde, BER.restrict([k])) == pytest.approx(targets[k], rel=1e-3)
    value = bermudan_pde(pde, BER)
    assert max(targets) < value < targets.sum()
    _bracket(hull_white_exercise_problem(hw, BER), value)


def test_tree_pde_and_monte_carlo_agree() -> None:
    from spxa.rates import HullWhite

    hw = HullWhite(DISC, 0.03, 0.009)
    ber = bermudan_from_times([2.0, 3.0, 4.0, 5.0, 6.0], 7.0, 0.03, notional=1e4, payer=False)
    pde = bermudan_pde(HullWhitePDE(hw, 401, 400), ber)
    assert bermudan_tree(HullWhiteTree(hw, 7.0, 1400), ber) == pytest.approx(pde, rel=5e-4)
    _bracket(hull_white_exercise_problem(hw, ber), pde)


@pytest.mark.parametrize("beta", [0.0, 0.3])
def test_cheyette_pde_and_monte_carlo_agree(beta: float) -> None:
    model = Cheyette(DISC, 0.05, 0.01, beta=beta)
    value = bermudan_pde(CheyettePDE(model, 161, 41, 200), BER)
    _bracket(cheyette_exercise_problem(model, BER), value)
    if beta == 0.0:
        hw = model.to_hull_white()
        assert value == pytest.approx(bermudan_pde(HullWhitePDE(hw, 401, 400), BER), rel=1e-3)
