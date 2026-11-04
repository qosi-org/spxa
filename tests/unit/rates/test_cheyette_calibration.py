"""Cheyette calibration."""

from __future__ import annotations

import pytest

from spxa.rates.cheyette import Cheyette
from spxa.rates.cheyette_calibration import calibrate_cheyette
from spxa.rates.cheyette_swaption import cheyette_swaption
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_capfloor import DISC, PROJ, REF

GRID = [
    swaption(REF, e, t, k, notional=1e4, payer=k >= 0.034)
    for e, t in (("1Y", "9Y"), ("3Y", "7Y"), ("5Y", "5Y"))
    for k in (0.025, 0.034, 0.045)
]


def test_recovers_levels_and_skew() -> None:
    truth = Cheyette(
        DISC,
        0.03,
        [0.011, 0.009, 0.008],
        beta=0.25,
        sigma_times=[DISC.time(GRID[0].expiry), DISC.time(GRID[3].expiry)],
    )
    tgt = [cheyette_swaption(truth, s, PROJ, n_time=64) for s in GRID]
    cal = calibrate_cheyette(DISC, GRID, tgt, PROJ)
    assert cal.max_relative_error < 1e-9
    assert cal.model.beta == pytest.approx(0.25, rel=1e-6)
    assert list(cal.model.sigma) == pytest.approx([0.011, 0.009, 0.008], rel=1e-6)


def test_validation() -> None:
    with pytest.raises(ValueError):
        calibrate_cheyette(DISC, GRID, [1.0] * 3, PROJ)
    with pytest.raises(ValueError):
        calibrate_cheyette(DISC, GRID[1::3], [1.0] * 3, PROJ)
