"""G2++ calibration."""

from __future__ import annotations

import pytest

from spxa.rates.g2pp import G2PP
from spxa.rates.g2pp_calibration import calibrate_g2pp
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_capfloor import PROJ
from tests.unit.rates.test_hull_white import CURVE

REF = "2024-03-15"
GRID = [
    swaption(REF, e, t, 0.034, notional=1e4)
    for e in ("1Y", "3Y", "7Y")
    for t in ("1Y", "5Y", "15Y")
]


def test_recovers_parameters_from_a_swaption_grid() -> None:
    truth = G2PP(CURVE, 0.6, 0.011, 0.04, 0.007, -0.6)
    tgt = [truth.swaption(s, PROJ) for s in GRID]
    cal = calibrate_g2pp(CURVE, GRID, tgt, PROJ)
    assert cal.max_relative_error < 1e-9
    m = cal.model
    assert (m.a, m.sigma, m.b, m.eta, m.rho) == pytest.approx(
        (0.6, 0.011, 0.04, 0.007, -0.6), rel=1e-5
    )


def test_validation() -> None:
    with pytest.raises(ValueError):
        calibrate_g2pp(CURVE, GRID[:3], [1.0] * 3, PROJ)
    with pytest.raises(ValueError):
        calibrate_g2pp(CURVE, GRID, [1.0] * 9, PROJ, initial=(0.05, 0.01, 0.1, 0.01, 0.0))
    with pytest.raises(TypeError):
        calibrate_g2pp(CURVE, [object()] * 5, [1.0] * 5, PROJ)
