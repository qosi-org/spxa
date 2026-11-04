"""Hull-White calibration."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.capfloor import cap_floor
from spxa.rates.hull_white import HullWhite
from spxa.rates.hw_calibration import calibrate_hull_white, market_prices, model_price
from spxa.rates.indices import EURIBOR6M
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_capfloor import PROJ
from tests.unit.rates.test_hull_white import CURVE

REF = "2024-03-15"
# co-terminal swaptions into a swap ending in 10 years
COTERM = [swaption(REF, f"{n}Y", f"{10 - n}Y", 0.035, notional=1e4) for n in (1, 3, 5, 7)]


def test_bootstrap_matches_coterminal_swaptions_exactly() -> None:
    T = [float(CURVE.time(s.expiry)) for s in COTERM]
    truth = HullWhite(CURVE, 0.04, [0.011, 0.009, 0.0085, 0.007], T[:-1])
    tgt = [model_price(truth, s, PROJ) for s in COTERM]
    cal = calibrate_hull_white(CURVE, COTERM[::-1], tgt[::-1], PROJ, a=0.04)
    assert cal.max_relative_error < 1e-11
    assert cal.model.sigma == pytest.approx(truth.sigma, rel=1e-9)
    assert cal.model.sigma_times == pytest.approx(truth.sigma_times, abs=0)


def test_global_fit_recovers_mean_reversion() -> None:
    insts = [swaption(REF, e, t, 0.034, notional=1e4) for e in ("1Y", "5Y") for t in ("2Y", "10Y")]
    insts.append(cap_floor(REF, "5Y", 0.036, EURIBOR6M, notional=1e4))
    truth = HullWhite(CURVE, 0.06, 0.0105)
    tgt = [model_price(truth, i, PROJ) for i in insts]
    cal = calibrate_hull_white(CURVE, insts, tgt, PROJ, a=None, mode="global")
    assert cal.model.a == pytest.approx(0.06, rel=1e-6)
    assert cal.model.sigma[0] == pytest.approx(0.0105, rel=1e-7)


def test_market_vols_and_validation() -> None:
    vols = [0.30, 0.28, 0.26, 0.25]
    tgt = market_prices(COTERM, vols, CURVE, PROJ)
    cal = calibrate_hull_white(CURVE, COTERM, tgt, PROJ, a=0.03)
    assert cal.max_relative_error < 1e-10
    assert np.all(cal.model.sigma > 0)
    with pytest.raises(ValueError):
        calibrate_hull_white(CURVE, COTERM, tgt, PROJ, a=None)
    with pytest.raises(ValueError):
        calibrate_hull_white(CURVE, COTERM, tgt[:2], PROJ)
    with pytest.raises(ValueError):
        calibrate_hull_white(CURVE, COTERM[:1], [1e4], PROJ)  # above the notional: unattainable
