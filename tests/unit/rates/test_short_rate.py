"""Common short-rate model interface."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.capfloor import cap_floor
from spxa.rates.cir_pp import CIRPlusPlus
from spxa.rates.g2pp import G2PP
from spxa.rates.hull_white import HullWhite
from spxa.rates.indices import EURIBOR6M
from spxa.rates.hw_calibration import model_price
from spxa.rates.short_rate import ShortRateModel, model_implied_vols
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_capfloor import DISC, PROJ, REF

HW = HullWhite(DISC, 0.05, 0.01)
MODELS = [HW, G2PP(DISC, 0.5, 0.01, 0.05, 0.008, -0.6), CIRPlusPlus(DISC, 0.3, 0.025, 0.08, 0.02)]
INSTR = [
    cap_floor(REF, "5Y", 0.035, EURIBOR6M, notional=1e4),
    swaption(REF, "2Y", "5Y", 0.034, notional=1e4),
    swaption(REF, "5Y", "10Y", 0.036, notional=1e4, payer=False),
]


@pytest.mark.parametrize("model", MODELS, ids=["hw", "g2pp", "cirpp"])
def test_models_share_the_protocol_and_vols_round_trip(model: ShortRateModel) -> None:
    assert isinstance(model, ShortRateModel)
    for quote in ("black", "normal"):
        vols = model_implied_vols(model, INSTR, PROJ, quote)
        assert np.all(vols > 0)
        repriced = [i.price(DISC, PROJ, v, quote) for i, v in zip(INSTR, vols)]
        prices = [model_price(model, i, PROJ) for i in INSTR]
        assert repriced == pytest.approx(prices, rel=1e-10)


def test_two_factor_limit_quotes_like_hull_white() -> None:
    g = G2PP(DISC, 0.05, 0.01, 0.6, 1e-9, 0.0)
    assert model_implied_vols(g, INSTR, PROJ) == pytest.approx(
        model_implied_vols(HW, INSTR, PROJ), rel=1e-7
    )


def test_hull_white_normal_vols_scale_with_sigma() -> None:
    # Gaussian model: normal implied vols are nearly proportional to sigma
    v1 = model_implied_vols(HW, INSTR[1:2], PROJ, "normal")[0]
    v2 = model_implied_vols(HW.with_sigma(0.02), INSTR[1:2], PROJ, "normal")[0]
    assert v2 / v1 == pytest.approx(2.0, rel=2e-3)


def test_validation() -> None:
    with pytest.raises(TypeError):
        model_price(HW, object())
    with pytest.raises(ValueError):
        model_implied_vols(HW, INSTR, PROJ, "lognormal")
