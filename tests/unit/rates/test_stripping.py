"""Caplet volatility stripping."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.capfloor import cap_floor
from spxa.rates.indices import EURIBOR6M
from spxa.rates.stripping import CapletVolCurve, strip_caplet_vols
from tests.unit.rates.test_capfloor import DISC, PROJ, REF

TENORS = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]


def _caps(strike: float) -> list:
    return [cap_floor(REF, t, strike, EURIBOR6M, notional=1e6) for t in TENORS]


def test_round_trip_of_flat_quotes() -> None:
    caps = _caps(0.035)
    flat = [0.30, 0.28, 0.26, 0.24, 0.23, 0.22]
    curve = strip_caplet_vols(caps, flat, DISC, PROJ)
    assert curve.vols.size == 6
    for cap, fv in zip(caps, flat):
        assert cap.price(DISC, PROJ, curve) == pytest.approx(cap.price(DISC, PROJ, fv), rel=1e-11)
    # a falling flat curve strips to a faster-falling caplet curve
    assert np.all(np.diff(curve.vols) < 0)
    assert curve.vols[-1] < flat[-1]


def test_recovers_a_known_caplet_curve() -> None:
    caps = _caps(0.034)
    true = CapletVolCurve(
        np.array([c.caplets[-1].expiry(DISC) for c in caps]),
        np.array([0.25, 0.3, 0.27, 0.22, 0.2, 0.19]),
    )
    flat = [cap.implied_flat_vol(cap.price(DISC, PROJ, true), DISC, PROJ) for cap in caps]
    got = strip_caplet_vols(caps, flat, DISC, PROJ)
    assert got.vols == pytest.approx(true.vols, rel=1e-9)


def test_flat_and_normal_quotes() -> None:
    caps = _caps(0.03)
    assert strip_caplet_vols(caps, [0.25] * 6, DISC, PROJ).vols == pytest.approx(
        np.full(6, 0.25), rel=1e-10
    )
    flat_n = [0.009, 0.0095, 0.0098, 0.0097, 0.0094, 0.009]
    curve = strip_caplet_vols(caps, flat_n, DISC, PROJ, model="normal")
    for cap, fv in zip(caps, flat_n):
        assert cap.price(DISC, PROJ, curve, "normal") == pytest.approx(
            cap.price(DISC, PROJ, fv, "normal"), rel=1e-11
        )
    with pytest.raises(ValueError):
        strip_caplet_vols(caps[::-1], [0.25] * 6, DISC, PROJ)
    with pytest.raises(ValueError):
        # an impossibly steep drop needs negative caplet variance
        strip_caplet_vols(caps[:2], [0.6, 0.05], DISC, PROJ)
