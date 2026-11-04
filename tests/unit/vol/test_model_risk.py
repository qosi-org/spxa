"""Cont's model-risk measures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.market import black_price
from spxa.vol.model_risk import ModelRisk

VOLS = {"bs18": 0.18, "bs20": 0.20, "bs22": 0.22, "bs25": 0.25}


def _bs_family() -> ModelRisk:
    # claims: 1y ATM call, 110 call, forward contract (model independent)
    prices = {
        n: [black_price(100.0, 100.0, 1.0, s, 1.0), black_price(100.0, 110.0, 1.0, s, 1.0), 0.0]
        for n, s in VOLS.items()
    }
    errors = {n: abs(s - 0.2) for n, s in VOLS.items()}
    return ModelRisk.from_prices(prices, errors)


def test_range_and_tolerance() -> None:
    mr = _bs_family()
    atm = mr.combine([1.0, 0.0, 0.0])
    lo, hi = black_price(100.0, 100.0, 1.0, 0.18, 1.0), black_price(100.0, 100.0, 1.0, 0.25, 1.0)
    assert (atm.lower(), atm.upper()) == pytest.approx((lo, hi), abs=1e-14)
    assert atm.width() == pytest.approx(hi - lo, abs=1e-14)
    assert atm.admissible(0.02 + 1e-12) == ("bs18", "bs20", "bs22")
    assert atm.width(0.0) == 0.0  # only the exact model is left
    assert atm.relative_width() == pytest.approx((hi - lo) / (0.5 * (hi + lo)), rel=1e-14)
    with pytest.raises(ValueError):
        atm.width(-1.0)


def test_coherence() -> None:
    mr = _bs_family()
    np.testing.assert_allclose(mr.width()[2], 0.0, atol=0)  # the forward is model independent
    # adding calibration instruments priced alike by every model leaves the risk unchanged
    assert mr.combine([1.0, 0.0, 7.0]).width() == pytest.approx(
        mr.combine([1.0, 0.0, 0.0]).width(), abs=1e-13
    )
    rng = np.random.default_rng(1)
    rand = ModelRisk(tuple("abcdef"), rng.normal(size=(6, 2)), np.zeros(6), np.zeros((6, 2)))
    for _ in range(200):
        w1, w2, c = rng.normal(size=2), rng.normal(size=2), rng.uniform(0, 5)
        a, b = rand.combine(w1).width(), rand.combine(w2).width()
        assert rand.combine(w1 + w2).width() <= a + b + 1e-12  # subadditive
        assert rand.combine(c * w1).width() == pytest.approx(c * a, abs=1e-12)  # homogeneous


def test_penalised_interval() -> None:
    atm = _bs_family().combine([1.0, 0.0, 0.0])
    lo0, hi0 = atm.penalised(0.0)
    assert (lo0, hi0) == (atm.lower(), atm.upper())
    widths = [atm.penalised_width(lam) for lam in np.linspace(0.0, 200.0, 41)]
    assert all(b <= a + 1e-15 for a, b in zip(widths, widths[1:]))
    # a large penalty leaves the exact model's price at both ends
    exact = black_price(100.0, 100.0, 1.0, 0.2, 1.0)
    assert atm.penalised(1e6) == pytest.approx((exact, exact), abs=1e-12)
    # the ATM vega at 20% is 100 phi(0.1) ~ 39.7 and falls with vol, so a penalty above it
    # drops the 22% and 25% models from the upper end
    vega = 100.0 * math.exp(-0.5 * 0.1**2) / math.sqrt(2 * math.pi)
    assert atm.penalised(vega * 1.01)[1] == pytest.approx(exact, abs=1e-12)
    with pytest.raises(ValueError):
        atm.penalised(-1.0)


def test_validation() -> None:
    with pytest.raises(ValueError):
        ModelRisk.from_prices({})
    with pytest.raises(ValueError):
        ModelRisk.from_prices({"a": 1.0}, {"a": -0.1})
    with pytest.raises(ValueError):
        ModelRisk.from_prices({"a": 1.0, "b": 2.0}).combine([1.0])
