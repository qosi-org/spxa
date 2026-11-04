"""spxa.risk options valued on implied-volatility surfaces."""

from __future__ import annotations

import math

import pytest

from spxa.market.black import black_price
from spxa.market.curves import EquityForward
from spxa.risk.instruments import BarrierOption, EuropeanOption
from spxa.risk.scenarios import StressTestingEngine
from spxa.vol import SSVI, PowerLawPhi, SurfaceQuote, surface_price, surface_pricing_fn
from spxa.vol.surface import VolSurface

FWD = EquityForward(100.0, 0.03, dividend_yield=0.01)
SURF = VolSurface(SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t), FWD)
BASE = {"SPX": 100.0, "SPX_SURFACE": SURF, "SPX_VOL": 0.2, "r": 0.03}


def put(K: float = 90.0, T: float = 1.0, n: float = 10.0) -> EuropeanOption:
    return EuropeanOption("P", "SPX", K, T, is_call=False, notional=n)


def test_unshifted_price_is_the_surface_price() -> None:
    for opt in (put(), EuropeanOption("C", "SPX", 110.0, 0.5, notional=3.0)):
        q = surface_price(opt, BASE)
        option = "call" if opt.is_call else "put"
        assert isinstance(q, SurfaceQuote)
        assert q.price == pytest.approx(
            opt.notional * float(SURF.price(opt.strike, opt.maturity, option)), abs=1e-10
        )
        assert q.vol == pytest.approx(float(SURF.implied_vol(opt.strike, opt.maturity)), abs=1e-14)


@pytest.mark.parametrize("dynamics", ["sticky_strike", "sticky_moneyness"])
def test_spot_delta_matches_surface_greeks(dynamics: str) -> None:
    q = surface_price(put(n=1.0), BASE, dynamics=dynamics)
    g = SURF.greeks(90.0, 1.0, "put", dynamics=dynamics)
    assert q.delta == pytest.approx(float(g["delta"]) * q.forward / 100.0, rel=1e-6)
    assert q.vega == pytest.approx(float(g["vega"]), rel=1e-6)


def test_sticky_moneyness_put_delta_is_less_negative_on_a_negative_skew() -> None:
    # spot up -> log(K/F) down -> higher vol at K: the put loses less

    strike = surface_price(put(n=1.0), BASE).delta
    moneyness = surface_price(put(n=1.0), BASE, dynamics="sticky_moneyness").delta
    assert strike < moneyness < 0


def test_lehman_scenario() -> None:
    engine = StressTestingEngine(surface_pricing_fn([(put(), 1.0)], BASE))
    out = engine.run_historical_suite(BASE)["LEHMAN_2008"]
    T, dr = 1.0, -0.02
    F = float(SURF.forward(T)) * 0.55 * math.exp(dr * T)
    D = float(SURF.discount(T)) * math.exp(-dr * T)
    sig = float(SURF.implied_vol(90.0, T)) + 0.45
    assert out["stressed_pv"] == pytest.approx(
        10.0 * black_price(F, 90.0, T, sig, D, "put"), abs=1e-9
    )
    assert out["base_pv"] == pytest.approx(surface_price(put(), BASE).price, abs=1e-12)


def test_errors() -> None:
    barrier = BarrierOption("B", "SPX", 100.0, 80.0, 1.0)
    with pytest.raises(TypeError):
        surface_price(barrier, BASE)
    with pytest.raises(ValueError):
        surface_price(put(), BASE, dynamics="sticky_delta")
    with pytest.raises(KeyError):
        surface_price(put(), {"SPX": 100.0})
    no_spot = VolSurface.flat(0.2, SURF)
    with pytest.raises(ValueError):
        surface_price(put(), {"SPX_SURFACE": no_spot})
