"""Unit tests for the risk-instrument adapter."""

import numpy as np
import pytest

from spxa.pde.instruments import PDEQuote, pde_price, pde_pricing_fn, supported_instruments
from spxa.pde.reference import barrier_price, bs_price, margrabe_price
from spxa.risk.instruments import (
    AsianOption,
    BarrierOption,
    BasketOption,
    EuropeanOption,
    InterestRateSwap,
)
from spxa.risk.portfolio import Portfolio
from spxa.risk.scenarios import MarketScenario, StressTestingEngine

MARKET = {"SPX": 100.0, "SPX_VOL": 0.2, "SPX_DIV": 0.01, "r": 0.03, "SX5E": 95.0, "SX5E_VOL": 0.25}


def test_supported():
    assert "EuropeanOption" in supported_instruments()


def test_european_with_notional_and_greeks():
    inst = EuropeanOption("c1", "SPX", 105.0, 0.5, is_call=True, notional=10.0)
    q = pde_price(inst, MARKET, vega=True)
    assert isinstance(q, PDEQuote) and q.instrument_id == "c1" and float(q) == q.price
    ref = bs_price(100.0, 105.0, 0.5, 0.03, 0.2, 0.01)
    assert q.price == pytest.approx(10 * ref, abs=2e-3)
    h = 1e-2
    assert q.delta["SPX"] == pytest.approx(
        10
        * (
            bs_price(100 + h, 105.0, 0.5, 0.03, 0.2, 0.01)
            - bs_price(100 - h, 105.0, 0.5, 0.03, 0.2, 0.01)
        )
        / (2 * h),
        abs=5e-3,
    )
    v = (
        10
        * (
            bs_price(100.0, 105.0, 0.5, 0.03, 0.201, 0.01)
            - bs_price(100.0, 105.0, 0.5, 0.03, 0.199, 0.01)
        )
        / 0.002
    )
    assert q.vega["SPX"] == pytest.approx(v, rel=2e-3)
    assert q.gamma["SPX"] > 0


@pytest.mark.parametrize("kind", ["down-and-out", "down-and-in", "up-and-out"])
def test_barrier(kind):
    H = 90.0 if kind.startswith("down") else 120.0
    inst = BarrierOption("b", "SPX", 100.0, H, 1.0, is_call=True, barrier_type=kind, rebate=1.5)
    q = pde_price(inst, MARKET)
    ref = barrier_price(100.0, 100.0, H, 1.0, 0.03, 0.2, 0.01, "call", kind, 1.5, "expiry")
    assert q.price == pytest.approx(ref, abs=2e-3)


def test_asian_linetsky():
    inst = AsianOption("a", "X", 2.0, 1.0)
    q = pde_price(inst, {"X": 2.0, "X_VOL": 0.5, "r": 0.05}, n_space=801, n_time=400)
    assert q.price == pytest.approx(0.2464156905, abs=1e-4)


def test_basket_margrabe():
    inst = BasketOption("x", ["SPX", "SX5E"], [1.0, -1.0], 0.0, 1.0, notional=2.0)
    mkt = dict(MARKET, SPX_SX5E_CORR=0.5)
    q = pde_price(inst, mkt, n1=201, n2=201, n_time=100)
    ref = margrabe_price(100.0, 95.0, 1.0, 0.2, 0.25, 0.5, 0.01, 0.0)
    assert q.price == pytest.approx(2 * ref, abs=3e-3)
    assert set(q.delta) == {"SPX", "SX5E"} and q.delta["SX5E"] < 0 < q.delta["SPX"]


def test_errors():
    with pytest.raises(KeyError, match="SPX_VOL"):
        pde_price(EuropeanOption("c", "SPX", 100.0, 1.0), {"SPX": 100.0, "r": 0.0})
    with pytest.raises(KeyError, match="CORR"):
        pde_price(BasketOption("x", ["SPX", "SX5E"], [1, 1], 100.0, 1.0), MARKET)
    with pytest.raises(NotImplementedError, match="two underlyings"):
        pde_price(BasketOption("x", ["A", "B", "C"], [1, 1, 1], 100.0, 1.0), MARKET)
    swap = InterestRateSwap.__new__(InterestRateSwap)
    swap.underlying_ids, swap.maturity = ["r"], 1.0
    with pytest.raises(NotImplementedError, match="no PDE pricer"):
        pde_price(swap, MARKET)


def test_pricing_fn_and_stress_engine():
    c = EuropeanOption("c", "SPX", 100.0, 1.0)
    p = EuropeanOption("p", "SPX", 95.0, 1.0, is_call=False)
    book = Portfolio([(c, 2.0), (p, -1.0)])
    fn = pde_pricing_fn(book, n_space=201, n_time=100)
    direct = (
        2 * pde_price(c, MARKET, n_space=201, n_time=100).price
        - pde_price(p, MARKET, n_space=201, n_time=100).price
    )
    assert fn(MARKET) == pytest.approx(direct, abs=1e-12)
    assert pde_pricing_fn(c)(MARKET) == pytest.approx(pde_price(c, MARKET).price)
    assert pde_pricing_fn([(c, 3.0)])(MARKET) == pytest.approx(3 * pde_price(c, MARKET).price)
    crash = MarketScenario("crash", {"SPX": -0.3}, {"SPX_VOL": 0.2}, rate_shifts_bps=-100)
    out = StressTestingEngine(fn).run_historical_suite(MARKET, {"crash": crash})["crash"]
    assert out["base_pv"] == pytest.approx(direct)
    assert out["stressed_pv"] == pytest.approx(fn(crash.apply(MARKET)))
    assert out["pnl_impact"] < 0.0  # long calls, short puts lose in a crash


def test_consistent_with_instrument_payoff():
    # the risk instrument's own path payoff, averaged over exact GBM paths
    from spxa.pde.lsm import gbm_paths

    inst = EuropeanOption("p", "SPX", 102.0, 0.75, is_call=False, notional=3.0)
    t, paths = gbm_paths(100.0, 0.03, 0.2, 0.75, 1, 1_000_000, q=0.01, seed=17)
    disc = np.exp(-0.03 * 0.75) * inst.payoff(paths, t)
    se = disc.std() / np.sqrt(disc.size)
    assert abs(pde_price(inst, MARKET).price - disc.mean()) < 3 * se
