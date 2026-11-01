"""Integration tests: spxa.pde across models, processes, risk and graphs."""

import numpy as np
import pytest

import spxa
from spxa.pde import (
    BlackScholesPDE,
    FokkerPlanck1D,
    Grid1D,
    HestonPDE,
    TwoAssetBlackScholesPDE,
    lsm_american,
    pde_pricing_fn,
    to_pde,
)
from spxa.pde.reference import bs_price, merton_price
from spxa.risk.instruments import EuropeanOption
from spxa.risk.scenarios import HISTORICAL_SCENARIOS, StressTestingEngine
from spxa.zoo.levy import MertonJumpDiffusion


def test_lazy_subpackage():
    assert "pde" in dir(spxa)
    assert spxa.pde.HestonPDE is HestonPDE
    assert all(hasattr(spxa.pde, n) for n in spxa.pde.__all__)
    assert spxa.pde.reference.bs_price is bs_price


def test_heston_degenerates_to_black_scholes():
    # vanishing vol-of-vol with v0 = theta: constant variance
    h = HestonPDE(0.03, 2.0, 0.04, 0.01, -0.5)
    got = h.european(100.0, 1.0, 100.0, 0.04, n_s=201, n_v=101, n_time=100).price
    assert got == pytest.approx(bs_price(100.0, 100.0, 1.0, 0.03, 0.2), abs=1e-3)


def test_forward_backward_duality():
    # E[e^{-rT} g(S_T)] from the Fokker-Planck density equals the backward PDE price
    r, s, T, K = 0.03, 0.2, 1.0, 100.0
    grid = Grid1D.uniform(np.log(100.0) - 2.0, np.log(100.0) + 2.0, 801)
    density = FokkerPlanck1D(drift=r - 0.5 * s * s, volatility=s, grid=grid).solve(
        np.log(100.0), T, n_time=200
    )
    forward = np.exp(-r * T) * density.expectation(lambda x: np.maximum(np.exp(x) - K, 0.0))
    backward = BlackScholesPDE(r, s).european(K, T, 100.0).price
    assert forward == pytest.approx(backward, abs=1.5e-3)
    assert density.mass() == pytest.approx(1.0, abs=1e-10)


def test_process_to_pide_merton():
    proc = MertonJumpDiffusion(drift=0.0, sigma=0.15, rate=0.1, jump_mean=-0.9, jump_std=0.45)
    pide = to_pde(proc, r=0.05)
    got = pide.european(100.0, 0.25, 100.0, "put").price
    ref = merton_price(100.0, 100.0, 0.25, 0.05, 0.15, 0.1, -0.9, 0.45, option="put")
    assert got == pytest.approx(ref, abs=2e-4)


def test_three_routes_to_an_american_put():
    bs = BlackScholesPDE(0.06, 0.2)
    dates = np.arange(1, 50) / 50
    pde = bs.bermudan(40.0, 1.0, 36.0, dates, "put", n_space=801, n_time=1000).price
    mc = lsm_american(40.0, 1.0, 36.0, 0.06, 0.2, n_dates=50, n_paths=200_000, seed=1)
    assert abs(mc.price - pde) < 3 * mc.std_error
    # the continuously exercisable value bounds the Bermudan one from above
    assert bs.american(40.0, 1.0, 36.0, "put").price > pde


def test_two_asset_reduces_to_one_asset():
    m = TwoAssetBlackScholesPDE(0.05, 0.3, 0.2, -0.4)
    res = m.price(
        lambda a, b: np.maximum(100.0 - b, 0.0),
        1.0,
        100.0,
        100.0,
        n1=101,
        n2=201,
        n_time=100,
        align=((), (100.0,)),
    )
    assert res.price == pytest.approx(
        bs_price(100.0, 100.0, 1.0, 0.05, 0.2, option="put"), abs=3e-3
    )


def test_stress_testing_with_pde_book():
    book = [
        (EuropeanOption("c", "SPX", 4000.0, 0.5), 1.0),
        (EuropeanOption("p", "SPX", 3600.0, 0.5, is_call=False), 2.0),
    ]
    market = {"SPX": 4000.0, "SPX_VOL": 0.18, "r": 0.04}
    engine = StressTestingEngine(pde_pricing_fn(book, n_space=201, n_time=100))
    out = engine.run_historical_suite(market)
    assert set(out) == set(HISTORICAL_SCENARIOS)
    for res in out.values():
        assert res["stressed_pv"] > res["base_pv"]  # long puts gain in every crash
