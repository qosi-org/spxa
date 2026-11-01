"""__story__ summaries of the PDE result objects."""

import numpy as np

from spxa.pde import (
    FokkerPlanck1D,
    HestonPDE,
    TwoAssetBlackScholesPDE,
    lsm_american,
    pde_price,
)
from spxa.pde.grid import Grid1D
from spxa.risk.instruments import EuropeanOption


def test_heston_story():
    res = HestonPDE(0.02, 2.0, 0.04, 0.3, -0.5).american(
        100.0, 0.5, 100.0, 0.04, "put", n_s=31, n_v=16, n_time=10
    )
    text = res.__story__()
    assert "Heston ADI" in text and "price =" in text and "Ikonen-Toivanen" in text
    assert "31 x 16 nodes" in text and "hundsdorfer_verwer" in text and "Natural2D" in text


def test_two_asset_story():
    res = TwoAssetBlackScholesPDE(0.03, 0.2, 0.3, 0.4).exchange(
        1.0, 100.0, 100.0, n1=21, n2=21, n_time=10
    )
    text = res.__story__()
    assert "S1 = 100" in text and "cross gamma" in text and "Linear2D" in text


def test_lsm_story():
    text = lsm_american(40.0, 1.0, 40.0, 0.06, 0.2, n_paths=2000, n_dates=5).__story__()
    assert "two-pass" in text and "95% CI" in text and "5 exercise dates" in text


def test_density_story():
    fp = FokkerPlanck1D(
        drift=lambda x, t: -x,
        volatility=0.5,
        grid=Grid1D.uniform(-4.0, 4.0, 81),
        time_homogeneous=True,
    )
    text = fp.solve(0.0, 1.0, n_time=20).__story__()
    assert "mass = " in text and "81 nodes" in text


def test_quote_story():
    q = pde_price(
        EuropeanOption("c1", "SPX", 100.0, 1.0),
        {"SPX": 100.0, "SPX_VOL": 0.2, "r": 0.01},
        vega=True,
    )
    text = q.__story__()
    assert text.startswith("c1:") and "SPX: delta" in text and "vega" in text
    assert np.isfinite(q.price)
