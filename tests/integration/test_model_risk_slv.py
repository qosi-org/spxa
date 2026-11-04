"""Model risk across stochastic-local-vol models fitted to one surface, inside the MOT bounds."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

import spxa.vol as v
from spxa.market import EquityForward
from spxa.pde import BlackScholesPDE

SURFACE = v.VolSurface(
    v.SSVI(-0.7, v.PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t),
    EquityForward(100.0, 0.03, dividend_yield=0.01),
)
HESTON = v.Heston(0.04, 1.5, 0.04, 0.5, -0.7)
ETAS = (0.0, 0.5, 1.0)
STRIKES = (0.9, 1.0, 1.1)
K_SMILE = np.array([-0.3, -0.15, 0.0, 0.1, 0.2])


@lru_cache(maxsize=1)
def _family() -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    lv = v.as_local_vol(SURFACE, 1.0)
    samples, errors = {}, {}
    for eta in ETAS:
        m = v.calibrate_slv(lv, HESTON, eta, 1.0, n_time=50, record=(0.5, 1.0))
        model = np.concatenate([m.smile(K_SMILE, T) for T in (0.5, 1.0)])
        market = np.concatenate([SURFACE.implied_vol_k(K_SMILE, T) for T in (0.5, 1.0)])
        errors[f"eta={eta}"] = v.smile_rmse(model, market)
        p = m.paths(1.0, 100, 50_000, record=(0.5, 1.0), seed=7)
        samples[f"eta={eta}"] = (np.exp(p.at(0.5)[0]), np.exp(p.at(1.0)[0]))
    return samples, errors


def _forward_starts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack([np.maximum(y - k * x, 0.0) for k in STRIKES])


def test_family_fits_the_surface_and_agrees_on_vanillas() -> None:
    samples, errors = _family()
    assert max(errors.values()) < 15e-4  # every model within 15 bp of the smile
    vanilla = v.ModelRisk.from_samples(samples, lambda x, y: np.maximum(y - 1.0, 0.0), errors)
    assert vanilla.width() < 1e-3  # vanillas: calibration noise only
    fwd = v.ModelRisk.from_samples(samples, _forward_starts, errors)
    # at and above the money the forward smiles differ well beyond the vanilla noise
    assert np.all(fwd.width()[1:] > 8 * vanilla.width()) and np.all(fwd.significant()[1:])
    # local vol (eta = 0) prices them highest and the full Heston mix lowest, by ~7% and ~20%
    order = [fwd.names.index(f"eta={e}") for e in ETAS]
    assert np.all(np.diff(fwd.prices[order, 1:], axis=0) < 0)
    assert [round(float(r), 2) for r in fwd.relative_width()[1:]] == [0.07, 0.21]


def test_family_lies_inside_the_martingale_transport_bounds() -> None:
    samples, errors = _family()
    fwd = v.ModelRisk.from_samples(samples, _forward_starts, errors)
    grid = np.linspace(0.02, 3.02, 101)
    bounds = [
        v.robust_bounds(SURFACE, 0.5, 1.0, v.forward_start_call(k), grid)[:2] for k in STRIKES
    ]
    lo, hi = np.array(bounds).T
    assert fwd.within(lo, hi)
    share = fwd.share(lo, hi)
    assert np.all(share < 0.15)  # one-factor-in-spot families explore a sliver of the interval
    assert [round(float(s), 2) for s in share] == [0.02, 0.08, 0.11]


def test_barrier_model_risk_by_the_slv_pde() -> None:
    r, q = 0.03, 0.01
    lv = v.as_local_vol(SURFACE, 1.0)
    prices, errors = {}, {}
    for eta in ETAS:
        m = v.calibrate_slv(lv, HESTON, eta, 1.0, n_time=50, record=(1.0,))
        pde = m.pde(r, q, n_x=81, n_v=31, n_time=40)
        prices[f"eta={eta}"] = [
            pde.european(100.0, 1.0, "call", spot=100.0),
            pde.barrier(100.0, 1.0, "call", 100.0, lower=85.0),
            pde.barrier(100.0, 1.0, "call", 100.0, upper=130.0),
        ]
        errors[f"eta={eta}"] = v.smile_rmse(
            m.smile(K_SMILE, 1.0), SURFACE.implied_vol_k(K_SMILE, 1.0)
        )
    mr = v.ModelRisk.from_prices(prices, errors)
    vanilla, down, up = mr.width()
    assert vanilla < 0.03  # 0.3% of the 9.32 call: grids and calibration only
    assert up > 15 * vanilla and down > 3 * vanilla
    # the up-and-out rises steadily with the mixing fraction; the down-and-out barely moves
    assert np.all(np.diff(mr.prices[:, 2]) > 0.15)
    assert [round(float(x), 2) for x in mr.relative_width()[1:]] == [0.01, 0.09]
    # eta = 0 is local volatility: the 2D SLV PDE agrees with the 1D Dupire PDE
    bs = BlackScholesPDE(r, lv, q)
    lv_down = float(bs.barrier(100.0, 1.0, 100.0, 85.0, "down-and-out", "call"))
    lv_up = float(bs.barrier(100.0, 1.0, 100.0, 130.0, "up-and-out", "call"))
    assert abs(prices["eta=0.0"][1] - lv_down) < 0.02 and abs(prices["eta=0.0"][2] - lv_up) < 0.02
    # charging calibration error shrinks the up-and-out interval monotonically
    widths = [float(mr.penalised_width(lam)[2]) for lam in (0.0, 300.0, 1000.0, 3000.0)]
    assert all(b < a for a, b in zip(widths, widths[1:]))
