"""The MOT superhedge of a forward-start call, run along Heston paths."""

from __future__ import annotations

import numpy as np

import spxa.vol as v
from spxa.vol.marginals import marginal_from_calls
from spxa.vol.mot import mot_solve
from spxa.vol.robust import forward_start_call, superhedge_pnl

HESTON = v.Heston(0.04, 1.5, 0.04, 0.5, -0.7)
GRID = np.linspace(0.05, 3.05, 121)


def test_superhedge_and_subhedge_along_heston_paths() -> None:
    c1 = np.asarray(HESTON.price(GRID, 0.5), float)
    c2 = np.asarray(HESTON.price(GRID, 1.0), float)
    mu = marginal_from_calls(GRID, c1)
    nu = marginal_from_calls(GRID, c2, floor=mu.call(GRID))
    pay = forward_start_call(1.0)
    sup, sub = mot_solve(mu, nu, pay, "max"), mot_solve(mu, nu, pay, "min")
    paths = v.slv_paths(
        HESTON, lambda y, t: np.ones_like(y), 1.0, 200, 50_000, record=(0.5, 1.0), seed=3
    )
    X, Y = np.exp(paths.at(0.5)[0]), np.exp(paths.at(1.0)[0])
    assert GRID[0] < Y.min() and Y.max() < GRID[-1]  # the grid covers every path
    f = pay(X, Y)
    model, se = f.mean(), f.std() / np.sqrt(f.size)
    assert sub.value + 3 * se < model < sup.value - 3 * se
    h = GRID[1] - GRID[0]
    for r in (sup, sub):
        pnl = superhedge_pnl(r, pay, X, Y)
        # exact on the grid; between nodes the linear interpolation can dip by a fraction of the spacing
        assert pnl.min() > -0.05 * h
        # on average the hedge earns the gap between the bound and the model price
        gap = r.value - model if r.sense == "max" else model - r.value
        assert abs(pnl.mean() - gap) < 4 * pnl.std() / np.sqrt(pnl.size) + 1e-4
