"""Robust forward-start bounds and the superhedge as vanillas plus forwards."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market import black_price
from spxa.vol.mot import mot_solve
from spxa.vol.robust import (
    forward_start_call,
    forward_start_straddle,
    robust_bounds,
    static_replication,
)
from tests.unit.vol.test_mot import MU, NU


class Flat:
    """A flat 20% surface."""

    def total_variance(self, k: np.ndarray, T: float) -> np.ndarray:
        return np.full(np.shape(k), 0.04 * T)


GRID = np.linspace(0.2, 3.0, 57)


def test_static_replication_reproduces_and_prices() -> None:
    f = np.abs(np.sin(4 * GRID)) + GRID**2
    rep = static_replication(GRID, f)
    np.testing.assert_allclose(rep(GRID), f, atol=1e-12)
    mid = 0.5 * (GRID[1:] + GRID[:-1])
    np.testing.assert_allclose(rep(mid), 0.5 * (f[1:] + f[:-1]), atol=1e-12)
    assert rep.price(NU) == pytest.approx(NU.weights @ f, abs=1e-12)
    with pytest.raises(ValueError):
        static_replication(GRID[::-1], f)


def test_superhedge_costs_the_upper_bound_in_vanillas() -> None:
    r = mot_solve(MU, NU, forward_start_straddle(), "max")
    phi, psi = static_replication(MU.points, r.phi), static_replication(NU.points, r.psi)
    # T1 and T2 vanillas priced off the two smiles; the forwards h(X)(Y - X) cost nothing
    assert phi.price(MU) + psi.price(NU) == pytest.approx(r.value, abs=1e-12)
    assert phi.calls.size == psi.calls.size == GRID.size - 2


def test_robust_bounds_flat_surface() -> None:
    for k in (0.9, 1.0, 1.1):
        lo, hi, sub, sup = robust_bounds(Flat(), 0.5, 1.0, forward_start_call(k), GRID, scale=100.0)
        bs = 100.0 * float(black_price(1.0, k, 0.5, 0.2))  # Black-Scholes forward-start call
        assert lo < bs < hi and sub.sense == "min" and sup.sense == "max"
    # the forward log contract is model independent: E log Y - E log X = -sigma^2 (T2 - T1) / 2 (up to the grid)
    lo, hi, sub, _ = robust_bounds(Flat(), 0.5, 1.0, lambda x, y: np.log(y) - np.log(x), GRID)
    assert hi - lo < 1e-9
    exact = sub.nu.expect(np.log) - sub.mu.expect(np.log)
    assert lo == pytest.approx(exact, abs=1e-9) and exact == pytest.approx(-0.01, abs=2e-4)
