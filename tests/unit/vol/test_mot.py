"""Martingale optimal transport by linear programming."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from spxa.market import black_price
from spxa.vol.marginals import DiscreteMarginal, marginal_from_calls
from spxa.vol.mot import mot_bounds, mot_solve, ot_solve

K = np.linspace(0.2, 3.0, 57)
C1 = np.asarray(black_price(1.0, K, 0.5, 0.2), float)
C2 = np.asarray(black_price(1.0, K, 1.0, 0.2), float)
MU = marginal_from_calls(K, C1)
NU = marginal_from_calls(K, C2, floor=C1)


def straddle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(y - x)


def test_strong_duality_and_hedge_inequalities() -> None:
    X, Y = np.meshgrid(MU.points, NU.points, indexing="ij")
    for sense in ("max", "min"):
        r = mot_solve(MU, NU, straddle, sense)
        assert r.dual_value == pytest.approx(r.value, abs=1e-12)
        gap = r.phi[:, None] + r.psi[None, :] + r.h[:, None] * (Y - X) - np.abs(Y - X)
        assert (gap.min() > -1e-10) if sense == "max" else (gap.max() < 1e-10)
        # the plan is a martingale coupling of the two marginals
        assert r.plan.min() > -1e-9
        np.testing.assert_allclose(r.plan.sum(1), MU.weights, atol=1e-12)
        np.testing.assert_allclose(r.plan.sum(0), NU.weights, atol=1e-12)
        assert np.abs((r.plan * (Y - X)).sum(1)).max() < 1e-12
        assert r.hedge(X[3, 5], Y[3, 5]) == pytest.approx(
            gap[3, 5] + abs(Y[3, 5] - X[3, 5]), abs=1e-14
        )


def test_bounds_contain_black_scholes_and_sit_inside_classical_ot() -> None:
    lo, hi = mot_bounds(MU, NU, straddle)
    bs = 2 * (2 * norm.cdf(0.2 * np.sqrt(0.5) / 2) - 1)  # E|Y - X| with flat 20% vol
    assert lo < bs < hi
    assert (
        ot_solve(MU, NU, straddle, "min").value < lo
        and hi < ot_solve(MU, NU, straddle, "max").value
    )
    assert (round(lo, 4), round(hi, 4)) == (0.0507, 0.1389)


def test_model_independent_claims_have_zero_width() -> None:
    # phi(x) + psi(y) + g(x)(y - x) has the same price under every martingale coupling
    def claim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.maximum(x - 1.0, 0.0) + np.maximum(1.1 - y, 0.0) + np.sin(3 * x) * (y - x)

    exact = MU.call(1.0) + (NU.call(1.1) - (1.0 - 1.1))
    lo, hi = mot_bounds(MU, NU, claim)
    assert lo == pytest.approx(exact, abs=1e-9) and hi == pytest.approx(exact, abs=1e-9)
    var = NU.expect(np.square) - MU.expect(np.square)  # E(Y - X)^2 = E Y^2 - E X^2
    lo, hi = mot_bounds(MU, NU, lambda x, y: (y - x) ** 2)
    assert lo == pytest.approx(var, abs=1e-9) and hi == pytest.approx(var, abs=1e-9)


def test_degenerate_first_marginal_and_validation() -> None:
    dirac = DiscreteMarginal(np.array([1.0]), np.array([1.0]))
    lo, hi = mot_bounds(dirac, NU, straddle)
    assert lo == pytest.approx(hi, abs=1e-12) and hi == pytest.approx(
        NU.expect(lambda y: np.abs(y - 1)), abs=1e-12
    )
    with pytest.raises(ValueError):
        mot_solve(NU, MU, straddle)  # wrong convex order
    with pytest.raises(ValueError):
        mot_solve(MU, NU, straddle, "sup")
