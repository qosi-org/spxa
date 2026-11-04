"""Exact G2++ simulation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.rates.g2pp_mc import g2pp_paths, step_covariance
from spxa.rates.model_cashflows import swap_cashflows
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_capfloor import PROJ
from tests.unit.rates.test_g2pp import M
from tests.unit.rates.test_hull_white import CURVE

TIMES = [0.5, 1.0, 2.0, 4.0]


@pytest.fixture(scope="module")
def paths():  # type: ignore[no-untyped-def]
    return g2pp_paths(M, TIMES, 200_000, seed=11)


def test_step_covariance_by_quadrature() -> None:
    h = 0.7
    C = step_covariance(M, h)
    a, b, s, e, r = M.a, M.b, M.sigma, M.eta, M.rho
    # Cov(eta_x, eta_y) = r s e int_0^h B_a(v) B_b(v) dv
    Ba = lambda v: (1 - np.exp(-a * v)) / a  # noqa: E731
    Bb = lambda v: (1 - np.exp(-b * v)) / b  # noqa: E731
    assert C[2, 3] == pytest.approx(r * s * e * quad(lambda v: Ba(v) * Bb(v), 0, h)[0], rel=1e-12)
    # Cov(eps_x, eta_y) = r s e int_0^h e^{-a v} B_b(v) dv
    assert C[0, 3] == pytest.approx(
        r * s * e * quad(lambda v: np.exp(-a * v) * Bb(v), 0, h)[0], rel=1e-12
    )
    assert np.all(np.linalg.eigvalsh(C) > 0)
    assert C[:2, :2] == pytest.approx(M.covariance(h), rel=1e-14)


def test_state_covariance_and_curve(paths) -> None:  # type: ignore[no-untyped-def]
    for t in TIMES:
        i = paths._row(t)
        emp = np.cov(np.vstack([paths.x[i], paths.y[i]]))
        assert emp == pytest.approx(M.covariance(t), rel=0.02)
        m, se = paths.price(np.ones(paths.x.shape[1]), t)
        assert abs(m - CURVE.discount(t)) < 4 * se
        m, se = paths.price(paths.zero_bond(t, t + 5.0), t)
        assert abs(m - CURVE.discount(t + 5.0)) < 4 * se


def test_swaption_monte_carlo(paths) -> None:  # type: ignore[no-untyped-def]
    sw = swaption("2024-03-15", "2Y", "5Y", 0.035, notional=1e4)
    T = float(CURVE.time(sw.expiry))
    sim = g2pp_paths(M, [T], 200_000, seed=3)
    cf = swap_cashflows(sw.swap, CURVE, PROJ)
    v = sim.zero_bond(T, cf.times) @ cf.amounts
    m, se = sim.price(np.maximum(v, 0.0), T)
    assert abs(m - M.swaption(sw, PROJ)) < 4 * se
    with pytest.raises(ValueError):
        g2pp_paths(M, [1.0, 1.0], 10)
