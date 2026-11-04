"""Exact Hull-White simulation under the risk-neutral measure."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.rates.hull_white import HullWhite
from spxa.rates.hw_mc import hull_white_paths
from tests.unit.rates.test_hull_white import CURVE, HW_PW

TIMES = [0.5, 1.0, 2.0, 3.5, 5.0]


@pytest.fixture(scope="module")
def paths():  # type: ignore[no-untyped-def]
    return hull_white_paths(HW_PW, TIMES, 200_000, seed=7)


def test_deterministic_parts() -> None:
    for t in (0.5, 2.0, 4.0):
        # m(t) = int e^{-a(t-u)} y(u) du, and Var(J) = int B(u,t)^2 sigma^2 du
        m = quad(lambda u: np.exp(-HW_PW.a * (t - u)) * HW_PW.y(u), 0, t, points=[1.0, 3.0])[0]
        assert HW_PW.mean_x(t) == pytest.approx(m, rel=1e-10)
        sig = lambda u: HW_PW.sigma[np.searchsorted(HW_PW.sigma_times, u, "right")]  # noqa: E731
        v = quad(lambda u: (HW_PW.B(u, t) * sig(u)) ** 2, 0, t, points=[1.0, 3.0])[0]
        assert HW_PW.integrated_variance(t) == pytest.approx(v, rel=1e-10)


def test_moments_of_x(paths) -> None:  # type: ignore[no-untyped-def]
    for t in TIMES:
        x, _ = paths.at(t)
        sd = np.sqrt(HW_PW.y(t))
        assert abs(x.mean() - HW_PW.mean_x(t)) < 4 * sd / np.sqrt(x.size)
        assert x.var() == pytest.approx(HW_PW.y(t), rel=4 * np.sqrt(2 / x.size))


def test_discount_bonds_reprice_the_curve(paths) -> None:  # type: ignore[no-untyped-def]
    for t in TIMES:
        m, se = paths.price(np.ones(paths.x.shape[1]), t)
        assert abs(m - CURVE.discount(t)) < 4 * se
        # D(t) P(t, t + 3) prices the bond maturing at t + 3
        m, se = paths.price(paths.zero_bond(t, t + 3.0), t)
        assert abs(m - CURVE.discount(t + 3.0)) < 4 * se


def test_bond_option_and_short_rate(paths) -> None:  # type: ignore[no-untyped-def]
    T, S = 2.0, 5.0
    K = float(CURVE.discount(S) / CURVE.discount(T))
    m, se = paths.price(np.maximum(paths.zero_bond(T, S) - K, 0.0), T)
    assert abs(m - HW_PW.zcb_option(T, S, K)) < 4 * se
    r = paths.short_rate(1.0)
    assert r.mean() == pytest.approx(
        HW_PW.phi(1.0) + HW_PW.mean_x(1.0), abs=4 * r.std() / np.sqrt(r.size)
    )
    with pytest.raises(KeyError):
        paths.at(1.5)
    with pytest.raises(ValueError):
        hull_white_paths(HullWhite(CURVE, 0.1, 0.01), [1.0, 0.5], 10)
