"""Cheyette model: bond reconstruction and the Hull-White limit."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from spxa.rates.cheyette import Cheyette
from tests.unit.rates.test_hull_white import CURVE

M = Cheyette(CURVE, 0.05, [0.012, 0.009, 0.007], beta=0.3, sigma_times=[1.0, 3.0])


def test_bonds_fit_the_curve_and_the_short_rate() -> None:
    T = np.array([0.5, 2.0, 10.0])
    assert M.discount_bond(0.0, T, 0.0, 0.0) == pytest.approx(CURVE.discount(T), rel=1e-15)
    # -d/dT log P(t, T) at T = t is the short rate f(0, t) + x, whatever y is
    t, x, y, h = 2.0, 0.004, 3e-4, 1e-6
    lp = np.log(M.discount_bond(t, np.array([t + h, t + 2 * h]), x, y))
    fd = -(4 * lp[0] - lp[1]) / (2 * h)  # one-sided, second order; log P(t, t) = 0
    assert fd == pytest.approx(float(M.short_rate(t, x)), rel=1e-6)


def test_bond_ratios_compose() -> None:
    # P(t,T) P(t,S)^-1 depends on (x, y) only through G(t,S) - G(t,T) and the quadratic term
    t, x, y = 1.5, -0.006, 2e-4
    T, S = 4.0, 9.0
    gT, gS = M.G(t, T), M.G(t, S)
    lhs = M.discount_bond(t, S, x, y) / M.discount_bond(t, T, x, y)
    rhs = CURVE.discount(S) / CURVE.discount(T) * np.exp(-(gS - gT) * x - 0.5 * (gS**2 - gT**2) * y)
    assert lhs == pytest.approx(rhs, rel=1e-14)


def test_local_vol_and_ybar() -> None:
    assert M.sigma_at([0.5, 1.0, 2.0, 5.0]) == pytest.approx([0.012, 0.009, 0.009, 0.007])
    assert M.local_vol(0.5, [-0.1, 0.0, 0.01]) == pytest.approx([0.0, 0.012, 0.015])
    t = 4.0
    ref = quad(
        lambda u: np.exp(-2 * 0.05 * (t - u)) * M.sigma_at(u) ** 2,
        0,
        t,
        points=[1, 3],
        epsabs=0,
        epsrel=1e-13,
    )[0]
    assert M.ybar(t) == pytest.approx(ref, rel=1e-12)


def test_beta_zero_is_hull_white() -> None:
    c = Cheyette(CURVE, 0.05, 0.01)
    hw = c.to_hull_white()
    T = np.array([3.0, 7.0])
    assert c.discount_bond(2.0, T, 0.003, c.ybar(2.0)) == pytest.approx(
        hw.discount_bond(2.0, T, 0.003), rel=1e-15
    )
    with pytest.raises(ValueError):
        M.to_hull_white()
    with pytest.raises(ValueError):
        Cheyette(CURVE, 0.0, 0.01)
