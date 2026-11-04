"""Interpolation-scheme diagnostics: positivity, smoothness, locality."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.diagnostics import bump_locality, compare_schemes, forward_roughness
from spxa.rates.interpolation import (
    CubicSplineRT,
    CubicZero,
    LinearZero,
    LogLinearDiscount,
    MonotoneConvex,
    MonotoneCubicRT,
    PiecewiseCurve,
)

# a hump-and-dip zero curve of the kind Hagan & West use to stress schemes
T = [0.1, 1, 4, 9, 20, 30]
R = [0.081, 0.07, 0.044, 0.07, 0.04, 0.03]


def test_comparison_table() -> None:
    rows = {
        d.scheme: d
        for d in compare_schemes(
            T,
            R,
            [
                LogLinearDiscount(),
                LinearZero(),
                CubicZero(),
                CubicSplineRT(),
                MonotoneCubicRT(),
                MonotoneConvex(),
            ],
        )
    }
    ll, lz = rows["LogLinearDiscount()"], rows["LinearZero()"]
    cz, mc = rows["CubicZero('natural')"], rows["MonotoneConvex(positivity=False)"]
    # local schemes: a bump moves only the two intervals next to the node
    assert ll.locality == 1 and lz.locality == 1
    # global splines: a bump reaches every interval
    assert cz.locality >= len(T) - 1 and rows["CubicSplineRT('natural')"].locality == len(T)
    # monotone schemes stay nearly local
    assert mc.locality <= 2 and rows["MonotoneCubicRT()"].locality <= 2
    # positivity: linear zeros and cubic zeros go negative here, monotone convex does not
    assert lz.min_forward < 0 and cz.min_forward < -0.05
    assert mc.min_forward > 0 and rows["MonotoneCubicRT()"].min_forward > 0
    # flat forwards: the minimum is the last discrete forward (0.03*30 - 0.04*20)/10
    assert ll.min_forward == pytest.approx(0.01, abs=1e-12)
    # smooth schemes are orders of magnitude less rough than piecewise-flat forwards
    assert max(cz.roughness, mc.roughness) < 1e-2 * ll.roughness


def test_log_linear_sensitivity_is_exact() -> None:
    # bumping r_j moves the flat forward on (t_{j-1}, t_j) by t_j / (t_j - t_{j-1}) per unit
    spread, sens = bump_locality(T, R, LogLinearDiscount())
    assert spread == 1
    t = np.concatenate([[0.0], T])
    expected = max(
        max(t[j] / (t[j] - t[j - 1]), t[j] / (t[j + 1] - t[j]) if j < len(T) else 0)
        for j in range(1, len(t))
    )
    assert sens == pytest.approx(expected, rel=1e-6) and expected == pytest.approx(3.0)


def test_roughness_of_a_quadratic_forward() -> None:
    # zero rate r(t) = a + b t gives f = a + 2 b t, so the integral of f'^2 over [0.5, 10] is 4 b^2 9.5
    t = np.linspace(0.5, 10, 20)
    c = PiecewiseCurve.from_zero_rates(t, 0.02 + 0.001 * t, LinearZero())
    assert forward_roughness(c, 10.0, t_min=0.5) == pytest.approx(4e-6 * 9.5, rel=1e-9)
    # from 0 the flat front segment adds a forward jump of b * 0.5 at t = 0.5: grid-dependent
    assert forward_roughness(c, 10.0) > 10 * forward_roughness(c, 10.0, t_min=0.5)
