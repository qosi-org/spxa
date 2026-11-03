"""Curve interpolation schemes and PiecewiseCurve."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.rates.interpolation import (
    LinearZero,
    LogLinearDiscount,
    PiecewiseCurve,
    discrete_forwards,
)

T = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
R = np.array([0.030, 0.032, 0.035, 0.038, 0.040])
SCHEMES = [LogLinearDiscount(), LinearZero()]


@pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
def test_nodes_reproduced(scheme: object) -> None:
    c = PiecewiseCurve.from_zero_rates(T, R, scheme)  # type: ignore[arg-type]
    assert c.discount(T) == pytest.approx(np.exp(-R * T), rel=1e-15)
    assert c.node_zero_rates == pytest.approx(R, rel=1e-14)
    assert c.discount(0.0) == 1.0


@pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
def test_forward_is_derivative_and_integrates(scheme: object) -> None:
    c = PiecewiseCurve.from_zero_rates(T, R, scheme)  # type: ignore[arg-type]
    t = np.linspace(0.01, 12.0, 397)
    t = t[np.min(np.abs(t[:, None] - T[None, :]), axis=1) > 1e-3]  # avoid kinks
    h = 1e-6
    fd = -(np.log(c.discount(t + h)) - np.log(c.discount(t - h))) / (2 * h)
    assert c.instantaneous_forward(t) == pytest.approx(fd, abs=1e-8)
    # the average forward over each interval is the discrete forward
    grid = np.concatenate([[0.0], T])
    fdisc = discrete_forwards(grid, np.concatenate([[0.0], -R * T]))
    for a, b, f in zip(grid[:-1], grid[1:], fdisc):
        x, w = np.polynomial.legendre.leggauss(40)
        s = 0.5 * (b - a) * x + 0.5 * (a + b)
        avg = 0.5 * np.sum(w * c.instantaneous_forward(s))
        assert avg == pytest.approx(f, rel=1e-6)  # kinks at nodes only cost quadrature accuracy


def test_log_linear_is_flat_forward() -> None:
    c = PiecewiseCurve.from_zero_rates(T, R)
    grid = np.concatenate([[0.0], T])
    fdisc = discrete_forwards(grid, np.concatenate([[0.0], -R * T]))
    for a, b, f in zip(grid[:-1], grid[1:], fdisc):
        s = np.linspace(a, b, 7)[1:-1]
        assert c.instantaneous_forward(s) == pytest.approx(np.full(5, f), abs=1e-15)
    # matches the market InterpolatedDiscountCurve
    from spxa.market.curves import InterpolatedDiscountCurve

    ref = InterpolatedDiscountCurve(T, np.exp(-R * T))
    t = np.linspace(0, 15, 301)
    assert c.discount(t) == pytest.approx(ref.discount(t), rel=1e-14)


def test_linear_zero() -> None:
    c = PiecewiseCurve.from_zero_rates(T, R, LinearZero())
    assert c.zero_rate(0.25) == pytest.approx(R[0], rel=1e-14)  # flat before t_1
    assert c.zero_rate(1.5) == pytest.approx(0.5 * (R[1] + R[2]), rel=1e-14)
    # f = r + t r' on [2, 5]
    slope = (R[3] - R[2]) / 3
    assert c.instantaneous_forward(3.0) == pytest.approx(R[2] + slope + 3 * slope, rel=1e-14)
    # flat-forward extrapolation with the end forward
    f_end = R[4] + 10 * (R[4] - R[3]) / 5
    assert c.instantaneous_forward(20.0) == pytest.approx(f_end, rel=1e-14)
    assert math.log(c.discount(20.0)) == pytest.approx(-R[4] * 10 - f_end * 10, rel=1e-14)


def test_extrapolation_continuous() -> None:
    for s in SCHEMES:
        c = PiecewiseCurve.from_zero_rates(T, R, s)
        lo, hi = c.discount(10.0 - 1e-9), c.discount(10.0 + 1e-9)
        assert abs(hi - lo) < 1e-10


def test_with_discounts_and_dates() -> None:
    c = PiecewiseCurve.from_zero_rates(T, R)
    p = c.node_discounts * 0.99
    c2 = c.with_discounts(p)
    assert c2.discount(T) == pytest.approx(p, rel=1e-15)
    assert c.discount(T) == pytest.approx(np.exp(-R * T), rel=1e-15)  # original untouched
    d = PiecewiseCurve(["2025-07-01", "2026-01-01"], [0.99, 0.98], reference_date="2025-01-01")
    assert d.node_times == pytest.approx([181 / 365, 365 / 365])
    assert d.discount("2026-01-01") == pytest.approx(0.98, rel=1e-15)
    assert "PiecewiseCurve" in repr(d)


def test_validation() -> None:
    with pytest.raises(ValueError):
        PiecewiseCurve([1.0, 0.5], [0.99, 0.98])
    with pytest.raises(ValueError):
        PiecewiseCurve([0.0, 1.0], [0.9, 0.98])
    with pytest.raises(ValueError):
        PiecewiseCurve([1.0], [-0.1])
    with pytest.raises(ValueError):
        PiecewiseCurve([1.0, 2.0], [0.99])
    c = PiecewiseCurve([0.0, 1.0], [1.0, 0.97])
    assert c.node_times.tolist() == [1.0]
    with pytest.raises(ValueError):
        c.discount(-0.1)
    with pytest.raises(ValueError):
        c.with_discounts([0.9, 0.8])
