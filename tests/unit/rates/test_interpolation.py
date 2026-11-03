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


# -- cubic schemes -------------------------------------------------------------
from spxa.rates.interpolation import (  # noqa: E402
    CubicSplineRT,
    CubicZero,
    MonotoneCubicRT,
    hyman_slopes,
)

ALL = [LogLinearDiscount(), LinearZero(), CubicSplineRT(), CubicSplineRT("not-a-knot")]
ALL += [CubicZero(), CubicZero("not-a-knot"), MonotoneCubicRT()]


@pytest.mark.parametrize("scheme", ALL, ids=repr)
def test_all_schemes_reproduce_nodes(scheme: object) -> None:
    c = PiecewiseCurve.from_zero_rates(T, R, scheme)  # type: ignore[arg-type]
    assert c.discount(T) == pytest.approx(np.exp(-R * T), rel=1e-14)
    t = np.linspace(0.01, 12.0, 397)
    t = t[np.min(np.abs(t[:, None] - T[None, :]), axis=1) > 1e-3]
    h = 1e-6
    fd = -(np.log(c.discount(t + h)) - np.log(c.discount(t - h))) / (2 * h)
    assert c.instantaneous_forward(t) == pytest.approx(fd, abs=1e-8)
    one = PiecewiseCurve([1.0], [0.97], scheme)  # type: ignore[arg-type]
    assert one.discount(2.0) == pytest.approx(0.97**2, rel=1e-14)


def _cubic_rt(t: np.ndarray) -> np.ndarray:
    return 0.02 * t + 0.003 * t**2 - 0.0002 * t**3


def test_not_a_knot_spline_exact_on_cubics() -> None:
    c = PiecewiseCurve(T, np.exp(-_cubic_rt(T)), CubicSplineRT("not-a-knot"))
    t = np.linspace(0, 10, 101)
    assert -np.log(c.discount(t)) == pytest.approx(_cubic_rt(t), abs=1e-15)
    assert c.instantaneous_forward(t) == pytest.approx(0.02 + 0.006 * t - 0.0006 * t**2, abs=1e-14)
    r = 0.03 + 0.001 * T - 0.0001 * T**2 + 0.000005 * T**3
    z = PiecewiseCurve.from_zero_rates(T, r, CubicZero("not-a-knot"))
    s = np.linspace(0.5, 10, 50)
    exact = 0.03 + 0.001 * s - 0.0001 * s**2 + 0.000005 * s**3
    assert z.zero_rate(s) == pytest.approx(exact, abs=1e-15)
    assert z.zero_rate(0.2) == pytest.approx(r[0], abs=1e-15)  # flat before t_1


def test_natural_spline_forwards_smooth_with_flat_ends() -> None:
    c = PiecewiseCurve.from_zero_rates(T, R, CubicSplineRT())
    for tn in T[:-1]:
        lo, hi = c.instantaneous_forward(tn - 1e-10), c.instantaneous_forward(tn + 1e-10)
        assert abs(hi - lo) < 1e-9
    h = 1e-5
    for e in (0.0, 10.0):
        a = max(e - h, 0.0)
        slope = (c.instantaneous_forward(a + h) - c.instantaneous_forward(a)) / h
        assert abs(slope) < 1e-6  # f' = (rt)'' = 0 at both ends


def test_hyman_exact_on_quadratics_and_filter() -> None:
    x = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    y = 0.02 * x + 0.001 * x**2
    assert hyman_slopes(x, y) == pytest.approx(0.02 + 0.002 * x, abs=1e-15)
    c = PiecewiseCurve(x[1:], np.exp(-y[1:]), MonotoneCubicRT())
    t = np.linspace(0, 10, 201)
    assert c.instantaneous_forward(t) == pytest.approx(0.02 + 0.002 * t, abs=1e-15)
    # filter: slopes clipped to 3 min(neighbouring secants), zero at extrema
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    s = hyman_slopes(xs, np.array([0.0, 1.0, 1.01, 3.0]))
    assert s[1] == pytest.approx(0.03) and s[2] == pytest.approx(0.03)
    assert hyman_slopes(xs, np.array([0.0, 1.0, 0.5, 2.0]))[1:3].tolist() == [0.0, 0.0]
    assert hyman_slopes(np.array([0.0, 1.0]), np.array([1.0, 3.0])).tolist() == [2.0, 2.0]


def test_monotone_keeps_forwards_positive() -> None:
    # a steep then nearly flat curve: discrete forwards 4%, 6%, 0.1%, 3%
    t = np.array([1.0, 2.0, 3.0, 4.0])
    rt = np.cumsum([0.04, 0.06, 0.001, 0.03])
    grid = np.linspace(0, 4, 4001)
    mono = PiecewiseCurve(t, np.exp(-rt), MonotoneCubicRT())
    assert np.min(mono.instantaneous_forward(grid)) >= 0.0
    nat = PiecewiseCurve(t, np.exp(-rt), CubicSplineRT())
    assert np.min(nat.instantaneous_forward(grid)) < -0.005  # the spline undershoots
    for a, b in zip([0.0, 1, 2, 3], [1.0, 2, 3, 4]):  # monotone discount factors
        seg = mono.discount(np.linspace(a, b, 101))
        assert np.all(np.diff(seg) <= 1e-16)


def test_scheme_flags_and_bc() -> None:
    assert LogLinearDiscount.local and LinearZero.local
    assert not (CubicSplineRT.local or CubicZero.local or MonotoneCubicRT.local)
    with pytest.raises(ValueError):
        CubicSplineRT("clamped")
    assert repr(CubicZero()) == "CubicZero('natural')"
