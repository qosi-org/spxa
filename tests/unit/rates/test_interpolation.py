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


# -- Hagan-West monotone convex -----------------------------------------------
from spxa.rates.interpolation import MonotoneConvex, _monotone_convex_g  # noqa: E402


def test_monotone_convex_reference_values() -> None:
    # tf-quant-finance's documented example of rates.hagan_west.monotone_convex
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.75, 4.0, 4.75, 5.0, 4.75]) / 100
    c = PiecewiseCurve.from_zero_rates(t, y, MonotoneConvex())
    got = c.instantaneous_forward(np.array([0.3, 1.3, 2.1, 4.5]))
    assert got == pytest.approx([0.0229375, 0.05010625, 0.0609, 0.03625], abs=1e-15)
    # node forwards: f_0 = 2.125%, interior averages, f_5 = 3.25%
    assert c.instantaneous_forward(t) == pytest.approx(
        [0.04, 0.0575, 0.06, 0.0475, 0.0325], abs=1e-15
    )
    assert c.instantaneous_forward(0.0) == pytest.approx(0.02125, abs=1e-15)
    assert c.discount(t) == pytest.approx(np.exp(-y * t), rel=1e-15)


@pytest.mark.parametrize("seed", range(5))
def test_monotone_convex_shapes(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, 20001)
    for g0, g1 in rng.normal(size=(200, 2)):
        g, G = _monotone_convex_g(x, np.full_like(x, g0), np.full_like(x, g1))
        assert g[0] == pytest.approx(g0, abs=1e-12) and g[-1] == pytest.approx(g1, abs=1e-12)
        assert abs(G[-1]) < 1e-12  # zero mean: the discrete forward is reproduced
        # G' = g: the trapezoid error is at most h^2 max|g''| / 12 (g'' piecewise constant)
        trap = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))])
        tol = np.max(np.abs(np.diff(g, 2))) / 12 + 1e-12
        assert np.max(np.abs(G - trap)) <= 1.01 * tol
    for g0, g1 in [(0.0, 0.0), (1.0, -2.0), (-1.0, 2.0), (1.0, -0.5), (0.0, 1.0), (1.0, 0.0)]:
        g, G = _monotone_convex_g(x, np.full_like(x, g0), np.full_like(x, g1))
        assert g[0] == pytest.approx(g0, abs=1e-12) and g[-1] == pytest.approx(g1, abs=1e-12)
        assert abs(G[-1]) < 1e-12 and np.all(np.isfinite(g))


def test_monotone_convex_curve_properties() -> None:
    t = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    fd = np.array([0.031, 0.034, 0.029, 0.036, 0.041, 0.038, 0.033, 0.035])
    rt = np.cumsum(fd * np.diff(np.concatenate([[0.0], t])))
    c = PiecewiseCurve(t, np.exp(-rt), MonotoneConvex())
    assert c.discount(t) == pytest.approx(np.exp(-rt), rel=1e-15)
    for tn in t[:-1]:  # continuous forwards
        assert (
            abs(c.instantaneous_forward(tn + 1e-12) - c.instantaneous_forward(tn - 1e-12)) < 1e-10
        )
    # interval averages of the forward are the discrete forwards
    grid = np.concatenate([[0.0], t])
    xg, wg = np.polynomial.legendre.leggauss(60)
    for a, b, f in zip(grid[:-1], grid[1:], fd):
        s = 0.5 * (b - a) * xg + 0.5 * (a + b)
        assert 0.5 * np.sum(wg * c.instantaneous_forward(s)) == pytest.approx(f, abs=1e-7)
    # locality: moving node i changes the curve only on [t_{i-2}, t_{i+2}]
    i = 4  # node t_4 = 2.0 (1-based)
    bumped = c.node_discounts.copy()
    bumped[i - 1] *= 0.999
    c2 = c.with_discounts(bumped)
    lo, hi = grid[i - 2], grid[i + 2]
    s = np.linspace(0, 12, 1201)
    out = (s < lo - 1e-9) | (s > hi + 1e-9)
    f1, f2 = c.instantaneous_forward(s), c2.instantaneous_forward(s)
    assert np.max(np.abs(f1[out & (s < lo)] - f2[out & (s < lo)])) == 0.0
    assert np.max(np.abs(f1[(s > hi + 1e-9)] - f2[(s > hi + 1e-9)])) == 0.0
    assert np.max(np.abs(f1 - f2)) > 1e-4


def test_monotone_convex_positivity_collar() -> None:
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fd = np.array([0.04, 0.001, 0.05, 0.002, 0.03])
    rt = np.cumsum(fd)
    s = np.linspace(0, 5, 5001)
    raw = PiecewiseCurve(t, np.exp(-rt), MonotoneConvex())
    assert np.min(raw.instantaneous_forward(s)) < 0
    safe = PiecewiseCurve(t, np.exp(-rt), MonotoneConvex(positivity=True))
    assert np.min(safe.instantaneous_forward(s)) >= 0.0
    assert safe.discount(t) == pytest.approx(np.exp(-rt), rel=1e-15)
    one = PiecewiseCurve([2.0], [0.94], MonotoneConvex())
    assert one.instantaneous_forward(np.array([0.5, 3.0])) == pytest.approx(
        [-math.log(0.94) / 2] * 2
    )
    assert repr(MonotoneConvex(True)) == "MonotoneConvex(positivity=True)"
