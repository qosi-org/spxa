r"""Interpolation schemes for curves built from discount-factor nodes.

A curve is described by node times :math:`0 = t_0 < t_1 < \dots < t_n`
and discount factors :math:`P_i = P(0, t_i)` with :math:`P_0 = 1`. A
scheme turns these nodes into a function :math:`t \mapsto \log P(0, t)`
and its derivative, the instantaneous forward
:math:`f(t) = -\partial_t \log P(0, t)`.

Every scheme reproduces the nodes exactly. They differ in what is
interpolated, and hence in the shape of the forward curve (Hagan & West,
*Methods for constructing a yield curve*, Wilmott, May 2008):

:class:`LogLinearDiscount`
    :math:`\log P` linear between nodes: piecewise-constant forwards equal
    to the discrete forwards :math:`f^d_i = (\log P_{i-1} - \log P_i)/(t_i - t_{i-1})`.
    Local, positive whenever the discrete forwards are, but discontinuous.
:class:`LinearZero`
    Zero rate :math:`r(t) = -\log P/t` linear between nodes, flat before
    :math:`t_1`. The forward :math:`f = r + t r'` has saw-tooth jumps.
:class:`CubicSplineRT`
    Cubic spline on :math:`r t = -\log P` through the origin: smooth
    forwards that can turn negative and are not local.
:class:`CubicZero`
    Cubic spline on :math:`r`, flat before :math:`t_1`.
:class:`MonotoneCubicRT`
    Hermite cubic on :math:`r t` with Hyman-filtered slopes: continuous
    forwards that stay non-negative when the discrete forwards are.
:class:`MonotoneConvex`
    Hagan–West monotone convex on the forwards themselves: continuous
    forwards, reproduces the discrete forwards' averages, optional
    positivity collar.

Beyond the last node every scheme extrapolates with the forward it has
at :math:`t_n` held flat. Negative times are rejected.
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Sequence

import numpy as np

from spxa.market.curves import TimeLike, YieldCurve
from spxa.market.dates import DateLike
from spxa.market.daycount import DayCount

__all__ = [
    "InterpolationScheme",
    "FittedInterpolation",
    "LogLinearDiscount",
    "LinearZero",
    "CubicSplineRT",
    "CubicZero",
    "MonotoneCubicRT",
    "MonotoneConvex",
    "hyman_slopes",
    "PiecewiseCurve",
    "discrete_forwards",
]


def discrete_forwards(times: np.ndarray, logp: np.ndarray) -> np.ndarray:
    r"""Continuously compounded forwards :math:`f^d_i` on :math:`[t_{i-1}, t_i]`."""
    return -np.diff(logp) / np.diff(times)


class FittedInterpolation(abc.ABC):
    r"""A scheme fitted to nodes ``times`` (starting at 0) and ``logp`` (starting at 0)."""

    def __init__(self, times: np.ndarray, logp: np.ndarray) -> None:
        self.times = times
        self.logp = logp

    @abc.abstractmethod
    def _inside(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(log P, f)`` for ``0 <= t <= t_n``."""

    def evaluate(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(log P(t), f(t))`` for arrays of non-negative times."""
        t = np.asarray(t, dtype=float)
        if np.any(t < 0):
            raise ValueError("negative time")
        tn = self.times[-1]
        inside = np.minimum(t, tn)
        logp, fwd = self._inside(inside)
        beyond = t > tn
        if np.any(beyond):
            f_end = self._inside(np.array([tn]))[1][0]
            logp = np.where(beyond, self.logp[-1] - f_end * (t - tn), logp)
            fwd = np.where(beyond, f_end, fwd)
        return logp, fwd

    def _locate(self, t: np.ndarray) -> np.ndarray:
        """Index ``i >= 1`` of the interval :math:`[t_{i-1}, t_i]` holding each ``t``."""
        return np.clip(np.searchsorted(self.times, t, side="right"), 1, self.times.size - 1)


class InterpolationScheme(abc.ABC):
    """Factory turning nodes into a :class:`FittedInterpolation`."""

    #: True if moving node ``i`` changes the curve only on adjacent intervals.
    local: bool = True

    @abc.abstractmethod
    def fit(self, times: np.ndarray, logp: np.ndarray) -> FittedInterpolation:
        """Fit to nodes; ``times[0] == 0`` and ``logp[0] == 0``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class _LogLinear(FittedInterpolation):
    def __init__(self, times: np.ndarray, logp: np.ndarray) -> None:
        super().__init__(times, logp)
        self.fd = discrete_forwards(times, logp)

    def _inside(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = self._locate(t)
        f = self.fd[i - 1]
        return self.logp[i - 1] - f * (t - self.times[i - 1]), f


class LogLinearDiscount(InterpolationScheme):
    r"""Linear in :math:`\log P` (flat forwards between nodes)."""

    def fit(self, times: np.ndarray, logp: np.ndarray) -> FittedInterpolation:
        return _LogLinear(times, logp)


class _LinearZero(FittedInterpolation):
    def __init__(self, times: np.ndarray, logp: np.ndarray) -> None:
        super().__init__(times, logp)
        r = -logp[1:] / times[1:]
        self.r = np.concatenate([[r[0]], r])
        self.slope = np.diff(self.r) / np.diff(times)
        self.slope[0] = 0.0

    def _inside(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = self._locate(t)
        s = self.slope[i - 1]
        r = self.r[i - 1] + s * (t - self.times[i - 1])
        return -r * t, r + s * t


class LinearZero(InterpolationScheme):
    r"""Continuously compounded zero rate linear between nodes, flat before :math:`t_1`."""

    def fit(self, times: np.ndarray, logp: np.ndarray) -> FittedInterpolation:
        return _LinearZero(times, logp)


class _Spline(FittedInterpolation):
    """scipy ``CubicSpline`` on ``r t = -log P`` (with the origin) or on ``r``."""

    def __init__(self, times: np.ndarray, logp: np.ndarray, on_rate: bool, bc: str) -> None:
        super().__init__(times, logp)
        from scipy.interpolate import CubicSpline

        self.on_rate = on_rate
        if on_rate:
            if times.size < 3 and bc == "not-a-knot":
                bc = "natural"
            r = -logp[1:] / times[1:]
            self.sp = CubicSpline(times[1:], r, bc_type=bc) if times.size > 2 else None
            self.r1 = r[0]
        else:
            self.sp = CubicSpline(times, -logp, bc_type=bc if times.size > 2 else "natural")

    def _inside(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.on_rate:
            assert self.sp is not None
            return -self.sp(t), self.sp(t, 1)
        if self.sp is None:
            return -self.r1 * t, np.full_like(t, self.r1)
        tc = np.maximum(t, self.times[1])
        r = np.where(t < self.times[1], self.r1, self.sp(tc))
        dr = np.where(t < self.times[1], 0.0, self.sp(tc, 1))
        return -r * t, r + t * dr


def _check_bc(bc: str) -> str:
    if bc not in ("natural", "not-a-knot"):
        raise ValueError("bc must be 'natural' or 'not-a-knot'")
    return bc


class CubicSplineRT(InterpolationScheme):
    r"""Cubic spline on :math:`r(t)\,t = -\log P` through the origin.

    Forwards are continuous and :math:`C^1`. ``bc`` is ``"natural"``
    (:math:`f' = 0` at both ends) or ``"not-a-knot"``. The spline is not
    local and can produce negative forwards from positive discrete
    forwards.
    """

    local = False

    def __init__(self, bc: str = "natural") -> None:
        self.bc = _check_bc(bc)

    def fit(self, times: np.ndarray, logp: np.ndarray) -> FittedInterpolation:
        return _Spline(times, logp, False, self.bc)

    def __repr__(self) -> str:
        return f"CubicSplineRT({self.bc!r})"


class CubicZero(InterpolationScheme):
    r"""Cubic spline on the zero rate through :math:`t_1, \dots, t_n`, flat before :math:`t_1`."""

    local = False

    def __init__(self, bc: str = "natural") -> None:
        self.bc = _check_bc(bc)

    def fit(self, times: np.ndarray, logp: np.ndarray) -> FittedInterpolation:
        return _Spline(times, logp, True, self.bc)

    def __repr__(self) -> str:
        return f"CubicZero({self.bc!r})"


def hyman_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Bessel (three-point) Hermite slopes with the Hyman (1983) monotonicity filter.

    Interior slopes are :math:`(h_i\Delta_{i-1} + h_{i-1}\Delta_i)/(h_{i-1} + h_i)`,
    exact for quadratics. Where the data are monotone the slope is clipped
    to :math:`[0, 3\min(\Delta_{i-1}, \Delta_i)]` (signs adjusted), and set to
    zero at local extrema, which makes the Hermite cubic monotone on every
    interval where the data are.
    """
    h = np.diff(x)
    d = np.diff(y) / h
    n = x.size
    s = np.empty(n)
    if n == 2:
        return np.full(2, d[0])
    s[1:-1] = (h[1:] * d[:-1] + h[:-1] * d[1:]) / (h[:-1] + h[1:])
    s[0] = d[0] + (d[0] - d[1]) * h[0] / (h[0] + h[1])
    s[-1] = d[-1] + (d[-1] - d[-2]) * h[-1] / (h[-1] + h[-2])
    left = np.concatenate([[d[0]], d])
    right = np.concatenate([d, [d[-1]]])
    for i in range(n):
        a, b = left[i], right[i]
        if a * b <= 0:
            s[i] = 0.0
        else:
            sg = 1.0 if a > 0 else -1.0
            s[i] = sg * min(max(sg * s[i], 0.0), 3.0 * min(abs(a), abs(b)))
    return s


class _Hermite(FittedInterpolation):
    def __init__(self, times: np.ndarray, logp: np.ndarray) -> None:
        super().__init__(times, logp)
        self.y = -logp
        self.s = hyman_slopes(times, self.y)

    def _inside(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = self._locate(t)
        x0, h = self.times[i - 1], np.diff(self.times)[i - 1]
        u = (t - x0) / h
        y0, y1 = self.y[i - 1], self.y[i]
        m0, m1 = self.s[i - 1] * h, self.s[i] * h
        h00, h10, h01, h11 = (
            2 * u**3 - 3 * u**2 + 1,
            u**3 - 2 * u**2 + u,
            -2 * u**3 + 3 * u**2,
            u**3 - u**2,
        )
        y = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
        d00, d10, d01, d11 = (
            6 * u**2 - 6 * u,
            3 * u**2 - 4 * u + 1,
            -6 * u**2 + 6 * u,
            3 * u**2 - 2 * u,
        )
        dy = (d00 * y0 + d10 * m0 + d01 * y1 + d11 * m1) / h
        return -y, dy


class MonotoneCubicRT(InterpolationScheme):
    r"""Hermite cubic on :math:`r t = -\log P` with Hyman-filtered Bessel slopes.

    Forwards are continuous, and non-negative wherever the discrete
    forwards are, because :math:`r t` is then interpolated monotonically.
    Each interval depends on its two neighbours, so the scheme is not
    local.
    """

    local = False

    def fit(self, times: np.ndarray, logp: np.ndarray) -> FittedInterpolation:
        return _Hermite(times, logp)


def _monotone_convex_g(
    x: np.ndarray, g0: np.ndarray, g1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Hagan–West :math:`g(x)` and :math:`G(x) = \int_0^x g`, elementwise.

    :math:`g` is the forward minus the discrete forward of the interval, with
    :math:`g(0) = g_0`, :math:`g(1) = g_1` and :math:`G(1) = 0`.
    """
    g = np.zeros_like(x)
    G = np.zeros_like(x)
    zero = (g0 == 0) & (g1 == 0)
    c1 = ~zero & (
        ((g0 < 0) & (-g0 / 2 <= g1) & (g1 <= -2 * g0))
        | ((g0 > 0) & (-g0 / 2 >= g1) & (g1 >= -2 * g0))
    )
    c2 = ~zero & ~c1 & (((g0 < 0) & (g1 > -2 * g0)) | ((g0 > 0) & (g1 < -2 * g0)))
    c3 = (
        ~zero
        & ~c1
        & ~c2
        & (((g0 > 0) & (g1 < 0) & (g1 > -g0 / 2)) | ((g0 < 0) & (g1 > 0) & (g1 < -g0 / 2)))
    )
    c4 = ~zero & ~c1 & ~c2 & ~c3
    with np.errstate(divide="ignore", invalid="ignore"):
        # (i) quadratic
        q0, q1 = (
            g0 * (1 - 4 * x + 3 * x * x) + g1 * (-2 * x + 3 * x * x),
            g0 * (x - 2 * x * x + x**3) + g1 * (x**3 - x * x),
        )
        g, G = np.where(c1, q0, g), np.where(c1, q1, G)
        # (ii) flat at g0, then quadratic to g1
        eta = (g1 + 2 * g0) / (g1 - g0)
        u = np.maximum(x - eta, 0.0)
        d = (1 - eta) ** 2
        g = np.where(c2, g0 + (g1 - g0) * u * u / d, g)
        G = np.where(c2, g0 * x + (g1 - g0) * u**3 / (3 * d), G)
        # (iii) quadratic from g0, then flat at g1
        eta = 3 * g1 / (g1 - g0)
        u = np.maximum(eta - x, 0.0)
        e2 = eta * eta
        g = np.where(c3, g1 + (g0 - g1) * u * u / e2, g)
        G = np.where(c3, g1 * x + (g0 - g1) * (eta**3 - u**3) / (3 * e2), G)
        # (iv) two quadratics meeting at the minimum (or maximum) A
        eta = g1 / (g0 + g1)
        A = -g0 * g1 / (g0 + g1)
        lft, rgt = np.maximum(eta - x, 0.0), np.maximum(x - eta, 0.0)
        e2, d = eta * eta, (1 - eta) ** 2
        # eta = 0 or 1 (one of g0, g1 is zero): a boundary layer of zero width
        left = (g0 - A) * np.where(e2 > 0, lft * lft / e2, x <= 0)
        right = (g1 - A) * np.where(d > 0, rgt * rgt / d, x >= 1)
        Gl = (g0 - A) * np.where(e2 > 0, (eta**3 - lft**3) / (3 * e2), 0.0)
        Gr = (g1 - A) * np.where(d > 0, rgt**3 / (3 * d), 0.0)
        g = np.where(c4, A + left + right, g)
        G = np.where(c4, A * x + Gl + Gr, G)
    return g, G


def _monotone_convex_scalar(x: float, g0: float, g1: float) -> tuple[float, float]:
    """Scalar version of :func:`_monotone_convex_g` (same cases, plain floats)."""
    if g0 == 0.0 and g1 == 0.0:
        return 0.0, 0.0
    if (g0 < 0 and -g0 / 2 <= g1 <= -2 * g0) or (g0 > 0 and -g0 / 2 >= g1 >= -2 * g0):
        g = g0 * (1 - 4 * x + 3 * x * x) + g1 * (-2 * x + 3 * x * x)
        return g, g0 * (x - 2 * x * x + x**3) + g1 * (x**3 - x * x)
    if (g0 < 0 and g1 > -2 * g0) or (g0 > 0 and g1 < -2 * g0):
        eta = (g1 + 2 * g0) / (g1 - g0)
        u, d = max(x - eta, 0.0), (1 - eta) ** 2
        return g0 + (g1 - g0) * u * u / d, g0 * x + (g1 - g0) * u**3 / (3 * d)
    if (g0 > 0 and -g0 / 2 < g1 < 0) or (g0 < 0 and 0 < g1 < -g0 / 2):
        eta = 3 * g1 / (g1 - g0)
        u, e2 = max(eta - x, 0.0), eta * eta
        return g1 + (g0 - g1) * u * u / e2, g1 * x + (g0 - g1) * (eta**3 - u**3) / (3 * e2)
    eta = g1 / (g0 + g1)
    A = -g0 * g1 / (g0 + g1)
    lft, rgt = max(eta - x, 0.0), max(x - eta, 0.0)
    e2, d = eta * eta, (1 - eta) ** 2
    left = (g0 - A) * (lft * lft / e2 if e2 > 0 else float(x <= 0))
    right = (g1 - A) * (rgt * rgt / d if d > 0 else float(x >= 1))
    Gl = (g0 - A) * ((eta**3 - lft**3) / (3 * e2) if e2 > 0 else 0.0)
    Gr = (g1 - A) * (rgt**3 / (3 * d) if d > 0 else 0.0)
    return A + left + right, A * x + Gl + Gr


class _MonotoneConvex(FittedInterpolation):
    def __init__(self, times: np.ndarray, logp: np.ndarray, positivity: bool) -> None:
        super().__init__(times, logp)
        fd = discrete_forwards(times, logp)
        self.fd = fd
        n = fd.size
        f = np.empty(n + 1)
        if n == 1:
            f[:] = fd[0]
        else:
            h = np.diff(times)
            f[1:-1] = (h[:-1] * fd[1:] + h[1:] * fd[:-1]) / (h[:-1] + h[1:])
            f[0] = fd[0] - 0.5 * (f[1] - fd[0])
            f[-1] = fd[-1] - 0.5 * (f[-2] - fd[-1])
            if positivity:
                f[0] = np.clip(f[0], 0.0, 2 * fd[0])
                f[1:-1] = np.clip(f[1:-1], 0.0, 2 * np.minimum(fd[:-1], fd[1:]))
                f[-1] = np.clip(f[-1], 0.0, 2 * fd[-1])
        self.f = f

    def _inside(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = self._locate(t)
        a, h = self.times[i - 1], np.diff(self.times)[i - 1]
        x = (t - a) / h
        fd = self.fd[i - 1]
        g0, g1 = self.f[i - 1] - fd, self.f[i] - fd
        if x.size == 1:  # scalar fast path: curve.discount(date) in pricing loops
            gs, Gs = _monotone_convex_scalar(float(x[0]), float(g0[0]), float(g1[0]))
            g, G = np.array([gs]), np.array([Gs])
        else:
            g, G = _monotone_convex_g(x, g0, g1)
        return self.logp[i - 1] - h * (fd * x + G), fd + g


class MonotoneConvex(InterpolationScheme):
    r"""Hagan–West monotone convex interpolation of forwards.

    The discrete forwards :math:`f^d_i` are the inputs. Node forwards are
    width-weighted averages of neighbouring :math:`f^d_i`, with
    :math:`f_0 = f^d_1 - (f_1 - f^d_1)/2` and similarly at :math:`t_n`. On
    each interval :math:`f = f^d_i + g(x)`, where :math:`g` is one of four
    piecewise-quadratic shapes, chosen so that :math:`g(0)`, :math:`g(1)`
    hit the node forwards, :math:`\int_0^1 g = 0` (the nodes are
    reproduced) and the forward stays between the extremes of the data
    where it can.

    With ``positivity=True`` the node forwards are collared to
    :math:`[0, 2\min(f^d_i, f^d_{i+1})]`, which makes every forward
    non-negative when the discrete forwards are. This is the positivity
    collar only; the "amelioration" of the 2008 paper is not applied.
    Moving one node changes the curve on at most four intervals.

    The forward is continuous except in a degenerate case. When exactly one
    of :math:`g_0, g_1` is zero and the other is not, shape (iv) collapses
    to a boundary layer of zero width and the forward jumps at that node.
    """

    local = False

    def __init__(self, positivity: bool = False) -> None:
        self.positivity = bool(positivity)

    def fit(self, times: np.ndarray, logp: np.ndarray) -> FittedInterpolation:
        return _MonotoneConvex(times, logp, self.positivity)

    def __repr__(self) -> str:
        return f"MonotoneConvex(positivity={self.positivity})"


class PiecewiseCurve(YieldCurve):
    r"""Discount curve through nodes :math:`(t_i, P_i)` under an interpolation scheme.

    Parameters
    ----------
    times, discounts
        Node times (years or dates) and discount factors. A node
        :math:`(0, 1)` is prepended if absent.
    scheme
        An :class:`InterpolationScheme`; defaults to :class:`LogLinearDiscount`.

    The instantaneous forward is analytic (the scheme's derivative), so
    :meth:`instantaneous_forward` ignores ``h``.
    """

    def __init__(
        self,
        times: Sequence[TimeLike],
        discounts: Sequence[float],
        scheme: Optional[InterpolationScheme] = None,
        reference_date: Optional[DateLike] = None,
        day_count: "str | DayCount" = "ACT/365F",
    ) -> None:
        super().__init__(reference_date, day_count)
        t = np.asarray(self.time(list(times)), dtype=float)
        p = np.asarray(discounts, dtype=float)
        if t.shape != p.shape or t.ndim != 1 or t.size == 0:
            raise ValueError("times and discounts must be 1-D arrays of equal length")
        if not np.all(np.isfinite(p)) or np.any(p <= 0):
            raise ValueError("discount factors must be positive and finite")
        if t[0] == 0.0:
            if abs(p[0] - 1.0) > 1e-14:
                raise ValueError("discount factor at t = 0 must be 1")
            t, p = t[1:], p[1:]
        if t.size == 0 or t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("node times must be positive and strictly increasing")
        self.scheme = scheme if scheme is not None else LogLinearDiscount()
        self.node_times = t
        self.node_discounts = p
        self._fit = self.scheme.fit(np.concatenate([[0.0], t]), np.concatenate([[0.0], np.log(p)]))

    @classmethod
    def from_zero_rates(
        cls,
        times: Sequence[float],
        rates: Sequence[float],
        scheme: Optional[InterpolationScheme] = None,
        **kwargs: object,
    ) -> "PiecewiseCurve":
        """Curve from continuously compounded zero rates at year fractions ``times``."""
        t = np.asarray(times, dtype=float)
        return cls(t, np.exp(-np.asarray(rates, dtype=float) * t), scheme, **kwargs)  # type: ignore[arg-type]

    def with_discounts(self, discounts: Sequence[float]) -> "PiecewiseCurve":
        """Same nodes and scheme with new discount factors."""
        out = PiecewiseCurve.__new__(PiecewiseCurve)
        out.reference_date, out.day_count = self.reference_date, self.day_count
        out.scheme, out.node_times = self.scheme, self.node_times
        p = np.asarray(discounts, dtype=float)
        if p.shape != self.node_times.shape or not np.all(np.isfinite(p)) or np.any(p <= 0):
            raise ValueError("need one positive finite discount factor per node")
        out.node_discounts = p
        out._fit = self.scheme.fit(
            np.concatenate([[0.0], self.node_times]), np.concatenate([[0.0], np.log(p)])
        )
        return out

    @property
    def node_zero_rates(self) -> np.ndarray:
        """Continuously compounded zero rates at the nodes."""
        return -np.log(self.node_discounts) / self.node_times

    def _discount(self, t: np.ndarray) -> np.ndarray:
        return np.exp(self._fit.evaluate(t)[0])

    def instantaneous_forward(self, t: TimeLike, h: float = 1e-5) -> Any:
        r""":math:`f(t) = -\partial_t \log P(0, t)` from the scheme (``h`` is ignored)."""
        tt, scalar = self._eval(t)
        out = self._fit.evaluate(np.atleast_1d(tt))[1]
        return float(out[0]) if scalar else out.reshape(tt.shape)

    def __repr__(self) -> str:
        return f"PiecewiseCurve(nodes={self.node_times.size}, scheme={self.scheme!r})"
