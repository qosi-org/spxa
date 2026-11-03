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
