r"""Carry implied by forwards: dividend yields, repo rates and forward curves.

A forward for delivery at :math:`T` satisfies

.. math:: F(T) = S\,\frac{Q(T)}{P_{\mathrm{repo}}(T)},

where :math:`Q` is the dividend "discount factor" and :math:`P_{\mathrm{repo}}`
the funding (repo) discount factor. Only the ratio is observable, so the
market forward pins down the *net* carry; splitting it into dividends and
repo needs one of the two from elsewhere. The helpers below return the
continuously compounded rates

.. math:: q = -\frac{1}{T}\log\frac{F P_{\mathrm{repo}}}{S}, \qquad
          r_{\mathrm{repo}} = \frac{1}{T}\log\frac{F}{S} + q .

:class:`ImpliedForwardCurve` interpolates market forwards linearly in
:math:`\log(F/S)` (piecewise-constant net carry) and exposes the
equivalent :class:`~spxa.market.curves.EquityForward`. Discrete cash
dividends make true forwards jump; between listed expiries the curve is
then only an interpolation.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from spxa.market.curves import EquityForward, InterpolatedDiscountCurve, TimeLike
from spxa.market.parity import ParityFit, implied_forwards
from spxa.market.quotes import OptionSurfaceQuotes

__all__ = [
    "ImpliedForwardCurve",
    "implied_carry",
    "implied_dividend_yield",
    "implied_repo_rate",
]


def _check(spot: Any, forward: Any, T: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s, f, t = (np.asarray(v, float) for v in (spot, forward, T))
    if np.any(s <= 0) or np.any(f <= 0):
        raise ValueError("spot and forward must be positive")
    if np.any(t <= 0):
        raise ValueError("T must be positive")
    return s, f, t


def _out(x: np.ndarray) -> Any:
    return float(x) if x.ndim == 0 else x


def implied_carry(spot: Any, forward: Any, T: Any) -> Any:
    r"""Net carry rate :math:`\log(F/S)/T` (continuous compounding)."""
    s, f, t = _check(spot, forward, T)
    return _out(np.log(f / s) / t)


def implied_dividend_yield(spot: Any, forward: Any, T: Any, repo_discount: Any) -> Any:
    r"""Continuous dividend yield :math:`-\log(F P_{\mathrm{repo}}/S)/T`.

    ``repo_discount`` is the funding discount factor to ``T`` (the
    risk-free one when stock borrow is general collateral). The result
    includes any borrow fee not captured by ``repo_discount``.
    """
    s, f, t = _check(spot, forward, T)
    p = np.asarray(repo_discount, float)
    if np.any(p <= 0):
        raise ValueError("repo_discount must be positive")
    return _out(-np.log(f * p / s) / t)


def implied_repo_rate(spot: Any, forward: Any, T: Any, dividend_yield: Any = 0.0) -> Any:
    r"""Continuous repo rate :math:`\log(F/S)/T + q` given the dividend yield ``q``."""
    s, f, t = _check(spot, forward, T)
    return _out(np.log(f / s) / t + np.asarray(dividend_yield, float))


class ImpliedForwardCurve:
    r"""Forward curve through market forwards.

    Parameters
    ----------
    spot : float
    times : sequence of float
        Expiries in years (strictly increasing, positive).
    forwards : sequence of float
    discounts : sequence of float, optional
        Risk-free discount factors at ``times`` (e.g. from
        :func:`~spxa.market.parity.implied_forward`); required for
        :meth:`to_equity_forward` and :meth:`dividend_yield`.

    Between nodes :math:`\log F` is linear in :math:`t`, starting from
    :math:`F(0) = S`; beyond the last node the last carry is extrapolated.
    """

    def __init__(
        self,
        spot: float,
        times: Sequence[float],
        forwards: Sequence[float],
        discounts: Optional[Sequence[float]] = None,
    ) -> None:
        t = np.asarray(times, float)
        f = np.asarray(forwards, float)
        if spot <= 0:
            raise ValueError("spot must be positive")
        if t.ndim != 1 or t.shape != f.shape or t.size == 0:
            raise ValueError("times and forwards must be 1-D arrays of equal length")
        if t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("times must be positive and strictly increasing")
        if np.any(f <= 0):
            raise ValueError("forwards must be positive")
        self.spot = float(spot)
        self.times, self.forwards = t, f
        self._t = np.concatenate([[0.0], t])
        self._logm = np.concatenate([[0.0], np.log(f / spot)])
        self.discounts: Optional[np.ndarray] = None
        if discounts is not None:
            d = np.asarray(discounts, float)
            if d.shape != t.shape or np.any(d <= 0):
                raise ValueError("discounts must be positive and match times")
            self.discounts = d

    # -- construction ---------------------------------------------------
    @classmethod
    def from_parity_fits(
        cls, spot: float, times: Sequence[float], fits: Sequence[ParityFit]
    ) -> "ImpliedForwardCurve":
        return cls(spot, times, [x.forward for x in fits], [x.discount for x in fits])

    @classmethod
    def from_surface(
        cls, surface: OptionSurfaceQuotes, spot: Optional[float] = None, **kwargs: Any
    ) -> "ImpliedForwardCurve":
        """Fit parity forwards on every chain (``kwargs`` go to
        :func:`~spxa.market.parity.implied_forwards`)."""
        s = spot if spot is not None else surface.spot
        if s is None:
            raise ValueError("spot is required")
        return cls.from_parity_fits(s, surface.expiries, implied_forwards(surface, **kwargs))

    # -- evaluation -----------------------------------------------------
    def _logm_at(self, t: np.ndarray) -> np.ndarray:
        if np.any(t < 0):
            raise ValueError("negative time")
        out = np.interp(t, self._t, self._logm)
        beyond = t > self._t[-1]
        if np.any(beyond):
            slope = (self._logm[-1] - self._logm[-2]) / (self._t[-1] - self._t[-2])
            out[beyond] = self._logm[-1] + slope * (t[beyond] - self._t[-1])
        return out

    def forward(self, t: TimeLike) -> Any:
        tt = np.asarray(t, float)
        return _out(self.spot * np.exp(self._logm_at(np.atleast_1d(tt)).reshape(tt.shape)))

    def carry(self, t: TimeLike) -> Any:
        """Average net carry :math:`\\log(F(t)/S)/t` (the instantaneous carry at ``t = 0``)."""
        tt = np.atleast_1d(np.asarray(t, float))
        lm = self._logm_at(tt)
        first = (self._logm[1] - self._logm[0]) / (self._t[1] - self._t[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(tt > 0, lm / np.where(tt > 0, tt, 1.0), first)
        return _out(out.reshape(np.shape(t)))

    def _need_discounts(self) -> np.ndarray:
        if self.discounts is None:
            raise ValueError("the curve was built without discount factors")
        return self.discounts

    def discount_curve(self) -> InterpolatedDiscountCurve:
        """Risk-free curve through the implied discount factors (log-linear)."""
        return InterpolatedDiscountCurve(self.times, self._need_discounts())

    def dividend_curve(self) -> InterpolatedDiscountCurve:
        r"""Dividend factors :math:`Q = F P / S` at the nodes, log-linear in between."""
        d = self._need_discounts()
        return InterpolatedDiscountCurve(self.times, self.forwards * d / self.spot)

    def dividend_yield(self, t: TimeLike) -> Any:
        """Continuous dividend yield (net of borrow) to ``t`` against the implied risk-free curve."""
        return self.dividend_curve().zero_rate(t)

    def to_equity_forward(self) -> EquityForward:
        """Equivalent :class:`EquityForward`; its forwards equal :meth:`forward` everywhere."""
        return EquityForward(self.spot, self.discount_curve(), dividend_yield=self.dividend_curve())

    def __repr__(self) -> str:
        return f"ImpliedForwardCurve(spot={self.spot}, nodes={self.times.size})"

    def __len__(self) -> int:
        return int(self.times.size)
