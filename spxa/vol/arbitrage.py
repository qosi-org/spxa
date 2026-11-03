r"""Static-arbitrage diagnostics for total-variance slices and surfaces.

**Butterfly arbitrage.** A slice :math:`w(k)` is free of butterfly
arbitrage if and only if the implied density of :math:`\log(S_T/F_T)`,

.. math:: p(k) = \frac{g(k)}{\sqrt{2\pi w(k)}}
          \exp\Bigl(-\frac{d_-(k)^2}{2}\Bigr), \qquad
          d_-(k) = -\frac{k}{\sqrt{w}} - \frac{\sqrt w}{2},

is non-negative and the call prices tend to zero as :math:`k \to \infty`
(Gatheral & Jacquier 2014, Lemma 2.2), where Durrleman's function is

.. math:: g(k) = \Bigl(1 - \frac{k\,w'}{2w}\Bigr)^2
          - \frac{w'^2}{4}\Bigl(\frac1w + \frac14\Bigr) + \frac{w''}{2}.

**Calendar arbitrage.** With proportional dividends a surface is free of
calendar-spread arbitrage if and only if :math:`T \mapsto w(k, T)` is
non-decreasing for every :math:`k` (Lemma 2.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "ButterflyCheck",
    "CalendarCheck",
    "check_butterfly",
    "check_calendar",
    "durrleman_g",
    "risk_neutral_density",
]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _derivatives(
    slice_: Any, k: np.ndarray, h: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``w, w', w''`` from the slice's own methods, or central differences of a callable."""
    if hasattr(slice_, "total_variance"):
        w = np.asarray(slice_.total_variance(k), float)
        f: Callable[[np.ndarray], Any] = slice_.total_variance
    else:
        f = slice_
        w = np.asarray(f(k), float)
    if hasattr(slice_, "dw") and hasattr(slice_, "d2w"):
        return w, np.asarray(slice_.dw(k), float), np.asarray(slice_.d2w(k), float)
    wp = np.asarray(f(k + h), float)
    wm = np.asarray(f(k - h), float)
    return w, (wp - wm) / (2 * h), (wp - 2 * w + wm) / (h * h)


def durrleman_g(k: Any, w: Any, dw: Any, d2w: Any) -> Any:
    """Durrleman's :math:`g(k)` from ``w``, ``w'`` and ``w''`` (arrays broadcast)."""
    k, w, dw, d2w = (np.asarray(x, float) for x in (k, w, dw, d2w))
    if np.any(w <= 0):
        raise ValueError("total variance must be positive")
    g = (1.0 - k * dw / (2.0 * w)) ** 2 - 0.25 * dw * dw * (1.0 / w + 0.25) + 0.5 * d2w
    return float(g) if g.ndim == 0 else g


def risk_neutral_density(slice_: Any, k: Any) -> Any:
    r"""Density of :math:`\log(S_T/F_T)` implied by a slice, at log-moneyness ``k``."""
    kk = np.asarray(k, float)
    w, dw, d2w = _derivatives(slice_, kk)
    g = durrleman_g(kk, w, dw, d2w)
    sw = np.sqrt(w)
    dm = -kk / sw - 0.5 * sw
    p = g / (_SQRT_2PI * sw) * np.exp(-0.5 * dm * dm)
    return float(p) if np.ndim(p) == 0 else p


@dataclass(frozen=True)
class ButterflyCheck:
    """Result of :func:`check_butterfly`."""

    arbitrage_free: bool
    min_g: float
    argmin: float
    negative_intervals: list[tuple[float, float]]

    def __bool__(self) -> bool:
        return self.arbitrage_free


def check_butterfly(slice_: Any, k: Any = None, tol: float = 0.0) -> ButterflyCheck:
    """Scan :math:`g(k)` on a grid (default 2001 points on [-3, 3]).

    ``negative_intervals`` lists the grid intervals where ``g < -tol``.
    """
    kk = np.linspace(-3.0, 3.0, 2001) if k is None else np.asarray(k, float)
    w, dw, d2w = _derivatives(slice_, kk)
    if np.any(w <= 0):
        i = int(np.argmin(w))
        return ButterflyCheck(False, -np.inf, float(kk[i]), [(float(kk[i]), float(kk[i]))])
    g = np.asarray(durrleman_g(kk, w, dw, d2w))
    neg = g < -tol
    intervals: list[tuple[float, float]] = []
    i = 0
    while i < kk.size:
        if neg[i]:
            j = i
            while j + 1 < kk.size and neg[j + 1]:
                j += 1
            intervals.append((float(kk[i]), float(kk[j])))
            i = j + 1
        else:
            i += 1
    i = int(np.argmin(g))
    return ButterflyCheck(not intervals, float(g[i]), float(kk[i]), intervals)


@dataclass(frozen=True)
class CalendarCheck:
    """Result of :func:`check_calendar`: the worst decrease of total variance."""

    arbitrage_free: bool
    worst: float
    at_k: float
    between: tuple[int, int]

    def __bool__(self) -> bool:
        return self.arbitrage_free


def check_calendar(slices: Sequence[Any], k: Any = None, tol: float = 0.0) -> CalendarCheck:
    """Check :math:`w(k, T_{i+1}) \\ge w(k, T_i)` for slices ordered by expiry."""
    if len(slices) < 2:
        return CalendarCheck(True, 0.0, float("nan"), (0, 0))
    kk = np.linspace(-3.0, 3.0, 2001) if k is None else np.asarray(k, float)
    ws = [
        np.asarray(s.total_variance(kk) if hasattr(s, "total_variance") else s(kk), float)
        for s in slices
    ]
    worst, at, pair = 0.0, float("nan"), (0, 0)
    for i in range(len(ws) - 1):
        d = ws[i] - ws[i + 1]
        j = int(np.argmax(d))
        if d[j] > worst:
            worst, at, pair = float(d[j]), float(kk[j]), (i, i + 1)
    return CalendarCheck(worst <= tol, worst, at, pair)
