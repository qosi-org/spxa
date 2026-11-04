r"""Implied-volatility bumps and bucketed vega.

A bump is a function :math:`\delta(k, T)` added to the implied volatility
at log-forward moneyness :math:`k` (so it moves with the forward). The
bumped surface has total variance

.. math:: w'(k, T) = \bigl(\sigma(k, T) + \delta(k, T)\bigr)^2 T.

Bump shapes:

* :func:`parallel_bump`: :math:`\delta = s`;
* :func:`skew_bump`: :math:`\delta = s\,k`, a rotation about the ATM point;
* :func:`expiry_buckets`: triangular hats in :math:`T` on pillar expiries,
  flat beyond the first and last pillars. The hats sum to one, so the
  bucket bumps add up to the parallel bump and bucketed vegas sum to the
  parallel vega to first order.

:func:`bucketed_vega` reprices any function of a surface under central
bumps.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "BumpedModel",
    "parallel_bump",
    "skew_bump",
    "expiry_buckets",
    "bucketed_vega",
]

Bump = Callable[[np.ndarray, float], Any]


class BumpedModel:
    """Total-variance model with an implied-volatility bump added to a base model."""

    def __init__(self, base: Any, bump: Bump) -> None:
        self.base, self.bump = base, bump

    def total_variance(self, k: Any, T: float) -> Any:
        ka = np.asarray(k, float)
        w = np.asarray(self.base.total_variance(ka, T), float)
        sig = np.sqrt(w / T) + np.asarray(self.bump(ka, T), float)
        if np.any(sig < 0):
            raise ValueError("bump makes an implied volatility negative")
        out = sig**2 * T
        return out if out.ndim else float(out)


def parallel_bump(size: float) -> Bump:
    return lambda k, T: np.full(np.shape(k), size)


def skew_bump(size: float) -> Bump:
    return lambda k, T: size * np.asarray(k, float)


def expiry_buckets(pillars: Sequence[float], size: float) -> list[Bump]:
    """One triangular bump per pillar expiry; together they form a partition of unity in ``T``."""
    p = np.asarray(pillars, float)
    if p.ndim != 1 or p.size == 0 or p[0] <= 0 or np.any(np.diff(p) <= 0):
        raise ValueError("pillars must be positive and strictly increasing")

    def hat(i: int) -> Bump:
        def f(k: np.ndarray, T: float) -> np.ndarray:
            e = np.zeros(p.size)
            e[i] = 1.0
            return np.full(np.shape(k), size * float(np.interp(T, p, e)))

        return f

    return [hat(i) for i in range(p.size)]


def bucketed_vega(
    value: Callable[[Any], float], surface: Any, bumps: Sequence[Bump], size: float
) -> np.ndarray:
    """Central-difference sensitivities ``(V(+b) - V(-b)) / (2 size)`` for each bump ``b``.

    The ``bumps`` must be built with ``size`` and are negated by
    reflection, ``b(-)(k, T) = -b(k, T)``.
    """
    out = []
    for b in bumps:
        up = surface.bumped(b)
        dn = surface.bumped(lambda k, T, b=b: -np.asarray(b(k, T), float))
        out.append((value(up) - value(dn)) / (2 * size))
    return np.array(out)
