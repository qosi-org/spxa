r"""Wing asymptotics: Lee's moment formula and the tails of the implied density.

**Lee's moment formula** (Lee 2004). Let :math:`\beta_R = \limsup_{k\to\infty}
w(k)/k` and :math:`\beta_L = \limsup_{k\to-\infty} w(k)/|k|`, and let

.. math:: \tilde p = \sup\{p : E[S_T^{1+p}] < \infty\}, \qquad
          \tilde q = \sup\{q : E[S_T^{-q}] < \infty\}.

Then :math:`\beta_R = \psi(\tilde p)` and :math:`\beta_L = \psi(\tilde q)` with

.. math:: \psi(x) = 2 - 4\bigl(\sqrt{x^2 + x} - x\bigr)
          = \frac{4}{2 + 4x + \sqrt{(2 + 4x)^2 - 4}} \in [0, 2],

the second form being free of cancellation. The inverse is
:math:`x = (2 - \beta)^2/(8\beta)`; in particular no slice can have a wing
steeper than 2.

**Density tails.** If :math:`w(k) = \beta|k| + O(1)` then, from the
lognormal kernel in :func:`spxa.vol.risk_neutral_density`, the density of
:math:`X = \log(S_T/F_T)` decays like :math:`e^{-\lambda|k|}` up to
sub-exponential factors, with

.. math:: \lambda_R = \frac{1}{2\beta_R} + \frac12 + \frac{\beta_R}{8}
          = 1 + \tilde p, \qquad
          \lambda_L = \frac{1}{2\beta_L} - \frac12 + \frac{\beta_L}{8} = \tilde q,

which recovers Lee's formula from the integrability of :math:`e^{(1+p)X}`
and :math:`e^{-qX}`.

**Rogers–Tehranchi.** A surface free of static arbitrage has
:math:`|\partial_k w| \le 4` for all :math:`k` (Rogers & Tehranchi 2010).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from spxa.vol.arbitrage import _derivatives

__all__ = [
    "WingAsymptotics",
    "lee_moment_from_slope",
    "lee_slope_from_moment",
    "rogers_tehranchi_slope",
    "wing_asymptotics",
]


def lee_slope_from_moment(x: float) -> float:
    r"""Wing slope :math:`\psi(x)` of a critical moment :math:`x \in [0, \infty]`."""
    x = float(x)
    if x < 0 or math.isnan(x):
        raise ValueError("critical moment must be non-negative")
    if math.isinf(x):
        return 0.0
    u = 2.0 + 4.0 * x
    return 4.0 / (u + math.sqrt((u - 2.0) * (u + 2.0)))


def lee_moment_from_slope(beta: float) -> float:
    r"""Critical moment :math:`(2 - \beta)^2/(8\beta)` of a wing slope :math:`\beta \in [0, 2]`."""
    beta = float(beta)
    if not 0.0 <= beta <= 2.0:
        raise ValueError("Lee's bound: wing slopes lie in [0, 2]")
    if beta == 0.0:
        return math.inf
    return (2.0 - beta) ** 2 / (8.0 * beta)


@dataclass(frozen=True)
class WingAsymptotics:
    r"""Wing slopes, critical moments and density decay rates of a slice.

    ``right_moment`` is :math:`\tilde p` (moments :math:`E[S^{1+p}]`),
    ``left_moment`` is :math:`\tilde q` (moments :math:`E[S^{-q}]`), and the
    density of :math:`\log(S_T/F_T)` decays like :math:`e^{-\lambda|k|}`
    with ``right_decay`` :math:`= 1 + \tilde p` and ``left_decay``
    :math:`= \tilde q`.
    """

    left_slope: float
    right_slope: float

    @property
    def right_moment(self) -> float:
        return lee_moment_from_slope(self.right_slope)

    @property
    def left_moment(self) -> float:
        return lee_moment_from_slope(self.left_slope)

    @property
    def right_decay(self) -> float:
        return 1.0 + self.right_moment

    @property
    def left_decay(self) -> float:
        return self.left_moment

    @property
    def satisfies_lee_bound(self) -> bool:
        return max(self.left_slope, self.right_slope) <= 2.0


def wing_asymptotics(slice_: Any, k_far: float = 1.0e4) -> WingAsymptotics:
    r"""Wing slopes of a slice, exact when it exposes ``left_slope``/``right_slope``.

    Otherwise :math:`\beta \approx (w(\pm 2K) - w(\pm K))/K` with
    :math:`K` = ``k_far``, which is exact for wings of the form
    :math:`\beta|k| + c + o(1)` up to the :math:`o(1)` term.
    Raises if a slope exceeds Lee's bound of 2.
    """
    if hasattr(slice_, "left_slope") and hasattr(slice_, "right_slope"):
        left, right = float(slice_.left_slope), float(slice_.right_slope)
    else:
        f = slice_.total_variance if hasattr(slice_, "total_variance") else slice_
        K = float(k_far)
        w = np.asarray(f(np.array([-2 * K, -K, K, 2 * K])), float)
        left, right = (w[0] - w[1]) / K, (w[3] - w[2]) / K
    for name, b in (("left", left), ("right", right)):
        if b > 2.0 + 1e-12:
            raise ValueError(f"{name} wing slope {b:.6g} violates Lee's bound of 2")
    return WingAsymptotics(max(left, 0.0), max(right, 0.0))


def rogers_tehranchi_slope(slice_: Any, k: Any = None) -> float:
    r""":math:`\max_k |w'(k)|` on a grid (default 4001 points on [-5, 5]); at most 4 if arbitrage-free."""
    kk = np.linspace(-5.0, 5.0, 4001) if k is None else np.asarray(k, float)
    _, dw, _ = _derivatives(slice_, kk)
    return float(np.max(np.abs(dw)))
