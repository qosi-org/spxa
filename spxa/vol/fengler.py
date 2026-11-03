r"""Arbitrage-free smoothing of call prices with cubic splines (Fengler, 2009).

Fengler, *Arbitrage-free smoothing of the implied volatility surface*,
Quantitative Finance 9(4), 2009, fits a natural cubic spline to call
prices under shape constraints. In normalised units :math:`c = C/(D F)`,
:math:`u = K/F` and with knots :math:`u_1 < \dots < u_n`, the spline is
described by its values :math:`g_i` and second derivatives
:math:`\gamma_i` (with :math:`\gamma_1 = \gamma_n = 0`). Green &
Silverman's band matrices :math:`Q` (:math:`n \times (n-2)`) and
:math:`R` (:math:`(n-2) \times (n-2)`) give the conditions:

* a valid spline (:math:`C^2` at the knots) when :math:`Q^\top g = R\gamma`;
* roughness :math:`\int g''^2 = \gamma^\top R \gamma`.

The fit solves the quadratic programme

.. math::

    \min_{g, \gamma}\ \sum_i w_i (y_i - g_i)^2 + \lambda\, \gamma^\top R \gamma
    \quad \text{s.t.} \quad Q^\top g = R \gamma,\ \gamma \ge 0,\
    g'(u_1) \ge -1,\ g'(u_n) \le 0,\ (1 - u_i)^+ \le g_i \le 1,

and the result is convex, decreasing, within the static bounds and
:math:`C^2`. It is therefore free of butterfly and vertical-spread
arbitrage on :math:`[u_1, u_n]`, and the risk-neutral density there is
:math:`\partial^2_K C = D\, g''(K/F)/F`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from spxa.market.implied import implied_black_volatility

__all__ = ["FenglerSmile", "fengler_smooth", "spline_matrices"]


def spline_matrices(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Green–Silverman ``Q`` and ``R`` of a natural cubic spline on knots ``u``."""
    u = np.asarray(u, dtype=float)
    n = u.size
    if n < 3 or np.any(np.diff(u) <= 0):
        raise ValueError("need at least 3 strictly increasing knots")
    h = np.diff(u)
    Q = np.zeros((n, n - 2))
    R = np.zeros((n - 2, n - 2))
    for j in range(n - 2):  # interior knot j + 1
        Q[j, j] = 1 / h[j]
        Q[j + 1, j] = -1 / h[j] - 1 / h[j + 1]
        Q[j + 2, j] = 1 / h[j + 1]
        R[j, j] = (h[j] + h[j + 1]) / 3
        if j < n - 3:
            R[j, j + 1] = R[j + 1, j] = h[j + 1] / 6
    return Q, R


@dataclass(frozen=True)
class FenglerSmile:
    """A fitted natural cubic call-price spline for one expiry."""

    knots: np.ndarray  # u_i = K_i / F
    values: np.ndarray  # g_i = C_i / (D F)
    gammas: np.ndarray  # g''(u_i), zero at both ends
    forward: float
    discount: float
    expiry: Optional[float] = None

    # ---------------------------------------------------------- spline maths
    def _locate(self, u: np.ndarray) -> np.ndarray:
        if np.any(u < self.knots[0] - 1e-14) or np.any(u > self.knots[-1] + 1e-14):
            raise ValueError("strike outside the fitted range")
        return np.clip(np.searchsorted(self.knots, u, side="right") - 1, 0, self.knots.size - 2)

    def normalised(self, u: Any, nu: int = 0) -> Any:
        """:math:`g^{(\\nu)}(u)` for ``nu`` = 0, 1, 2."""
        ua = np.asarray(u, dtype=float)
        i = self._locate(ua)
        x, g, c = self.knots, self.values, self.gammas
        h = x[i + 1] - x[i]
        a, b = ua - x[i], x[i + 1] - ua
        g0, g1, c0, c1 = g[i], g[i + 1], c[i], c[i + 1]
        if nu == 0:
            out = (
                (c0 * b**3 + c1 * a**3) / (6 * h)
                + (g0 / h - c0 * h / 6) * b
                + (g1 / h - c1 * h / 6) * a
            )
        elif nu == 1:
            out = (c1 * a**2 - c0 * b**2) / (2 * h) + (g1 - g0) / h - (c1 - c0) * h / 6
        elif nu == 2:
            out = (b * c0 + a * c1) / h
        else:
            raise ValueError("nu must be 0, 1 or 2")
        return out if out.ndim else float(out)

    def price(self, K: Any) -> Any:
        """Call price at strike ``K``."""
        return (
            self.discount
            * self.forward
            * np.asarray(self.normalised(np.asarray(K, float) / self.forward))
        )

    def density(self, K: Any) -> Any:
        """Risk-neutral density :math:`\\partial_K^2 C / D = g''(K/F)/F`."""
        return np.asarray(self.normalised(np.asarray(K, float) / self.forward, 2)) / self.forward

    def implied_vol(self, K: Any) -> Any:
        if self.expiry is None:
            raise ValueError("expiry is needed for implied volatilities")
        return implied_black_volatility(self.price(K), self.forward, K, self.expiry, self.discount)

    @property
    def roughness(self) -> float:
        """:math:`\\int g''^2\\,du = \\gamma^\\top R \\gamma` over the interior knots."""
        _, R = spline_matrices(self.knots)
        c = self.gammas[1:-1]
        return float(c @ R @ c)


def fengler_smooth(
    strikes: Sequence[float],
    calls: Sequence[float],
    forward: float,
    discount: float = 1.0,
    expiry: Optional[float] = None,
    smoothing: float = 1e-6,
    weights: Optional[Sequence[float]] = None,
) -> FenglerSmile:
    """Fit Fengler's arbitrage-free natural cubic spline to call prices.

    Parameters
    ----------
    strikes, calls
        Strikes (strictly increasing, at least 3) and call premia.
    forward, discount
        Forward and discount factor of the expiry.
    smoothing
        :math:`\\lambda`, the weight of :math:`\\int g''^2` against the
        squared fit errors in normalised price units.
    weights
        Weights :math:`w_i` of the squared errors (default 1).
    """
    K = np.asarray(strikes, dtype=float)
    C = np.asarray(calls, dtype=float)
    if K.shape != C.shape or K.ndim != 1:
        raise ValueError("strikes and calls must be 1-D arrays of equal length")
    if forward <= 0 or discount <= 0 or smoothing < 0:
        raise ValueError("forward and discount must be positive and smoothing non-negative")
    u, y = K / forward, C / (discount * forward)
    Q, R = spline_matrices(u)
    n = u.size
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or np.any(w <= 0):
        raise ValueError("need one positive weight per strike")
    h = np.diff(u)
    scale = 1.0 / n

    def split(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return z[:n], z[n:]

    def obj(z: np.ndarray) -> float:
        g, c = split(z)
        return scale * (float(w @ (g - y) ** 2) + smoothing * float(c @ R @ c))

    def grad(z: np.ndarray) -> np.ndarray:
        g, c = split(z)
        return scale * np.concatenate([2 * w * (g - y), 2 * smoothing * (R @ c)])

    A_eq = np.hstack([Q.T, -R])
    m = n - 2
    # slopes at the ends of a natural spline: g'(u_1) and g'(u_n)
    left = np.zeros(n + m)
    left[0], left[1], left[n] = -1 / h[0], 1 / h[0], -h[0] / 6
    right = np.zeros(n + m)
    right[n - 2], right[n - 1], right[n + m - 1] = -1 / h[-1], 1 / h[-1], h[-1] / 6
    lower = np.maximum(1 - u, 0.0)
    cons = [
        {"type": "eq", "fun": lambda z: A_eq @ z, "jac": lambda z: A_eq},
        {
            "type": "ineq",
            "fun": lambda z: np.array([left @ z + 1, -(right @ z)]),
            "jac": lambda z: np.vstack([left, -right]),
        },
    ]
    bounds = [(lo, 1.0) for lo in lower] + [(0.0, None)] * m
    # start from the data, projected into the bounds, with consistent gammas
    g0 = np.clip(y, lower, 1.0)
    c0 = np.maximum(np.linalg.solve(R, Q.T @ g0), 0.0)
    res = minimize(
        obj,
        np.concatenate([g0, c0]),
        jac=grad,
        bounds=bounds,
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-16, "maxiter": 1000},
    )
    if not res.success and np.max(np.abs(A_eq @ res.x)) > 1e-9:
        raise RuntimeError(f"Fengler QP failed: {res.message}")
    g, c = split(res.x)
    return FenglerSmile(
        u,
        g,
        np.concatenate([[0.0], np.maximum(c, 0.0), [0.0]]),
        float(forward),
        float(discount),
        expiry,
    )
