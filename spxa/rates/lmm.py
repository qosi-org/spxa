r"""
LIBOR market model (Brace–Gątarek–Musiela, Jamshidian, Miltersen–Sandmann–Sondermann).

Forward rates :math:`L_i(t) = (P(t, T_i)/P(t, T_{i+1}) - 1)/\tau_i` on a
tenor structure :math:`T_0 < T_1 < \dots < T_n` are (displaced) lognormal
under their own forward measures,

.. math::
    d(L_i + \delta) = \sigma_i(t)\,(L_i + \delta)\,dW_i^{T_{i+1}}, \qquad
    d\langle W_i, W_j\rangle = \rho_{ij}\,dt,

so a caplet on :math:`L_i` is a (displaced) Black formula with variance
:math:`\int_0^{T_i}\sigma_i^2\,dt`. The volatilities follow Rebonato's
time-homogeneous abcd form scaled forward by forward,

.. math:: \sigma_i(t) = k_i\,\bigl[(a + b(T_i - t))e^{-c(T_i - t)} + d\bigr],

and the correlation is parametric, :math:`\rho_{ij} = e^{-\beta|T_i - T_j|}`
or Rebonato's :math:`\rho_\infty + (1 - \rho_\infty)e^{-\beta|T_i - T_j|}`,
optionally reduced to a few factors by principal components. Swaptions
are priced by Rebonato's frozen-weights approximation of the swap-rate
volatility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from spxa.market.black import black_price
from spxa.market.curves import YieldCurve

__all__ = [
    "AbcdVol",
    "LiborMarketModel",
    "exponential_correlation",
    "rebonato_correlation",
    "reduce_rank",
]


@dataclass(frozen=True)
class AbcdVol:
    r"""Instantaneous volatility :math:`(a + b\tau)e^{-c\tau} + d` in time to expiry :math:`\tau`."""

    a: float
    b: float
    c: float
    d: float

    def __call__(self, tau: Any) -> Any:
        tau = np.asarray(tau, float)
        return (self.a + self.b * tau) * np.exp(-self.c * tau) + self.d

    def covariance(self, Ti: float, Tj: float, t0: float, t1: float) -> float:
        r""":math:`\int_{t_0}^{t_1} f(T_i - t) f(T_j - t)\,dt` in closed form (Rebonato 2004)."""
        a, b, c, d = self.a, self.b, self.c, self.d

        def F(t: float) -> float:
            ti, tj = Ti - t, Tj - t
            ei, ej = math.exp(-c * ti), math.exp(-c * tj)
            # antiderivative in t of [(a + b ti) e^{-c ti} + d][(a + b tj) e^{-c tj} + d]
            if c == 0.0:
                raise ValueError("c must be non-zero")
            prod = ei * ej
            term1 = (
                prod
                / (4 * c**3)
                * (
                    2 * c * c * (a * a + a * b * (ti + tj) + b * b * ti * tj)
                    + 2 * a * b * c
                    + b * b * (1 + c * (ti + tj))
                )
            )
            term2 = d / (c * c) * ((c * (a + b * ti) + b) * ei + (c * (a + b * tj) + b) * ej)
            return term1 + term2 + d * d * t

        return float(F(t1) - F(t0))


def exponential_correlation(times: Sequence[float], beta: float) -> np.ndarray:
    r""":math:`\rho_{ij} = e^{-\beta|T_i - T_j|}`."""
    T = np.asarray(times, float)
    return np.exp(-beta * np.abs(T[:, None] - T[None, :]))


def rebonato_correlation(times: Sequence[float], beta: float, rho_inf: float) -> np.ndarray:
    r""":math:`\rho_{ij} = \rho_\infty + (1 - \rho_\infty)e^{-\beta|T_i - T_j|}`."""
    return rho_inf + (1 - rho_inf) * exponential_correlation(times, beta)


def reduce_rank(rho: np.ndarray, n_factors: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Rank-``n_factors`` correlation by principal components.

    Keeps the largest eigenpairs, :math:`B = V_k\Lambda_k^{1/2}`, and rescales
    each row to unit length so that :math:`BB^{\mathsf T}` is again a
    correlation matrix. Returns ``(B, B @ B.T)``.
    """
    w, V = np.linalg.eigh(np.asarray(rho, float))
    idx = np.argsort(w)[::-1][: int(n_factors)]
    B = V[:, idx] * np.sqrt(np.maximum(w[idx], 0.0))
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return B, B @ B.T


class LiborMarketModel:
    """Displaced lognormal forward rates on ``times`` (``n + 1`` dates give ``n`` forwards).

    Parameters
    ----------
    curve : YieldCurve
        Initial discount curve (single-curve setting) or anything with
        ``discount(t)``.
    times : increasing floats ``T_0 < ... < T_n``
    vol : AbcdVol
    k : per-forward scalings (default ones)
    correlation : ``(n, n)`` matrix (default perfect correlation)
    shift : displacement :math:`\\delta`
    """

    def __init__(
        self,
        curve: YieldCurve,
        times: Sequence[float],
        vol: AbcdVol,
        k: Optional[Sequence[float]] = None,
        correlation: Optional[np.ndarray] = None,
        shift: float = 0.0,
    ) -> None:
        T = np.asarray(times, float)
        if T.ndim != 1 or T.size < 2 or np.any(np.diff(T) <= 0) or T[0] < 0:
            raise ValueError("times must be increasing, non-negative and at least two")
        self.curve = curve
        self.times = T
        self.tau = np.diff(T)
        self.n = T.size - 1
        self.vol = vol
        self.k = np.ones(self.n) if k is None else np.asarray(k, float)
        if self.k.shape != (self.n,):
            raise ValueError("need one scaling per forward")
        rho = np.ones((self.n, self.n)) if correlation is None else np.asarray(correlation, float)
        if (
            rho.shape != (self.n, self.n)
            or not np.allclose(rho, rho.T)
            or not np.allclose(np.diag(rho), 1)
        ):
            raise ValueError("correlation must be a symmetric unit-diagonal (n, n) matrix")
        self.correlation = rho
        self.shift = float(shift)
        P = np.array([float(curve.discount(t)) for t in T])
        self.discounts = P
        self.forwards = (P[:-1] / P[1:] - 1.0) / self.tau

    def sigma(self, t: float) -> np.ndarray:
        """Instantaneous vols of all forwards at ``t`` (zero once a forward has fixed)."""
        tau = self.times[:-1] - t
        return np.where(tau >= 0, self.k * self.vol(np.maximum(tau, 0.0)), 0.0)

    def covariance(self, t0: float, t1: float) -> np.ndarray:
        r""":math:`C_{ij} = \rho_{ij}\int_{t_0}^{t_1}\sigma_i\sigma_j\,dt` (forwards alive over the interval)."""
        Ti = self.times[:-1]
        C = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                end = min(t1, Ti[i], Ti[j])
                if end > t0:
                    C[i, j] = C[j, i] = (
                        self.k[i] * self.k[j] * self.vol.covariance(Ti[i], Ti[j], t0, end)
                    )
        return C * self.correlation

    def caplet_variance(self, i: int) -> float:
        return float(self.covariance(0.0, self.times[i])[i, i])

    def caplet(self, i: int, K: Any, notional: float = 1.0, option: str = "call") -> Any:
        """Caplet (``call``) or floorlet on forward ``i``, paid at :math:`T_{i+1}`."""
        T = self.times[i]
        if T <= 0:
            raise ValueError("the forward has already fixed")
        v = math.sqrt(self.caplet_variance(i) / T)
        F, d = self.forwards[i] + self.shift, self.shift
        df = self.discounts[i + 1] * self.tau[i] * notional
        return black_price(F, np.asarray(K, float) + d, T, v, df, option)

    def caplet_vol(self, i: int) -> float:
        """Black (shifted) volatility of caplet ``i``."""
        return math.sqrt(self.caplet_variance(i) / self.times[i])

    # ------------------------------------------------------------ swaptions
    def swap_weights(self, alpha: int, beta: int) -> tuple[np.ndarray, float, float]:
        r"""Weights :math:`w_i = \tau_i P(0, T_{i+1})/A` of the forward swap rate on
        :math:`[T_\alpha, T_\beta]`, the swap rate :math:`S = \sum w_i L_i` and the annuity :math:`A`."""
        if not 0 <= alpha < beta <= self.n:
            raise ValueError("need 0 <= alpha < beta <= n")
        idx = slice(alpha, beta)
        pa = self.tau[idx] * self.discounts[alpha + 1 : beta + 1]
        A = float(pa.sum())
        w = pa / A
        return w, float(w @ self.forwards[idx]), A

    def swaption_vol(self, alpha: int, beta: int) -> float:
        r"""Rebonato's frozen-weights Black volatility of the :math:`T_\alpha \times (T_\beta - T_\alpha)` swaption.

        .. math::
            \sigma_{\alpha\beta}^2 T_\alpha = \frac{1}{(S + \delta)^2}\sum_{i,j = \alpha}^{\beta - 1}
            w_i w_j (L_i + \delta)(L_j + \delta)\,\rho_{ij}\int_0^{T_\alpha}\sigma_i\sigma_j\,dt,

        freezing the weights and the forwards at their initial values; the
        displaced swap rate is then treated as lognormal.
        """
        w, S, _ = self.swap_weights(alpha, beta)
        T = self.times[alpha]
        if T <= 0:
            raise ValueError("the swaption has expired")
        idx = slice(alpha, beta)
        x = w * (self.forwards[idx] + self.shift)
        C = self.covariance(0.0, T)[idx, idx]
        return math.sqrt(float(x @ C @ x) / (S + self.shift) ** 2 / T)

    def swaption(
        self, alpha: int, beta: int, K: Any, payer: bool = True, notional: float = 1.0
    ) -> Any:
        """Swaption on the swap over :math:`[T_\\alpha, T_\\beta]` by Rebonato's formula (Black on the annuity)."""
        _, S, A = self.swap_weights(alpha, beta)
        v = self.swaption_vol(alpha, beta)
        d = self.shift
        return black_price(
            S + d,
            np.asarray(K, float) + d,
            self.times[alpha],
            v,
            A * notional,
            "call" if payer else "put",
        )
