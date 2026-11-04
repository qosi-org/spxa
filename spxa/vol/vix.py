r"""
VIX in rough Bergomi and Heston.

The VIX squared at :math:`T` is the forward variance over the next 30 days,

.. math:: \mathrm{VIX}_T^2 = \frac{1}{\Delta}\int_T^{T+\Delta} E_T[v_u]\,du = \frac1\Delta\int_T^{T+\Delta}\xi_T(u)\,du,
    \qquad \Delta = 30/365 .

**Rough Bergomi.** The forward variances are lognormal,

.. math::
    \xi_T(u) = \xi_0(u)\exp\bigl(\eta G_u - \tfrac12\eta^2\operatorname{Var}G_u\bigr), \qquad
    G_u = \sqrt{2H}\int_0^T (u - s)^{H - 1/2}\,dW_s,

and :math:`(G_u)_{u \in [T, T+\Delta]}` is a Gaussian process with

.. math:: \operatorname{Cov}(G_u, G_w) = 2H\int_{u-T}^{u} x^{\alpha}(x + w - u)^{\alpha}\,dx, \qquad \alpha = H - \tfrac12,\ u \le w,

in closed form through :math:`{}_2F_1`. :class:`RBergomiVIX` puts the
:math:`u`-integral on Gauss–Legendre nodes and samples the Gaussian vector
exactly, so the only error in VIX futures and options is Monte Carlo
noise. :math:`E[\mathrm{VIX}_T^2] = \frac1\Delta\int\xi_0` holds exactly, and
VIX futures lie below its square root by Jensen's inequality.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import numpy as np
from scipy.special import hyp2f1

__all__ = ["RBergomiVIX", "VIX_WINDOW"]

VIX_WINDOW = 30.0 / 365.0
Curve = Union[float, Callable[[np.ndarray], np.ndarray]]


def _f(b: np.ndarray, c: np.ndarray, a: float) -> np.ndarray:
    # int_0^b x^a (x + c)^a dx for b >= 0, c > 0
    return b ** (a + 1) * c**a / (a + 1) * hyp2f1(-a, a + 1, a + 2, -b / c)


class RBergomiVIX:
    """VIX futures and options in rough Bergomi with Hurst ``hurst``, vol-of-vol ``eta``
    and forward-variance curve ``xi`` (a number or a function of time)."""

    def __init__(
        self, hurst: float, eta: float, xi: Curve, delta: float = VIX_WINDOW, n_nodes: int = 24
    ) -> None:
        if not 0 < hurst < 0.5:
            raise ValueError("need 0 < H < 1/2")
        self.hurst, self.eta, self.delta, self.n_nodes = (
            float(hurst),
            float(eta),
            float(delta),
            int(n_nodes),
        )
        self._xi = xi

    def xi(self, t: Any) -> np.ndarray:
        tt = np.asarray(t, float)
        return (
            np.full(tt.shape, float(self._xi))
            if np.isscalar(self._xi)
            else np.asarray(self._xi(tt), float)
        )  # type: ignore[operator, arg-type]

    def nodes(self, T: float) -> tuple[np.ndarray, np.ndarray]:
        """Gauss–Legendre nodes on :math:`[T, T + \\Delta]` and weights normalised to sum to one."""
        x, w = np.polynomial.legendre.leggauss(self.n_nodes)
        return T + 0.5 * self.delta * (x + 1.0), 0.5 * w

    def covariance(self, T: float) -> np.ndarray:
        r""":math:`\operatorname{Cov}(G_{u_i}, G_{u_j})` at the nodes."""
        u, _ = self.nodes(T)
        a, H = self.hurst - 0.5, self.hurst
        lo, hi = np.minimum.outer(u, u), np.maximum.outer(u, u)
        c = hi - lo
        C = np.empty_like(lo)
        diag = c < 1e-14
        C[diag] = (lo[diag] ** (2 * H) - (lo[diag] - T) ** (2 * H)) / (2 * H)
        off = ~diag
        C[off] = _f(lo[off], c[off], a) - _f(lo[off] - T, c[off], a)
        return 2 * H * C

    def expected_vix_squared(self, T: float) -> float:
        r""":math:`E[\mathrm{VIX}_T^2] = \frac1\Delta\int_T^{T+\Delta}\xi_0(u)\,du` (by quadrature)."""
        u, w = self.nodes(T)
        return float(w @ self.xi(u))

    def vix_squared(self, T: float, n_paths: int, seed: Optional[int] = None) -> np.ndarray:
        """Exact samples of :math:`\\mathrm{VIX}_T^2` (antithetic)."""
        u, w = self.nodes(T)
        C = self.covariance(T)
        lam, V = np.linalg.eigh(C)
        A = V * np.sqrt(np.maximum(lam, 0.0))
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(((n_paths + 1) // 2, u.size))
        z = np.concatenate([z, -z])[:n_paths]
        G = z @ A.T
        xi_T = self.xi(u) * np.exp(self.eta * G - 0.5 * self.eta**2 * np.diag(C))
        return xi_T @ w

    def futures(
        self, T: float, n_paths: int = 200_000, seed: Optional[int] = None
    ) -> tuple[float, float]:
        r""":math:`E[\mathrm{VIX}_T]` and its standard error."""
        v = np.sqrt(self.vix_squared(T, n_paths, seed))
        return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size))

    def lognormal_moments(self, T: float) -> tuple[float, float]:
        r"""Mean and log-variance :math:`s^2` of the lognormal matching the first two moments of :math:`\mathrm{VIX}_T^2`.

        :math:`E[\mathrm{VIX}^4] = \sum_{ij} w_i w_j \xi_0(u_i)\xi_0(u_j)e^{\eta^2 C_{ij}}`.
        """
        u, w = self.nodes(T)
        x = w * self.xi(u)
        m1 = float(x.sum())
        m2 = float(x @ np.exp(self.eta**2 * self.covariance(T)) @ x)
        return m1, math.log(m2 / (m1 * m1))

    def futures_lognormal(self, T: float) -> float:
        r"""Lognormal approximation :math:`E[\mathrm{VIX}_T] \approx \sqrt{m_1}\,e^{-s^2/8}`."""
        m1, s2 = self.lognormal_moments(T)
        return math.sqrt(m1) * math.exp(-s2 / 8)
