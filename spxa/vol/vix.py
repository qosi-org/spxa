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

**Heston.** :math:`E_T[v_u] = \theta + (v_T - \theta)e^{-\kappa(u-T)}`, so
:math:`\mathrm{VIX}_T^2 = a v_T + b` with :math:`a = (1 - e^{-\kappa\Delta})/(\kappa\Delta)`
and :math:`b = \theta(1 - a)`, where :math:`v_T` is a scaled noncentral
:math:`\chi^2`. :class:`HestonVIX` prices futures and options by one-dimensional
quadrature against that density.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import numpy as np
from scipy.special import hyp2f1
from scipy.stats import ncx2

from spxa.market import black_price, implied_black_volatility
from spxa.vol.heston import Heston

__all__ = ["HestonVIX", "RBergomiVIX", "VIX_WINDOW"]

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
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, u.size))
        xi_u, comp = self.xi(u), 0.5 * self.eta**2 * np.diag(C)
        out = np.empty(n_paths)
        for lo in range(0, half, 16384):  # blocks bound the memory of G
            G = z[lo : lo + 16384] @ A.T
            m = G.shape[0]
            out[lo : lo + m] = (xi_u * np.exp(self.eta * G - comp)) @ w
            neg = out[half + lo : min(half + lo + m, n_paths)]
            neg[:] = ((xi_u * np.exp(-self.eta * G - comp)) @ w)[: neg.size]
        return out

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

    def options(
        self,
        T: float,
        K: Any,
        option: str = "call",
        n_paths: int = 200_000,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Undiscounted VIX option prices at strikes ``K``, their standard errors, and the futures price.

        Prices use the futures as a control variate (exact, since futures are priced on
        the same sample), so put-call parity :math:`C - P = F - K` holds sample by sample.
        """
        vix = np.sqrt(self.vix_squared(T, n_paths, seed))
        Ks = np.atleast_1d(np.asarray(K, float))
        eta = 1.0 if option == "call" else -1.0
        pay = np.maximum(eta * (vix[:, None] - Ks[None, :]), 0.0)
        return pay.mean(0), pay.std(0, ddof=1) / math.sqrt(vix.size), float(vix.mean())

    def implied_vols(
        self, T: float, K: Any, n_paths: int = 200_000, seed: Optional[int] = None
    ) -> np.ndarray:
        """Black implied volatilities of VIX options on the futures (out-of-the-money side)."""
        Ks = np.atleast_1d(np.asarray(K, float))
        calls, _, fut = self.options(T, Ks, "call", n_paths, seed)
        kinds = np.where(Ks >= fut, "call", "put")
        otm = np.where(Ks >= fut, calls, calls - (fut - Ks))
        return np.asarray(implied_black_volatility(otm, fut, Ks, T, 1.0, kinds), float)

    def options_lognormal(self, T: float, K: Any, option: str = "call") -> np.ndarray:
        r"""Black prices with the lognormal moments: futures :math:`\sqrt{m_1}e^{-s^2/8}` and total variance :math:`s^2/4`."""
        m1, s2 = self.lognormal_moments(T)
        fut = math.sqrt(m1) * math.exp(-s2 / 8)
        return np.asarray(
            black_price(fut, np.asarray(K, float), T, math.sqrt(s2 / 4 / T), 1.0, option), float
        )


class HestonVIX:
    """VIX futures and options in the Heston model (closed form up to one quadrature)."""

    def __init__(self, model: Heston, delta: float = VIX_WINDOW) -> None:
        self.model, self.delta = model, float(delta)
        k = model.kappa * self.delta
        self.a = -math.expm1(-k) / k
        self.b = model.theta * (1.0 - self.a)

    def variance_law(self, T: float) -> tuple[float, float, float]:
        r"""``(c, d, lam)`` with :math:`v_T = c\,\chi'^2_d(\lambda)`."""
        m = self.model
        e = math.exp(-m.kappa * T)
        c = m.xi**2 * (1.0 - e) / (4.0 * m.kappa)
        return c, 4.0 * m.kappa * m.theta / m.xi**2, m.v0 * e / c

    def expected_vix_squared(self, T: float) -> float:
        return float(self.a * self.model.mean_variance(T) + self.b)

    def _expect(self, T: float, g: Callable[[np.ndarray], np.ndarray], lower: float = 0.0) -> float:
        # E[g(x); x > lower] for x ~ chi'^2_d(lam). The mass below x0 = 1e-13 hi is lumped at x0
        # (g is Lipschitz there); above it, x = y^p with p d/2 >= 2 tames the x^{d/2-1}
        # singularity, and composite Gauss-Legendre in y runs up to mean + 14 sd + 80.
        c, d, lam = self.variance_law(T)
        hi = d + lam + 14.0 * math.sqrt(2.0 * (d + 2.0 * lam)) + 80.0  # density tail ~ e^{-x/2}
        if lower >= hi:
            return 0.0
        x0 = 1e-13 * hi
        head = 0.0
        if lower < x0:
            head = float(ncx2.cdf(x0, d, lam) - ncx2.cdf(lower, d, lam)) * float(
                g(np.array([x0]))[0]
            )
            lower = x0
        p = min(max(1.0, 4.0 / d), 60.0)
        edges = np.linspace(lower ** (1 / p), hi ** (1 / p), 33)
        x, w = np.polynomial.legendre.leggauss(24)
        mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
        y = (mid[:, None] + half[:, None] * x[None, :]).ravel()
        wy = (half[:, None] * w[None, :]).ravel()
        xs = y**p
        dens = ncx2.pdf(xs, d, lam) * p * y ** (p - 1)
        return head + float(np.sum(wy * dens * g(xs)))

    def futures(self, T: float) -> float:
        r""":math:`E[\mathrm{VIX}_T] = E\sqrt{a v_T + b}`."""
        c = self.variance_law(T)[0]
        return self._expect(T, lambda x: np.sqrt(self.a * c * x + self.b))

    def options(self, T: float, K: Any, option: str = "call") -> np.ndarray:
        """Undiscounted VIX option prices; puts by parity on the futures."""
        c = self.variance_law(T)[0]
        Ks = np.atleast_1d(np.asarray(K, float))
        calls = np.array(
            [
                self._expect(
                    T,
                    lambda x, k=k: np.sqrt(self.a * c * x + self.b) - k,
                    max((k * k - self.b) / (self.a * c), 0.0),
                )
                for k in Ks
            ]
        )
        return calls if option == "call" else calls - (self.futures(T) - Ks)

    def implied_vols(self, T: float, K: Any) -> np.ndarray:
        """Black implied volatilities of VIX options on the futures."""
        Ks = np.atleast_1d(np.asarray(K, float))
        fut = self.futures(T)
        calls = self.options(T, Ks)
        kinds = np.where(Ks >= fut, "call", "put")
        otm = np.where(Ks >= fut, calls, calls - (fut - Ks))
        return np.asarray(implied_black_volatility(otm, fut, Ks, T, 1.0, kinds), float)
