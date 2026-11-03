r"""Risk-neutral distribution implied by a total-variance slice.

Let :math:`X = \log(S_T/F_T)` and let :math:`c(k)` be the normalised
undiscounted call price at log-moneyness :math:`k` with total variance
:math:`w(k)`. Writing :math:`d_\pm = -k/\sqrt w \pm \sqrt w/2` and
differentiating :math:`c(k) = \Phi(d_+) - e^k\Phi(d_-)` along the slice
(Breeden–Litzenberger), the survival function and the density of
:math:`X` are

.. math::
    \mathbb{Q}(X > k) = \Phi(d_-) - \varphi(d_-)\,\frac{w'(k)}{2\sqrt{w}},
    \qquad
    p(k) = \frac{g(k)}{\sqrt{2\pi w}}\,e^{-d_-^2/2},

where :math:`g` is Durrleman's function. The first identity is the price of
a normalised digital call; for a flat slice it reduces to the lognormal
:math:`\Phi(d_-)`. Both are exact in :math:`w, w', w''`; only the moments,
quantiles and samples below involve quadrature or root finding.

A slice defines a probability distribution exactly when it is free of
butterfly arbitrage and its call prices vanish in the right wing (Gatheral &
Jacquier 2014, Lemma 2.2). :class:`ImpliedDistribution` does not enforce
this; :attr:`ImpliedDistribution.butterfly_free` reports it, and
:meth:`ImpliedDistribution.quantile` and
:meth:`ImpliedDistribution.sample` refuse slices that fail.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr

from spxa.vol.arbitrage import _derivatives, check_butterfly, risk_neutral_density

__all__ = ["ImpliedDistribution"]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _out(x: np.ndarray) -> Any:
    return float(x) if np.ndim(x) == 0 else x


class ImpliedDistribution:
    r"""Distribution of :math:`\log(S_T/F_T)` implied by a slice :math:`w(k)`.

    Parameters
    ----------
    slice_
        Object with ``total_variance(k)`` (and optionally ``dw``/``d2w``), or
        a callable ``w(k)``; derivatives fall back to central differences.
    k_min, k_max
        Truncation of the log-moneyness axis used by :meth:`expectation`,
        :meth:`quantile` and :meth:`sample`. The default is
        smallest of :math:`\mp(12\sqrt{w(0)} + 1)\cdot 1.25^j` (capped at
        400) beyond which the exact tail mass :math:`\mathbb{Q}(X < k_{\min})`,
        respectively the share-weighted mass
        :math:`e^{k_{\max}}\,\mathbb{Q}(X > k_{\max})`, is below
        :math:`10^{-17}`.
    """

    def __init__(self, slice_: Any, k_min: float | None = None, k_max: float | None = None) -> None:
        self.slice = slice_
        w0 = float(self._w(np.asarray(0.0)))
        if w0 <= 0:
            raise ValueError("total variance at the money must be positive")
        start = 12.0 * np.sqrt(w0) + 1.0
        self.k_min = self._edge(-start, left=True) if k_min is None else float(k_min)
        self.k_max = self._edge(start, left=False) if k_max is None else float(k_max)
        if not self.k_min < self.k_max:
            raise ValueError("k_min must be below k_max")
        self.butterfly_free = bool(
            check_butterfly(slice_, np.linspace(self.k_min, self.k_max, 4001))
        )

    def _edge(self, k: float, left: bool, tol: float = 1e-17, cap: float = 400.0) -> float:
        """Widen ``k`` until the exact tail mass beyond it (share-weighted on the right) < tol."""
        while abs(k) < cap:
            tail = abs(float(self.cdf(k))) if left else abs(float(self.survival(k))) * np.exp(k)
            if tail < tol:
                break
            k *= 1.25
        return float(np.clip(k, -cap, cap))

    def _w(self, k: np.ndarray) -> np.ndarray:
        f = self.slice.total_variance if hasattr(self.slice, "total_variance") else self.slice
        return np.asarray(f(k), float)

    # ------------------------------------------------------------------ exact
    def pdf(self, k: Any) -> Any:
        """Density of the log-moneyness :math:`X` at ``k``."""
        return risk_neutral_density(self.slice, k)

    def survival(self, k: Any) -> Any:
        r""":math:`\mathbb{Q}(X > k)`, the normalised undiscounted digital call."""
        kk = np.asarray(k, float)
        w, dw, _ = _derivatives(self.slice, kk)
        if np.any(w <= 0):
            raise ValueError("total variance must be positive")
        sw = np.sqrt(w)
        dm = -kk / sw - 0.5 * sw
        s = ndtr(dm) - np.exp(-0.5 * dm * dm) / _SQRT_2PI * dw / (2.0 * sw)
        return _out(s)

    def cdf(self, k: Any) -> Any:
        r""":math:`\mathbb{Q}(X \le k)`."""
        return _out(1.0 - np.asarray(self.survival(k), float))

    def digital_put(self, k: Any) -> Any:
        """Normalised undiscounted digital put, equal to :meth:`cdf`."""
        return self.cdf(k)

    def digital_call(self, k: Any) -> Any:
        """Normalised undiscounted digital call, equal to :meth:`survival`."""
        return self.survival(k)

    # -------------------------------------------------------------- numerical
    def expectation(self, f: Callable[[float], float], points: Any = None) -> float:
        r""":math:`\int f(k)\,p(k)\,dk` over ``[k_min, k_max]`` by adaptive quadrature.

        ``points`` are passed to :func:`scipy.integrate.quad` (kinks of ``f``).
        """
        pts = None if points is None else list(np.atleast_1d(np.asarray(points, float)))
        val, _ = quad(
            lambda x: f(x) * float(self.pdf(x)),
            self.k_min,
            self.k_max,
            points=pts,
            limit=400,
            epsabs=1e-13,
            epsrel=1e-12,
        )
        return float(val)

    def total_mass(self) -> float:
        """:math:`\\int p` on the truncated axis (one for an arbitrage-free slice)."""
        return self.expectation(lambda x: 1.0, points=[0.0])

    def forward_ratio(self) -> float:
        r""":math:`\mathbb{E}[S_T/F_T] = \mathbb{E}[e^X]` (one for a martingale)."""
        return self.expectation(np.exp, points=[0.0])

    def moment(self, n: int) -> float:
        r"""Raw moment :math:`\mathbb{E}[X^n]` of the log-moneyness."""
        return self.expectation(lambda x: x**n, points=[0.0])

    def call(self, k: float) -> float:
        r""":math:`\mathbb{E}[(e^X - e^k)^+]` by quadrature (reprices the slice)."""
        return self.expectation(lambda x: max(np.exp(x) - np.exp(k), 0.0), points=[k])

    def quantile(self, u: Any) -> Any:
        r"""Inverse of :meth:`cdf` on ``[k_min, k_max]`` by Brent's method."""
        if not self.butterfly_free:
            raise ValueError("slice has butterfly arbitrage: its cdf is not monotone")
        uu = np.asarray(u, float)
        if np.any((uu <= 0) | (uu >= 1)):
            raise ValueError("probabilities must lie in (0, 1)")
        lo, hi = float(self.cdf(self.k_min)), float(self.cdf(self.k_max))

        def one(p: float) -> float:
            if p <= lo:
                return self.k_min
            if p >= hi:
                return self.k_max
            return float(
                brentq(lambda x: float(self.cdf(x)) - p, self.k_min, self.k_max, xtol=1e-14)
            )

        return _out(np.vectorize(one, otypes=[float])(uu))

    def sample(self, n: int, rng: Any = None, grid: int = 4001) -> np.ndarray:
        r"""``n`` draws of :math:`X` by inverse transform on a ``grid``-point cdf table.

        The cdf is tabulated exactly on a uniform grid of ``[k_min, k_max]``
        and inverted by linear interpolation, so the error is the
        interpolation error of the quantile function, :math:`O(h^2)`.
        """
        if not self.butterfly_free:
            raise ValueError("slice has butterfly arbitrage: its cdf is not monotone")
        gen = np.random.default_rng(rng)
        x = np.linspace(self.k_min, self.k_max, grid)
        F = np.maximum.accumulate(np.asarray(self.cdf(x), float))
        u = gen.uniform(F[0], F[-1], size=n)
        return np.asarray(np.interp(u, F, x))
