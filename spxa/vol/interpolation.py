r"""Arbitrage-free interpolation of implied-volatility slices in time.

Gatheral & Jacquier (2014, §5.3) interpolate *prices*, not variances.
Given slices at expiries :math:`t_1 < \dots < t_n` with strictly
increasing ATM total variances :math:`\theta_i = w_i(0)`, choose an
increasing ATM curve :math:`\theta_t` (here linear in :math:`t` between
the nodes, from :math:`\theta_0 = 0` at :math:`t_0 = 0`) and set, for
:math:`t_i \le t \le t_{i+1}`,

.. math:: c(k, t) = \alpha_t\,c_i(k) + (1 - \alpha_t)\,c_{i+1}(k), \qquad
          \alpha_t = \frac{\sqrt{\theta_{i+1}} - \sqrt{\theta_t}}
                          {\sqrt{\theta_{i+1}} - \sqrt{\theta_i}},

where :math:`c_i(k)` is the normalised undiscounted call price of slice
:math:`i` at log-moneyness :math:`k` and :math:`c_0(k) = (1 - e^k)^+`.

* **No butterfly arbitrage.** :math:`c(\cdot, t)` is a convex combination of
  call-price functions, so it is the call-price function of the mixture of
  the two risk-neutral laws. It is convex and decreasing in strike, its
  density is :math:`\alpha_t p_i + (1 - \alpha_t)p_{i+1}`, and the forward
  is preserved.
* **No calendar arbitrage.** If the slices are calendar-free,
  :math:`c_i \le c_{i+1}` pointwise. Since :math:`\alpha_t` decreases in
  :math:`t`, :math:`c(k, t)` is non-decreasing in :math:`t` for every
  :math:`k`, within and across buckets.
* **ATM level.** In :math:`s = \sqrt\theta` the ATM price is
  :math:`2\Phi(s/2) - 1 = (s - s^3/24 + O(s^5))/\sqrt{2\pi}`, so weights
  linear in :math:`s` reproduce it up to the interpolation error of
  :math:`s^3`, which is exactly
  :math:`(s - s_i)(s_{i+1} - s)(s_i + s + s_{i+1})`. The interpolated ATM
  total variance therefore falls short of :math:`\theta_t` by

  .. math:: \theta_t - w(0, t) = \frac{s}{12}(s - s_i)(s_{i+1} - s)(s_i + s + s_{i+1})
            \,\bigl(1 + O(\theta_{i+1})\bigr), \qquad s = \sqrt{\theta_t}.

Between :math:`0` and :math:`t_1` the mixture is with the point mass at the
forward, so the density has an atom at :math:`k = 0` for
:math:`t < t_1`. Beyond :math:`t_n` no extrapolation is attempted.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from spxa.market.black import black_price
from spxa.market.implied import implied_black_volatility
from spxa.vol.arbitrage import risk_neutral_density

__all__ = ["PriceInterpolatedSurface"]


def _w(slice_: Any, k: np.ndarray) -> np.ndarray:
    f = slice_.total_variance if hasattr(slice_, "total_variance") else slice_
    return np.asarray(f(k), float)


def _otm(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Normalised out-of-the-money price: call for k >= 0, put for k < 0."""
    opt = np.where(k >= 0, 1.0, -1.0)
    return np.asarray(black_price(1.0, np.exp(k), 1.0, np.sqrt(w), 1.0, opt), float)


class PriceInterpolatedSurface:
    r"""Implied-volatility surface from slices, interpolated in price (Gatheral–Jacquier).

    Parameters
    ----------
    expiries : sequence of float
        Strictly increasing positive expiries :math:`t_i`.
    slices : sequence
        Slice objects with ``total_variance(k)`` (or callables ``w(k)``),
        assumed free of butterfly and calendar arbitrage.
    """

    def __init__(self, expiries: Sequence[float], slices: Sequence[Any]) -> None:
        T = np.asarray(expiries, float)
        if T.ndim != 1 or T.size == 0 or T[0] <= 0 or np.any(np.diff(T) <= 0):
            raise ValueError("expiries must be positive and strictly increasing")
        if len(slices) != T.size:
            raise ValueError("need one slice per expiry")
        th = np.array([float(_w(s, np.array(0.0))) for s in slices])
        if th[0] <= 0 or np.any(np.diff(th) <= 0):
            raise ValueError("ATM total variances must be positive and strictly increasing")
        self.expiries = T
        self.slices = list(slices)
        self.thetas = th
        self._t = np.concatenate([[0.0], T])
        self._th = np.concatenate([[0.0], th])

    # ---------------------------------------------------------------- weights
    def theta(self, t: float) -> float:
        r"""ATM total variance :math:`\theta_t`, linear in :math:`t` between nodes."""
        t = float(t)
        if not 0.0 <= t <= self.expiries[-1] * (1 + 1e-14):
            raise ValueError(f"t = {t} outside [0, {self.expiries[-1]}]: no extrapolation")
        return float(np.interp(t, self._t, self._th))

    def bucket(self, t: float) -> tuple[int, float]:
        r"""``(i, alpha)``: :math:`t` lies in :math:`[t_i, t_{i+1}]` (``i = 0`` is :math:`t = 0`)."""
        th = self.theta(t)
        i = int(np.searchsorted(self._t, float(t), side="right")) - 1
        i = min(max(i, 0), self._t.size - 2)
        lo, hi = np.sqrt(self._th[i]), np.sqrt(self._th[i + 1])
        alpha = float((hi - np.sqrt(th)) / (hi - lo))
        return i, min(max(alpha, 0.0), 1.0)

    def _slice_otm(self, j: int, k: np.ndarray) -> np.ndarray:
        if j == 0:
            return np.zeros_like(k)
        return _otm(k, _w(self.slices[j - 1], k))

    # ----------------------------------------------------------------- prices
    def otm_price(self, k: Any, t: float) -> Any:
        """Normalised undiscounted out-of-the-money price (put for ``k < 0``)."""
        kk = np.asarray(k, float)
        i, a = self.bucket(t)
        p = a * self._slice_otm(i, kk) + (1.0 - a) * self._slice_otm(i + 1, kk)
        return float(p) if p.ndim == 0 else p

    def call_price(self, k: Any, t: float) -> Any:
        """Normalised undiscounted call price :math:`c(k, t) = C/(P F)`."""
        kk = np.asarray(k, float)
        p = np.asarray(self.otm_price(kk, t), float)
        c = np.where(kk >= 0, p, p + 1.0 - np.exp(kk))
        return float(c) if c.ndim == 0 else c

    def total_variance(self, k: Any, t: float) -> Any:
        """Implied total variance of :meth:`otm_price` (exact at the nodes)."""
        kk = np.asarray(k, float)
        i, a = self.bucket(t)
        if a == 1.0 and i > 0:
            return _w(self.slices[i - 1], kk) if kk.ndim else float(_w(self.slices[i - 1], kk))
        if a == 0.0:
            out = _w(self.slices[i], kk)
            return out if kk.ndim else float(out)
        opt = np.where(kk >= 0, 1.0, -1.0)
        vol = implied_black_volatility(self.otm_price(kk, t), 1.0, np.exp(kk), 1.0, 1.0, opt)
        w = np.asarray(vol, float) ** 2
        return float(w) if w.ndim == 0 else w

    def implied_vol(self, k: Any, t: float) -> Any:
        if t <= 0:
            raise ValueError("t must be positive")
        w = np.asarray(self.total_variance(k, t), float)
        v = np.sqrt(w / t)
        return float(v) if v.ndim == 0 else v

    def density(self, k: Any, t: float) -> Any:
        r"""Mixture density :math:`\alpha_t p_i + (1 - \alpha_t)p_{i+1}` of :math:`\log(S_t/F_t)`.

        For :math:`t < t_1` this is the absolutely continuous part only;
        the remaining mass :math:`\alpha_t` sits at :math:`k = 0`.
        """
        kk = np.asarray(k, float)
        i, a = self.bucket(t)
        hi = np.asarray(risk_neutral_density(self.slices[i], kk), float)
        lo = 0.0 if i == 0 else np.asarray(risk_neutral_density(self.slices[i - 1], kk), float)
        p = a * lo + (1.0 - a) * hi
        return float(p) if np.ndim(p) == 0 else p
