r"""Dupire local volatility from an implied-volatility surface.

In log-forward moneyness :math:`y = \log(K/F_T)` and total implied
variance :math:`w(y, T)`, Dupire's formula takes the form (Gatheral, *The
Volatility Surface*, eq. 1.10)

.. math::
    \sigma^2_{\mathrm{loc}}(K, T) = \frac{\partial_T w}
    {1 - \dfrac{y}{w}\,\partial_y w + \dfrac14\Bigl(-\dfrac14 - \dfrac1w + \dfrac{y^2}{w^2}\Bigr)(\partial_y w)^2
    + \dfrac12\,\partial_{yy} w}.

Rates and dividend yields drop out because the strike is measured against
the forward, so the same :math:`\sigma_{\mathrm{loc}}(K, T)` drives
:math:`dS_t = (r_t - q_t)S_t\,dt + \sigma_{\mathrm{loc}}(S_t, t)S_t\,dW_t`
for any deterministic :math:`r_t, q_t`.

The denominator is Durrleman's :math:`g(y)` (it expands to the same
expression), so it is positive exactly where the slice has no butterfly
arbitrage. The numerator is the forward implied variance at fixed
moneyness, non-negative exactly where there is no calendar arbitrage.
Points that violate either condition have no local volatility and are
returned as ``nan``.

Derivatives are five-point central differences of the surface's
``total_variance``. Any :class:`~spxa.vol.surface.VolSurface` works,
including price-interpolated ones; at an expiry node of a
piecewise-in-time surface the time derivative averages the two sides.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["LocalVolSurface", "dupire_local_variance"]


def dupire_local_variance(y: Any, w: Any, w_y: Any, w_yy: Any, w_T: Any) -> Any:
    """Gatheral's form of Dupire's formula; ``nan`` where it has no positive solution."""
    y, w, w_y, w_yy, w_T = (np.asarray(a, float) for a in (y, w, w_y, w_yy, w_T))
    den = 1.0 - y / w * w_y + 0.25 * (-0.25 - 1.0 / w + y * y / (w * w)) * w_y**2 + 0.5 * w_yy
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((den > 0) & (w_T >= 0), w_T / den, np.nan)
    return float(out) if out.ndim == 0 else out


def _d5(f: Any, x: np.ndarray, h: float, order: int) -> np.ndarray:
    """Five-point central first or second derivative of ``f`` at ``x``."""
    fm2, fm1, fp1, fp2 = (np.asarray(f(x + s * h), float) for s in (-2, -1, 1, 2))
    if order == 1:
        return (fm2 - 8 * fm1 + 8 * fp1 - fp2) / (12 * h)
    f0 = np.asarray(f(x), float)
    return (-fm2 + 16 * fm1 - 30 * f0 + 16 * fp1 - fp2) / (12 * h * h)


class LocalVolSurface:
    """Dupire local volatility of a :class:`~spxa.vol.surface.VolSurface`.

    Parameters
    ----------
    surface
        The implied-volatility surface (its forward curve fixes
        :math:`F_T`).
    dk, dt_rel
        Steps of the moneyness and (relative) time differences.
    """

    def __init__(self, surface: Any, dk: float = 1e-3, dt_rel: float = 1e-3) -> None:
        self.surface = surface
        self.dk = float(dk)
        self.dt_rel = float(dt_rel)

    def terms(self, y: Any, T: float) -> dict[str, np.ndarray]:
        """``w``, ``w_y``, ``w_yy``, ``w_T`` and the denominator ``g`` at moneyness ``y``."""
        if T <= 0:
            raise ValueError("local volatility needs T > 0")
        yy = np.asarray(y, float)
        s = self.surface
        w = np.asarray(s.total_variance(yy, T), float)
        w_y = _d5(lambda k: s.total_variance(k, T), yy, self.dk, 1)
        w_yy = _d5(lambda k: s.total_variance(k, T), yy, self.dk, 2)
        h = self.dt_rel * T
        w_T = _d5(lambda t: s.total_variance(yy, float(t)), np.asarray(T, float), h, 1)
        g = 1.0 - yy / w * w_y + 0.25 * (-0.25 - 1.0 / w + yy * yy / (w * w)) * w_y**2 + 0.5 * w_yy
        return {"w": w, "w_y": w_y, "w_yy": w_yy, "w_T": np.asarray(w_T, float), "g": g}

    def local_variance_k(self, y: Any, T: float) -> Any:
        """:math:`\\sigma^2_{\\mathrm{loc}}` at log-forward moneyness ``y``."""
        t = self.terms(y, T)
        return dupire_local_variance(y, t["w"], t["w_y"], t["w_yy"], t["w_T"])

    def local_vol_k(self, y: Any, T: float) -> Any:
        return np.sqrt(self.local_variance_k(y, T))

    def local_vol(self, K: Any, T: float) -> Any:
        """Local volatility at strike(s) ``K`` and maturity ``T``."""
        y = np.log(np.asarray(K, float) / float(self.surface.forward(T)))
        return self.local_vol_k(y, T)

    def __call__(self, S: Any, t: float) -> Any:
        """``sigma(S, t)`` for the PDE solvers of :mod:`spxa.pde` (spot level, calendar time)."""
        return self.local_vol(S, max(float(t), 1e-8))
