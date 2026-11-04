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

__all__ = ["LocalVolGrid", "LocalVolSurface", "dupire_local_variance"]


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

    def __init__(
        self,
        surface: Any,
        dk: float = 1e-3,
        dt_rel: float = 1e-3,
        y_range: tuple[float, float] = (-4.0, 4.0),
        t_min: float = 1e-4,
    ) -> None:
        self.surface = surface
        self.dk = float(dk)
        self.dt_rel = float(dt_rel)
        self.y_range = (float(y_range[0]), float(y_range[1]))
        self.t_min = float(t_min)

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
        """``sigma(S, t)`` for the solvers of :mod:`spxa.pde` (spot level, calendar time).

        Grids reach far beyond the quoted range, so the moneyness is clipped
        to ``y_range`` (flat extrapolation of the local volatility) and the
        time to at least ``t_min``. At :math:`S = 0` the volatility is that
        at the lower clip.
        """
        T = max(float(t), self.t_min)
        F = float(self.surface.forward(T))
        SS = np.asarray(S, float)
        with np.errstate(divide="ignore"):
            y = np.log(np.maximum(SS, 0.0) / F)
        return self.local_vol_k(np.clip(y, *self.y_range), T)

    def dupire_pde(self, r: float, q: float = 0.0) -> Any:
        """:class:`spxa.pde.DupirePDE` driven by this local volatility (constant ``r``, ``q``)."""
        from spxa.pde import DupirePDE

        return DupirePDE(r, self, q)

    def black_scholes_pde(self, r: float, q: float = 0.0) -> Any:
        """:class:`spxa.pde.BlackScholesPDE` with this local volatility (backward pricing)."""
        from spxa.pde import BlackScholesPDE

        return BlackScholesPDE(r, self, q)


class LocalVolGrid:
    """Local variance tabulated on a (moneyness, time) grid for fast lookups.

    :math:`\\sigma^2` is bilinear in :math:`(y, t)` on the grid, with
    :math:`y = \\log(S/F_t)`, flat beyond its edges, and clipped to
    ``[floor**2, cap**2]``. Missing values (``nan`` where the source surface
    has arbitrage) are filled from the nearest valid value in moneyness at
    the same time, and an all-``nan`` row raises. Bilinear interpolation in
    the variance keeps it non-negative, and a convex combination never
    leaves the range of the grid values.
    """

    def __init__(
        self,
        ys: Any,
        ts: Any,
        variance: Any,
        forward: Any,
        floor: float = 0.01,
        cap: float = 3.0,
    ) -> None:
        self.ys = np.asarray(ys, float)
        self.ts = np.asarray(ts, float)
        v = np.array(variance, float)
        if v.shape != (self.ts.size, self.ys.size):
            raise ValueError("variance must have shape (len(ts), len(ys))")
        if np.any(np.diff(self.ys) <= 0) or np.any(np.diff(self.ts) <= 0):
            raise ValueError("grids must be strictly increasing")
        for i in range(v.shape[0]):
            ok = np.isfinite(v[i])
            if not ok.any():
                raise ValueError(f"no valid local variance at t = {self.ts[i]}")
            if not ok.all():
                idx = np.flatnonzero(ok)
                near = idx[np.abs(np.arange(v.shape[1])[:, None] - idx[None, :]).argmin(axis=1)]
                v[i] = v[i, near]
        self.variance = np.clip(v, floor**2, cap**2)
        self.forward = forward
        self.floor, self.cap = float(floor), float(cap)

    @classmethod
    def from_surface(
        cls, lv: "LocalVolSurface", ys: Any, ts: Any, floor: float = 0.01, cap: float = 3.0
    ) -> "LocalVolGrid":
        """Tabulate a :class:`LocalVolSurface` (times must be positive)."""
        ts = np.asarray(ts, float)
        var = np.vstack(
            [np.asarray(lv.local_variance_k(np.asarray(ys, float), t), float) for t in ts]
        )
        return cls(ys, ts, var, lv.surface.forward, floor, cap)

    def local_variance_k(self, y: Any, t: float) -> Any:
        yy = np.clip(np.asarray(y, float), self.ys[0], self.ys[-1])
        tt = min(max(float(t), self.ts[0]), self.ts[-1])
        j = (
            int(np.clip(np.searchsorted(self.ts, tt) - 1, 0, self.ts.size - 2))
            if self.ts.size > 1
            else 0
        )
        if self.ts.size == 1:
            row = self.variance[0]
        else:
            a = (tt - self.ts[j]) / (self.ts[j + 1] - self.ts[j])
            row = (1 - a) * self.variance[j] + a * self.variance[j + 1]
        out = np.interp(yy, self.ys, row)
        return float(out) if np.ndim(out) == 0 else out

    def local_vol_k(self, y: Any, t: float) -> Any:
        return np.sqrt(self.local_variance_k(y, t))

    def __call__(self, S: Any, t: float) -> Any:
        """``sigma(S, t)`` for the PDE and Monte Carlo engines."""
        F = float(self.forward(max(float(t), self.ts[0])))
        with np.errstate(divide="ignore"):
            y = np.log(np.maximum(np.asarray(S, float), 0.0) / F)
        return self.local_vol_k(np.maximum(y, self.ys[0]), t)
