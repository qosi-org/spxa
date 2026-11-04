r"""
Stochastic local volatility: grids and leverage functions.

The SLV model in forward log-moneyness :math:`y = \log(S_t/F_t)` is

.. math::
    dy = -\tfrac12 L^2(t, y)\,v\,dt + L(t, y)\sqrt v\,dW^1, \qquad
    dv = \kappa(\theta - v)\,dt + \eta\,\xi\sqrt v\,dW^2,

a Heston variance (:class:`spxa.vol.heston.Heston`) with its vol-of-variance
scaled by the mixing fraction :math:`\eta`, and a leverage function
:math:`L`. By Gyöngy's theorem the model reproduces the local volatility
:math:`\sigma_{LV}` (and so every vanilla) exactly when

.. math:: L^2(t, y)\,E[v_t \mid y_t = y] = \sigma_{LV}^2(t, y).

This module holds the grid on which the joint density of :math:`(y, v)` is
computed and the tabulated leverage :class:`LeverageSurface`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from spxa.pde.grid import Grid1D
from spxa.vol.heston import Heston

__all__ = ["LeverageSurface", "SLVGrid", "slv_grid", "variance_quantile", "variance_std"]


def variance_std(model: Heston, t: float) -> float:
    r"""Standard deviation of :math:`v_t`:
    :math:`\operatorname{Var} v_t = v_0\frac{\xi^2}{\kappa}e^{-\kappa t}(1 - e^{-\kappa t})
    + \theta\frac{\xi^2}{2\kappa}(1 - e^{-\kappa t})^2`."""
    e = math.exp(-model.kappa * t)
    k, x2 = model.kappa, model.xi**2
    return math.sqrt(model.v0 * x2 / k * e * (1 - e) + model.theta * x2 / (2 * k) * (1 - e) ** 2)


def variance_quantile(model: Heston, t: float, q: float) -> float:
    r"""Quantile of :math:`v_t`, a scaled non-central chi-square with
    :math:`4\kappa\theta/\xi^2` degrees of freedom."""
    from scipy.stats import ncx2

    if t <= 0:
        return model.v0
    c = model.xi**2 * (1 - math.exp(-model.kappa * t)) / (4 * model.kappa)
    df = 4 * model.kappa * model.theta / model.xi**2
    return float(c * ncx2.ppf(q, df, model.v0 * math.exp(-model.kappa * t) / c))


@dataclass(frozen=True)
class SLVGrid:
    """Tensor grid in log-moneyness ``y`` (uniform, symmetric about 0) and variance ``v``."""

    y: Grid1D
    v: Grid1D

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y.n, self.v.n)

    @property
    def weights(self) -> np.ndarray:
        """Control-volume areas, shape :attr:`shape`."""
        return np.outer(self.y.cell_widths, self.v.cell_widths)

    def node(self, y: float, v: float) -> tuple[int, int]:
        return int(np.argmin(np.abs(self.y.nodes - y))), int(np.argmin(np.abs(self.v.nodes - v)))


def slv_grid(
    model: Heston,
    T: float,
    n_y: int = 121,
    n_v: int = 61,
    n_sd: float = 5.0,
    vol: float | None = None,
    v_tail: float = 1e-6,
) -> SLVGrid:
    r"""Grid covering ``n_sd`` standard deviations of :math:`y_T` and :math:`v_T`.

    The ``y`` range is :math:`\pm n_{sd}\,s\sqrt T` with ``s`` the larger of
    ``vol`` (a typical local volatility) and :math:`\sqrt{\max(v_0, \theta)}`;
    ``n_y`` is made odd so that :math:`y = 0` is a node. Variance nodes are
    sinh-clustered towards :math:`v = 0`, where the density of a non-Feller
    variance concentrates, with :math:`v_0` placed exactly on a node; the
    top is the largest :math:`(1 - v_{tail})`-quantile of :math:`v_t` over
    the horizon (non-central chi-square), at least :math:`3\max(v_0, \theta)`.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    vbar = max(model.v0, model.theta)
    v_hi = max(
        max(variance_quantile(model, t, 1 - v_tail) for t in np.linspace(T / 8, T, 8)), 3 * vbar
    )
    v = Grid1D.sinh(0.0, v_hi, n_v, center=0.0, alpha=vbar / 2, align=[model.v0])
    s = max(math.sqrt(vbar), vol or 0.0)
    y_hi = n_sd * s * math.sqrt(T)
    y = Grid1D.uniform(-y_hi, y_hi, n_y + (1 - n_y % 2))
    return SLVGrid(y, v)


class LeverageSurface:
    r"""Leverage :math:`L(t, y)` tabulated at times ``ts`` and nodes ``ys``.

    Linear in ``y`` with flat extrapolation; piecewise constant in time,
    with row ``k`` applying on :math:`[t_k, t_{k+1})`, which is how the
    forward induction of :func:`spxa.vol.calibrate_leverage` produces it.
    """

    def __init__(self, ts: Any, ys: Any, values: Any) -> None:
        self.ts = np.asarray(ts, float)
        self.ys = np.asarray(ys, float)
        self.values = np.asarray(values, float)
        if self.values.shape != (self.ts.size, self.ys.size):
            raise ValueError("values must have shape (len(ts), len(ys))")
        if np.any(np.diff(self.ts) <= 0) or np.any(np.diff(self.ys) <= 0):
            raise ValueError("grids must be strictly increasing")
        if np.any(~np.isfinite(self.values)) or np.any(self.values < 0):
            raise ValueError("leverage must be finite and non-negative")

    @classmethod
    def constant(cls, value: float = 1.0) -> "LeverageSurface":
        return cls([0.0], [-1.0, 1.0], [[value, value]])

    def row(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.ts, t, side="right") - 1, 0, self.ts.size - 1))

    def __call__(self, y: Any, t: float) -> Any:
        out = np.interp(np.asarray(y, float), self.ys, self.values[self.row(t)])
        return float(out) if np.ndim(out) == 0 else out
