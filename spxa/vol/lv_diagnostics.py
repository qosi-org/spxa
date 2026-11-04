"""Diagnostics for a local-volatility surface.

Two questions decide whether a Dupire local volatility can be used:

* Is it defined? Dupire's formula has no positive solution where the
  implied surface has butterfly or calendar arbitrage
  (:class:`~spxa.vol.local_vol.LocalVolSurface` returns ``nan`` there).
  :func:`local_vol_diagnostics` maps those points on a (moneyness, time)
  grid.
* Does it reprice? Vanillas are solved with the forward Dupire PDE of
  :mod:`spxa.pde` on one grid for all expiries, then converted back to
  implied vols and compared with the surface. The errors combine
  discretisation of the PDE and the finite differences in Dupire's
  formula. They should shrink as the grids are refined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["LocalVolDiagnostics", "local_vol_diagnostics"]


@dataclass(frozen=True)
class LocalVolDiagnostics:
    """Repricing errors and the undefined region of a local-vol surface."""

    expiries: np.ndarray
    strikes: np.ndarray
    model_vols: np.ndarray  # (n_expiries, n_strikes), from the Dupire PDE
    surface_vols: np.ndarray
    y_grid: np.ndarray
    t_grid: np.ndarray
    undefined: np.ndarray  # (n_t, n_y) boolean mask of nan local vol

    @property
    def errors(self) -> np.ndarray:
        return self.model_vols - self.surface_vols

    @property
    def max_error(self) -> float:
        return float(np.nanmax(np.abs(self.errors)))

    @property
    def undefined_fraction(self) -> float:
        return float(self.undefined.mean())

    def worst(self) -> tuple[float, float, float]:
        """``(T, K, error)`` of the largest absolute implied-vol error."""
        i, j = np.unravel_index(int(np.nanargmax(np.abs(self.errors))), self.errors.shape)
        return float(self.expiries[i]), float(self.strikes[j]), float(self.errors[i, j])

    def first_undefined_time(self) -> Optional[float]:
        """Earliest grid time with an undefined local vol, or ``None``."""
        rows = np.flatnonzero(self.undefined.any(axis=1))
        return float(self.t_grid[rows[0]]) if rows.size else None


def local_vol_diagnostics(
    lv: Any,
    spot: float,
    r: float,
    q: float,
    strikes: Sequence[float],
    expiries: Sequence[float],
    n_space: int = 801,
    n_time: int = 400,
    y_grid: Optional[Sequence[float]] = None,
    n_t: int = 40,
    reprice: bool = True,
) -> LocalVolDiagnostics:
    """Reprice ``strikes`` x ``expiries`` under ``lv`` and map where it is undefined.

    ``lv`` is a :class:`~spxa.vol.local_vol.LocalVolSurface`. The
    undefined region is sampled on ``y_grid`` (default 81 points on
    :math:`\\pm 1.5`) and ``n_t`` times up to the last expiry. With
    ``reprice=False`` (or if the local vol is undefined before the last
    expiry), the model vols are ``nan``.
    """
    K = np.asarray(strikes, float)
    T = np.sort(np.asarray(expiries, float))
    if T.size == 0 or T[0] <= 0:
        raise ValueError("expiries must be positive")
    y = np.linspace(-1.5, 1.5, 81) if y_grid is None else np.asarray(y_grid, float)
    t = np.linspace(T[-1] / n_t, T[-1], n_t)
    undefined = np.array([np.isnan(np.asarray(lv.local_vol_k(y, float(ti)), float)) for ti in t])
    surf = np.array([np.asarray(lv.surface.implied_vol(K, float(Ti)), float) for Ti in T])
    model = np.full_like(surf, np.nan)
    if reprice and not undefined.any():
        sol = lv.dupire_pde(r, q).solve(
            spot, float(T[-1]), maturities=list(T[:-1]), n_space=n_space, n_time=n_time
        )
        model = np.array([np.asarray(sol.implied_vol(K, float(Ti)), float) for Ti in T])
    return LocalVolDiagnostics(T, K, model, surf, y, t, undefined)
