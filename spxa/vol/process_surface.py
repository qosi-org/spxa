r"""Implied-volatility surfaces of the processes in :mod:`spxa.zoo`.

:func:`process_vol_surface` turns an exponential-Lévy model
:math:`S_T = S_0 e^{(r - q)T + X_T}` into a
:class:`~spxa.vol.surface.VolSurface`. Here :math:`X` is the process's
martingale-normalised Lévy process (:func:`spxa.pde.reference.levy_cf`).
For each expiry:

1. Price out-of-the-money options with the Lewis (2001) formula
   (:func:`spxa.pde.reference.lewis_call_price`) at log-moneyness
   :math:`k = x\,\sigma_{\text{ATM}}\sqrt{T}`, with :math:`x` evenly spaced on
   ``[-width, width]``.
2. Invert Black's formula for the total variances
   :math:`w = \sigma^2 T`.
3. Fit raw SVI with :func:`spxa.vol.calibration.calibrate_svi`.

The slices are joined by price interpolation
(:meth:`VolSurface.from_slices`). The result records the model's implied
vols, so the SVI fitting error is visible next to the surface.
:meth:`spxa.core.process.Process.to_vol_surface` calls this function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from spxa.market.curves import EquityForward
from spxa.market.implied import implied_black_volatility
from spxa.pde.reference import levy_cf, lewis_call_price
from spxa.vol.calibration import calibrate_svi
from spxa.vol.surface import VolSurface

__all__ = ["ProcessSurface", "process_vol_surface"]


@dataclass
class ProcessSurface:
    """A process's surface with the model implied vols it was fitted to.

    ``log_moneyness[i]`` and ``model_vols[i]`` belong to ``expiries[i]``.
    ``max_vol_error`` is the largest absolute difference between the
    surface's and the model's implied vols at those points.
    """

    surface: VolSurface
    expiries: list[float]
    log_moneyness: list[np.ndarray]
    model_vols: list[np.ndarray]
    max_vol_error: float


def process_vol_surface(
    process: Any,
    spot: float,
    rate: float,
    expiries: Sequence[float],
    dividend_yield: float = 0.0,
    width: float = 2.5,
    n_strikes: int = 21,
) -> ProcessSurface:
    """Implied-volatility surface of an exponential-Lévy model (see the module docstring)."""
    T_all = [float(t) for t in expiries]
    if not T_all or any(t <= 0 for t in T_all) or any(np.diff(T_all) <= 0):
        raise ValueError("expiries must be positive and increasing")
    if width <= 0 or n_strikes < 5:
        raise ValueError("width must be positive and n_strikes at least 5")
    fwd = EquityForward(spot, rate, dividend_yield)
    slices, ks, vols = [], [], []
    for T in T_all:
        cf = levy_cf(process, T)
        F, D = float(fwd.forward(T)), float(fwd.discount(T))

        def vol_at(
            k: np.ndarray, T: float = T, F: float = F, D: float = D, cf: Any = cf
        ) -> np.ndarray:
            out = []
            for kj in k:
                K = F * math.exp(kj)
                option = "put" if kj < 0 else "call"
                p = lewis_call_price(cf, spot, K, T, rate, dividend_yield, option)
                out.append(float(implied_black_volatility(p, F, K, T, D, option)))
            return np.array(out)

        atm = float(vol_at(np.array([0.0]))[0])
        k = np.linspace(-width, width, n_strikes) * atm * math.sqrt(T)
        v = vol_at(k)
        if not np.all(np.isfinite(v)):
            raise ValueError(f"implied volatility failed at T={T:g}; reduce width")
        slices.append(calibrate_svi(k, v * v * T).params)
        ks.append(k)
        vols.append(v)
    surface = VolSurface.from_slices(T_all, slices, fwd)
    err = max(
        float(np.max(np.abs(np.asarray(surface.implied_vol_k(k, T), float) - v)))
        for k, v, T in zip(ks, vols, T_all)
    )
    return ProcessSurface(surface, T_all, ks, vols, err)
