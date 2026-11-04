r"""
Variance, volatility and gamma swaps.

**Variance swaps.** For a continuous price process with no jumps,
:math:`\int_0^T \sigma_t^2\,dt = -2\log(S_T/F_T) + 2\int_0^T dS_t/S_t - \dots`.
The realised variance is therefore replicated by a log contract plus a
delta-hedge, and the log contract by out-of-the-money vanillas (Carr–Madan,
Demeterfi–Derman–Kamal–Zou):

.. math::
    K_{\mathrm{var}}T = E[-2\log X_T] = 2\int_0^\infty \frac{\mathrm{OTM}(k)}{k^2}\,dk
    = 2\int_{-\infty}^{\infty}\mathrm{OTM}(e^x)e^{-x}\,dx,

with :math:`X_T = S_T/F_T` and :math:`\mathrm{OTM}(k)` the undiscounted
out-of-the-money price over the forward. A gamma swap weights the
variance by :math:`S_t/S_0`, and its fair strike uses the weight
:math:`1/k` instead of :math:`1/k^2`.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from spxa.vol.conventions import normalised_call_from_total_variance

__all__ = [
    "fair_variance",
    "gamma_swap_strike",
    "variance_swap_mtm",
    "variance_swap_strike",
]

OTM = Callable[[np.ndarray], np.ndarray]
_CUTS = np.array(
    [
        -40.0,
        -20.0,
        -12.0,
        -8.0,
        -5.0,
        -3.0,
        -1.5,
        -0.5,
        0.0,
        0.5,
        1.5,
        3.0,
        5.0,
        8.0,
        12.0,
        20.0,
        40.0,
    ]
)
_GL = np.polynomial.legendre.leggauss(32)


def _otm_from_surface(surface: Any, T: float) -> OTM:
    def otm(k: np.ndarray) -> np.ndarray:
        x = np.log(k)
        c = np.asarray(
            normalised_call_from_total_variance(x, np.asarray(surface.total_variance(x, T), float)),
            float,
        )
        return np.where(k >= 1.0, c, c - (1.0 - k))

    return otm


def _replicate(otm: OTM, power: float, sd: float) -> float:
    # 2 * int OTM(e^x) e^{(1-power) x} dx by composite Gauss-Legendre in one vectorised call;
    # panels scale with the at-the-money standard deviation sd and stop at 40 sd (|x| <= 40)
    cuts = np.clip(_CUTS * sd, -40.0, 40.0)
    cuts = np.unique(cuts)
    mid, half = 0.5 * (cuts[1:] + cuts[:-1]), 0.5 * np.diff(cuts)
    x = (mid[:, None] + half[:, None] * _GL[0][None, :]).ravel()
    w = (half[:, None] * _GL[1][None, :]).ravel()
    return 2.0 * float(np.sum(w * np.asarray(otm(np.exp(x)), float) * np.exp((1.0 - power) * x)))


def _atm_sd(surface: Any, T: float) -> float:
    return math.sqrt(float(np.asarray(surface.total_variance(np.zeros(1), T), float)[0]))


def fair_variance(otm: OTM, T: float, sd: float | None = None) -> float:
    r"""Fair (annualised) variance-swap strike from normalised OTM prices :math:`k \mapsto \mathrm{OTM}(k)`.

    ``otm`` is vectorised. ``sd`` is the scale of the log-moneyness grid, e.g. the
    at-the-money total volatility (default :math:`0.2\sqrt T`).
    """
    if T <= 0:
        raise ValueError("T must be positive")
    return _replicate(otm, 2.0, sd if sd is not None else 0.2 * math.sqrt(T)) / T


def variance_swap_strike(surface: Any, T: float) -> float:
    """Fair variance strike from a surface with ``total_variance(k, T)`` (log-forward moneyness)."""
    return fair_variance(_otm_from_surface(surface, T), T, _atm_sd(surface, T))


def gamma_swap_strike(surface: Any, T: float) -> float:
    r"""Fair gamma-swap strike, :math:`\frac{2}{T}\int \mathrm{OTM}(k)/k\,dk = \frac{2}{T}E[X_T\log X_T]`."""
    if T <= 0:
        raise ValueError("T must be positive")
    return _replicate(_otm_from_surface(surface, T), 1.0, _atm_sd(surface, T)) / T


def variance_swap_mtm(
    realised_variance: float,
    t: float,
    T: float,
    strike: float,
    fair_remaining: float,
    df: float,
    notional: float = 1.0,
) -> float:
    r"""Value at :math:`t` of a variance swap struck at ``strike`` (variance units, annualised).

    The final payoff is :math:`N(\sigma^2_{\text{realised}, [0,T]} - K)`. The variance
    realised so far and the fair variance of the rest add by time weight.
    """
    if not 0 <= t < T:
        raise ValueError("need 0 <= t < T")
    expected = (realised_variance * t + fair_remaining * (T - t)) / T
    return notional * df * (expected - strike)
