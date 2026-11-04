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

**Volatility swaps** pay :math:`\sqrt{V_T/T}` and are not replicable by
vanillas. In Heston the Laplace transform of the integrated variance
:math:`V_T = \int_0^T v_t\,dt` has a closed form, and
:math:`E\sqrt{V} = \frac{1}{2\sqrt\pi}\int_0^\infty \frac{1 - E e^{-sV}}{s^{3/2}}\,ds`.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.integrate import quad

from spxa.vol.conventions import normalised_call_from_total_variance
from spxa.vol.heston import Heston

__all__ = [
    "ForwardVarianceCurve",
    "variance_swap_curve",
    "vix_index",
    "fair_variance",
    "gamma_swap_strike",
    "heston_integrated_variance_laplace",
    "heston_volatility_swap",
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


def _heston_log_laplace(model: Heston, T: float, s: Any) -> Any:
    # log E exp(-s V_T) written without cancellation as xi -> 0: with g = sqrt(k^2 + 2 xi^2 s)
    # and d = g - k = 2 xi^2 s / (g + k), the denominator is 2g - d(1 - e^{-gT})
    s = np.asarray(s, float)
    k, th, xi, v0 = model.kappa, model.theta, model.xi, model.v0
    g = np.sqrt(k * k + 2.0 * xi * xi * s)
    d = 2.0 * xi * xi * s / (g + k)
    one_e = -np.expm1(-g * T)
    den = 2.0 * g - d * one_e
    B = 2.0 * s * one_e / den
    c = 2.0 * k * th / (xi * xi)
    logA = -c * np.log1p(-d * one_e / (2.0 * g)) - (4.0 * k * th * s / (g + k)) * T / 2.0
    return logA - B * v0


def heston_integrated_variance_laplace(model: Heston, T: float, s: Any) -> Any:
    r""":math:`E\exp(-s\int_0^T v_t\,dt)` in Heston (the CIR bond-price formula)."""
    return np.exp(_heston_log_laplace(model, T, s))


def heston_volatility_swap(model: Heston, T: float) -> float:
    r"""Fair volatility-swap strike :math:`E\sqrt{V_T/T}` in Heston."""
    if T <= 0:
        raise ValueError("T must be positive")

    def f(u: float) -> float:
        # s = u^2 removes the s^{-3/2} singularity at zero: ds / s^{3/2} = 2 du / u^2
        s = u * u
        return 2.0 * float(-np.expm1(_heston_log_laplace(model, T, s))) / (u * u)

    cuts = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0]
    val = sum(quad(f, a, b, limit=200, epsabs=1e-14)[0] for a, b in zip(cuts, cuts[1:]))
    val += quad(f, cuts[-1], np.inf, limit=200, epsabs=1e-14)[0]
    return val / (2.0 * math.sqrt(math.pi)) / math.sqrt(T)


def variance_swap_curve(surface: Any, expiries: Any) -> np.ndarray:
    """Fair variance strikes :math:`K_{\\mathrm{var}}(T_i)` for each expiry."""
    return np.array([variance_swap_strike(surface, float(T)) for T in np.atleast_1d(expiries)])


class ForwardVarianceCurve:
    r"""Piecewise-flat forward variance :math:`\xi_0(t)` from variance swap strikes.

    Between expiries :math:`T_{i-1} < t \le T_i` (with :math:`T_0 = 0`), the forward variance is
    :math:`\xi_i = (K_i T_i - K_{i-1}T_{i-1})/(T_i - T_{i-1})`, so
    :math:`\int_0^{T_i}\xi_0 = K_i T_i`. It is flat beyond the last expiry. This is the
    input curve of the forward-variance models (rough Bergomi, :class:`RBergomiVIX`).
    """

    def __init__(self, expiries: Any, strikes: Any) -> None:
        T, K = np.asarray(expiries, float), np.asarray(strikes, float)
        if T.ndim != 1 or T.shape != K.shape or T.size == 0 or np.any(np.diff(T) <= 0) or T[0] <= 0:
            raise ValueError("need increasing positive expiries and matching strikes")
        total = np.concatenate([[0.0], K * T])
        xi = np.diff(total) / np.diff(np.concatenate([[0.0], T]))
        if np.any(xi < 0):
            raise ValueError("total variance K T must be non-decreasing (calendar arbitrage)")
        self.expiries, self.strikes, self.levels = T, K, xi

    def __call__(self, t: Any) -> np.ndarray:
        idx = np.searchsorted(self.expiries, np.asarray(t, float), side="left")
        return self.levels[np.minimum(idx, self.levels.size - 1)]

    def integral(self, t: Any) -> np.ndarray:
        r""":math:`\int_0^t\xi_0(u)\,du`."""
        tt = np.asarray(t, float)
        knots = np.concatenate([[0.0], self.expiries])
        cum = np.concatenate([[0.0], np.cumsum(self.levels * np.diff(knots))])
        idx = np.minimum(np.searchsorted(self.expiries, tt, side="left"), self.levels.size - 1)
        return cum[idx] + self.levels[idx] * (tt - knots[idx])

    def variance_swap(self, T: Any) -> np.ndarray:
        """Fair variance strike implied by the curve at any maturity."""
        return self.integral(T) / np.asarray(T, float)

    def forward_variance_swap(self, T1: float, T2: float) -> float:
        """Fair strike of a variance swap on :math:`[T_1, T_2]` seen today."""
        if not 0 <= T1 < T2:
            raise ValueError("need 0 <= T1 < T2")
        return float((self.integral(T2) - self.integral(T1)) / (T2 - T1))


def vix_index(surface: Any, delta: float = 30.0 / 365.0) -> float:
    r"""The model-free VIX of a surface: :math:`100\sqrt{K_{\mathrm{var}}(\Delta)}` with a 30-day window."""
    return 100.0 * math.sqrt(variance_swap_strike(surface, delta))
