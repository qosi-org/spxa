r"""Surface SVI (SSVI) of Gatheral & Jacquier (2014).

.. math:: w(k, \theta_t) = \frac{\theta_t}{2}\Bigl(1 + \rho\varphi(\theta_t)k
          + \sqrt{(\varphi(\theta_t)k + \rho)^2 + 1 - \rho^2}\Bigr),

where :math:`\theta_t = w(0, t)` is the ATM total variance and
:math:`\varphi` a curvature function. Each slice is a natural SVI slice
with :math:`(\Delta, \mu, \omega, \zeta) = (0, 0, \theta_t, \varphi(\theta_t))`.

No-arbitrage conditions
    *Calendar* (Theorem 4.1): :math:`\theta_t` non-decreasing and
    :math:`0 \le \partial_\theta(\theta\varphi(\theta)) \le
    \frac{1 + \sqrt{1-\rho^2}}{\rho^2}\varphi(\theta)`.
    *Butterfly* (Theorem 4.2, sufficient):
    :math:`\theta\varphi(\theta)(1 + |\rho|) < 4` and
    :math:`\theta\varphi(\theta)^2(1 + |\rho|) \le 4`.

Curvature functions
    :class:`HestonLikePhi`, :math:`\varphi(\theta) = \frac{1}{\lambda\theta}
    \bigl(1 - \frac{1 - e^{-\lambda\theta}}{\lambda\theta}\bigr)`, is free of
    static arbitrage when :math:`\lambda \ge (1 + |\rho|)/4` (Corollary 4.1).
    :class:`PowerLawPhi`, :math:`\varphi(\theta) = \eta\,\theta^{-\gamma}(1+\theta)^{\gamma-1}`
    (eq. 4.5), is free of static arbitrage when :math:`0 < \gamma \le 1/2`
    and :math:`\eta(1 + |\rho|) \le 2` (Remark 4.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Union

import numpy as np

from spxa.vol.svi import SVINatural, SVIRaw

__all__ = ["ATMVarianceCurve", "HestonLikePhi", "PowerLawPhi", "SSVI"]


def _out(x: np.ndarray) -> Any:
    return float(x) if np.ndim(x) == 0 else x


# Taylor coefficients (highest power first, for np.polyval) of
#   (1 - (1 - e^{-x})/x)/x = sum_j (-1)^j x^j / (j+2)!
#   (1 - e^{-x}(1 + x))/x^2 = sum_j (-1)^j (j+1) x^j / (j+2)!
# 22 terms make the truncation error below 1e-20 for x < 1.
_PHI_SERIES = np.array([(-1) ** j / math.factorial(j + 2) for j in range(22)][::-1])
_DPHI_SERIES = np.array([(-1) ** j * (j + 1) / math.factorial(j + 2) for j in range(22)][::-1])


@dataclass(frozen=True)
class HestonLikePhi:
    r""":math:`\varphi(\theta) = \frac{1}{\lambda\theta}\bigl(1 - \frac{1-e^{-\lambda\theta}}{\lambda\theta}\bigr)`."""

    lam: float

    def __post_init__(self) -> None:
        if self.lam <= 0:
            raise ValueError("lam must be positive")

    def __call__(self, theta: Any) -> Any:
        x = self.lam * np.asarray(theta, float)
        if np.any(x <= 0):
            raise ValueError("theta must be positive")
        # lam (1 - (1 - e^{-x})/x)/x, with a Taylor series where it cancels
        small = x < 1.0
        xs = np.where(small, 1.0, x)
        big = (1.0 - (-np.expm1(-xs)) / xs) / xs
        return _out(self.lam * np.where(small, np.polyval(_PHI_SERIES, x), big))

    def theta_phi_derivative(self, theta: Any) -> Any:
        r""":math:`\partial_\theta(\theta\varphi(\theta)) = \bigl(1 - e^{-x}(1+x)\bigr)/x^2` with :math:`x = \lambda\theta`."""
        x = self.lam * np.asarray(theta, float)
        small = x < 1.0
        xs = np.where(small, 1.0, x)
        big = (1.0 - np.exp(-xs) * (1.0 + xs)) / (xs * xs)
        return _out(np.where(small, np.polyval(_DPHI_SERIES, x), big))


@dataclass(frozen=True)
class PowerLawPhi:
    r""":math:`\varphi(\theta) = \eta\,\theta^{-\gamma}(1+\theta)^{\gamma-1}`."""

    eta: float
    gamma: float

    def __post_init__(self) -> None:
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1)")

    def __call__(self, theta: Any) -> Any:
        t = np.asarray(theta, float)
        if np.any(t <= 0):
            raise ValueError("theta must be positive")
        return _out(self.eta * t ** (-self.gamma) * (1.0 + t) ** (self.gamma - 1.0))

    def theta_phi_derivative(self, theta: Any) -> Any:
        r""":math:`\partial_\theta(\theta\varphi) = \eta(1-\gamma)\,\theta^{-\gamma}(1+\theta)^{\gamma-2}`."""
        t = np.asarray(theta, float)
        g = self.gamma
        return _out(self.eta * (1.0 - g) * t ** (-g) * (1.0 + t) ** (g - 2.0))


Phi = Union[HestonLikePhi, PowerLawPhi]


class ATMVarianceCurve:
    r"""ATM total variance :math:`\theta_t`, piecewise linear in :math:`t`.

    The curve starts at :math:`\theta_0 = 0`; beyond the last node the last
    slope is extrapolated. Nodes must be non-decreasing (no calendar
    arbitrage at the money).
    """

    def __init__(self, times: Sequence[float], thetas: Sequence[float]) -> None:
        t = np.asarray(times, float)
        th = np.asarray(thetas, float)
        if t.ndim != 1 or t.shape != th.shape or t.size == 0:
            raise ValueError("times and thetas must be 1-D arrays of equal length")
        if t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("times must be positive and increasing")
        if th[0] <= 0 or np.any(np.diff(th) < 0):
            raise ValueError("ATM total variances must be positive and non-decreasing")
        self.times, self.thetas = t, th
        self._t = np.concatenate([[0.0], t])
        self._th = np.concatenate([[0.0], th])

    def __call__(self, T: Any) -> Any:
        tt = np.asarray(T, float)
        if np.any(tt < 0):
            raise ValueError("negative time")
        out = np.interp(tt, self._t, self._th)
        beyond = tt > self._t[-1]
        if np.any(beyond):
            slope = (self._th[-1] - self._th[-2]) / (self._t[-1] - self._t[-2])
            out = np.where(beyond, self._th[-1] + slope * (tt - self._t[-1]), out)
        return _out(out)


class SSVI:
    """SSVI surface.

    Parameters
    ----------
    rho : float
        Correlation-like skew parameter, ``|rho| < 1``.
    phi : HestonLikePhi or PowerLawPhi or callable
        Curvature function of the ATM total variance.
    theta : ATMVarianceCurve or callable
        ATM total variance as a function of expiry.
    """

    def __init__(
        self, rho: float, phi: Union[Phi, Callable[[Any], Any]], theta: Callable[[Any], Any]
    ) -> None:
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")
        self.rho = float(rho)
        self.phi = phi
        self.theta = theta

    # -- evaluation ---------------------------------------------------------
    @staticmethod
    def _w(k: Any, th: Any, ph: Any, rho: float) -> np.ndarray:
        y = ph * np.asarray(k, float)
        return 0.5 * th * (1.0 + rho * y + np.sqrt((y + rho) ** 2 + 1.0 - rho * rho))

    def total_variance(self, k: Any, T: Any) -> Any:
        th = np.asarray(self.theta(T), float)
        return _out(self._w(k, th, np.asarray(self.phi(th), float), self.rho))

    def total_variance_theta(self, k: Any, theta: Any) -> Any:
        """:math:`w(k, \\theta)` as a function of the ATM total variance."""
        th = np.asarray(theta, float)
        return _out(self._w(k, th, np.asarray(self.phi(th), float), self.rho))

    def implied_vol(self, k: Any, T: Any) -> Any:
        T_ = np.asarray(T, float)
        if np.any(T_ <= 0):
            raise ValueError("T must be positive")
        return _out(np.sqrt(np.asarray(self.total_variance(k, T_)) / T_))

    def slice(self, T: float) -> SVIRaw:
        """The expiry-``T`` slice as raw SVI."""
        th = float(self.theta(T))
        return SVINatural(0.0, 0.0, self.rho, th, float(self.phi(th))).to_raw()

    # -- no-arbitrage conditions ----------------------------------------------
    def butterfly_conditions(self, thetas: Any) -> np.ndarray:
        r"""Theorem 4.2 at each :math:`\theta`: :math:`\theta\varphi(1+|\rho|) < 4` and :math:`\theta\varphi^2(1+|\rho|) \le 4`."""
        th = np.asarray(thetas, float)
        ph = np.asarray(self.phi(th), float)
        a = 1.0 + abs(self.rho)
        return (th * ph * a < 4.0) & (th * ph * ph * a <= 4.0)

    def calendar_conditions(self, thetas: Any) -> np.ndarray:
        r"""Theorem 4.1 at each :math:`\theta`: :math:`0 \le \partial_\theta(\theta\varphi) \le \frac{1+\sqrt{1-\rho^2}}{\rho^2}\varphi`."""
        th = np.asarray(thetas, float)
        ph = np.asarray(self.phi(th), float)
        if hasattr(self.phi, "theta_phi_derivative"):
            d = np.asarray(self.phi.theta_phi_derivative(th), float)
        else:
            h = 1e-6 * np.maximum(th, 1e-8)
            d = (
                (th + h) * np.asarray(self.phi(th + h)) - (th - h) * np.asarray(self.phi(th - h))
            ) / (2 * h)
        upper = np.inf if self.rho == 0 else (1.0 + math.sqrt(1.0 - self.rho**2)) / self.rho**2 * ph
        return (d >= 0) & (d <= upper)

    def is_arbitrage_free(self, thetas: Any) -> bool:
        """Theorems 4.1 and 4.2 on the given ATM variances (and theta non-decreasing)."""
        th = np.asarray(thetas, float)
        return bool(np.all(self.butterfly_conditions(th)) and np.all(self.calendar_conditions(th)))
