r"""SABR implied volatilities (Hagan, Kumar, Lesniewski & Woodward 2002).

The forward follows

.. math:: dF = \alpha_t (F + s)^\beta\,dW, \qquad d\alpha = \nu\alpha\,dZ, \qquad dW\,dZ = \rho\,dt,

where the shift :math:`s` (zero for classic SABR) lets rates go down to
:math:`-s`. Hagan's singular-perturbation expansion gives the Black
volatility of the shifted forward,

.. math::
    \sigma_B = \frac{\alpha}{(FK)^{(1-\beta)/2}\bigl[1 + \frac{(1-\beta)^2}{24}\ell^2
    + \frac{(1-\beta)^4}{1920}\ell^4\bigr]}\,\frac{z}{\chi(z)}
    \Bigl[1 + \Bigl(\frac{(1-\beta)^2\alpha^2}{24 (FK)^{1-\beta}}
    + \frac{\rho\beta\nu\alpha}{4 (FK)^{(1-\beta)/2}} + \frac{2 - 3\rho^2}{24}\nu^2\Bigr)T\Bigr],

and the normal (Bachelier) volatility,

.. math::
    \sigma_N = \alpha (FK)^{\beta/2}\,
    \frac{1 + \ell^2/24 + \ell^4/1920}{1 + \frac{(1-\beta)^2}{24}\ell^2 + \frac{(1-\beta)^4}{1920}\ell^4}
    \,\frac{z}{\chi(z)}
    \Bigl[1 + \Bigl(\frac{-\beta(2-\beta)\alpha^2}{24 (FK)^{1-\beta}}
    + \frac{\rho\beta\nu\alpha}{4 (FK)^{(1-\beta)/2}} + \frac{2 - 3\rho^2}{24}\nu^2\Bigr)T\Bigr].

Here :math:`F, K` are the shifted forward and strike,
:math:`\ell = \log(F/K)`, :math:`z = \frac{\nu}{\alpha}(FK)^{(1-\beta)/2}\ell`
and :math:`\chi(z) = \log\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho}`.
Near the money, :math:`z/\chi(z)` is evaluated by its series
:math:`1 - \rho z/2 + (2 - 3\rho^2) z^2/12`. Both formulas are first-order
accurate in :math:`T`. The normal vol is a volatility of the unshifted
rate, because Bachelier prices do not depend on the shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spxa.market.black import bachelier_price, black_price

__all__ = ["SABR", "sabr_lognormal_vol", "sabr_normal_vol", "z_over_chi"]


def z_over_chi(z: Any, rho: float) -> Any:
    """:math:`z/\\chi(z)` with the near-zero series."""
    zz = np.asarray(z, float)
    small = np.abs(zz) < 1e-5
    zs = np.where(small, 1.0, zz)
    chi = np.log((np.sqrt(1.0 - 2.0 * rho * zs + zs * zs) + zs - rho) / (1.0 - rho))
    series = 1.0 - 0.5 * rho * zz + (2.0 - 3.0 * rho * rho) / 12.0 * zz * zz
    out = np.where(small, series, zs / chi)
    return float(out) if out.ndim == 0 else out


def _check(alpha: float, beta: float, rho: float, nu: float) -> None:
    if alpha <= 0 or not 0 <= beta <= 1 or not -1 < rho < 1 or nu < 0:
        raise ValueError("need alpha > 0, 0 <= beta <= 1, -1 < rho < 1, nu >= 0")


def _parts(F: Any, K: Any, alpha: float, beta: float, rho: float, nu: float, shift: float) -> tuple:
    Fs = np.asarray(F, float) + shift
    Ks = np.asarray(K, float) + shift
    if np.any(Fs <= 0) or np.any(Ks <= 0):
        raise ValueError("SABR needs F + shift > 0 and K + shift > 0")
    ell = np.log(Fs / Ks)
    fk = Fs * Ks
    zc = z_over_chi(nu / alpha * fk ** ((1 - beta) / 2) * ell, rho)
    return fk, ell, np.asarray(zc, float)


def sabr_lognormal_vol(
    F: Any, K: Any, T: Any, alpha: float, beta: float, rho: float, nu: float, shift: float = 0.0
) -> Any:
    """Hagan's Black volatility (of the shifted forward)."""
    _check(alpha, beta, rho, nu)
    fk, ell, zc = _parts(F, K, alpha, beta, rho, nu, shift)
    b = 1.0 - beta
    den = fk ** (b / 2) * (1 + b * b / 24 * ell**2 + b**4 / 1920 * ell**4)
    corr = b * b * alpha**2 / (24 * fk**b) + rho * beta * nu * alpha / (4 * fk ** (b / 2))
    corr = corr + (2 - 3 * rho * rho) / 24 * nu * nu
    out = alpha / den * zc * (1 + corr * np.asarray(T, float))
    return float(out) if np.ndim(out) == 0 else out


def sabr_normal_vol(
    F: Any, K: Any, T: Any, alpha: float, beta: float, rho: float, nu: float, shift: float = 0.0
) -> Any:
    """Hagan's normal (Bachelier) volatility."""
    _check(alpha, beta, rho, nu)
    fk, ell, zc = _parts(F, K, alpha, beta, rho, nu, shift)
    b = 1.0 - beta
    num = 1 + ell**2 / 24 + ell**4 / 1920
    den = 1 + b * b / 24 * ell**2 + b**4 / 1920 * ell**4
    corr = -beta * (2 - beta) * alpha**2 / (24 * fk**b) + rho * beta * nu * alpha / (
        4 * fk ** (b / 2)
    )
    corr = corr + (2 - 3 * rho * rho) / 24 * nu * nu
    out = alpha * fk ** (beta / 2) * num / den * zc * (1 + corr * np.asarray(T, float))
    return float(out) if np.ndim(out) == 0 else out


@dataclass(frozen=True)
class SABR:
    """SABR parameters with Hagan's implied volatilities and prices."""

    alpha: float
    beta: float
    rho: float
    nu: float
    shift: float = 0.0

    def __post_init__(self) -> None:
        _check(self.alpha, self.beta, self.rho, self.nu)

    def lognormal_vol(self, F: Any, K: Any, T: Any) -> Any:
        return sabr_lognormal_vol(F, K, T, self.alpha, self.beta, self.rho, self.nu, self.shift)

    def normal_vol(self, F: Any, K: Any, T: Any) -> Any:
        return sabr_normal_vol(F, K, T, self.alpha, self.beta, self.rho, self.nu, self.shift)

    def price(
        self, F: Any, K: Any, T: Any, df: Any = 1.0, option: Any = "call", vol: str = "lognormal"
    ) -> Any:
        """Option value from the Black (shifted) or Bachelier formula."""
        if vol == "lognormal":
            s = self.shift
            sig = self.lognormal_vol(F, K, T)
            return black_price(np.asarray(F) + s, np.asarray(K) + s, T, sig, df, option)
        if vol == "normal":
            return bachelier_price(F, K, T, self.normal_vol(F, K, T), df, option)
        raise ValueError("vol must be 'lognormal' or 'normal'")
