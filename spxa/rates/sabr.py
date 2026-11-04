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

__all__ = [
    "SABR",
    "SABRFit",
    "alpha_from_atm",
    "calibrate_sabr",
    "sabr_lognormal_vol",
    "sabr_normal_vol",
    "z_over_chi",
]


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


def _x_over_expm1(x: np.ndarray) -> np.ndarray:
    small = np.abs(x) < 1e-8
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - 0.5 * x, xs / np.expm1(xs))


def sabr_lognormal_vol(
    F: Any,
    K: Any,
    T: Any,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
    method: str = "hagan",
) -> Any:
    r"""Black volatility (of the shifted forward) by Hagan's or Obłój's formula.

    ``method='obloj'`` (Obłój 2008, *Fine-tune your smile*) keeps Hagan's
    first-order term but replaces the leading term by
    :math:`\nu\ell/\chi(z_O)` with
    :math:`z_O = \nu (F^{1-\beta} - K^{1-\beta})/(\alpha(1-\beta))`
    (:math:`\nu\ell/\alpha` for :math:`\beta = 1`). That leading term is the
    exact short-expiry limit of the implied volatility (the
    Berestycki–Busca–Florent geodesic distance), which Hagan's
    :math:`\ell`-series only approximates away from the money.
    """
    _check(alpha, beta, rho, nu)
    if method not in ("hagan", "obloj"):
        raise ValueError("method must be 'hagan' or 'obloj'")
    fk, ell, zc = _parts(F, K, alpha, beta, rho, nu, shift)
    b = 1.0 - beta
    corr = b * b * alpha**2 / (24 * fk**b) + rho * beta * nu * alpha / (4 * fk ** (b / 2))
    corr = corr + (2 - 3 * rho * rho) / 24 * nu * nu
    if method == "hagan":
        den = fk ** (b / 2) * (1 + b * b / 24 * ell**2 + b**4 / 1920 * ell**4)
        lead = alpha / den * zc
    else:
        Ks = np.asarray(K, float) + shift
        # F^b - K^b = K^b expm1(b ell); alpha b ell / (F^b - K^b) -> alpha / K^b expm1 ratio
        ratio = alpha * _x_over_expm1(b * ell) / Ks**b
        z = nu * ell / ratio
        lead = ratio * np.asarray(z_over_chi(z, rho), float)
    out = lead * (1 + corr * np.asarray(T, float))
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

    def lognormal_vol(self, F: Any, K: Any, T: Any, method: str = "hagan") -> Any:
        return sabr_lognormal_vol(
            F, K, T, self.alpha, self.beta, self.rho, self.nu, self.shift, method
        )

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


def alpha_from_atm(
    atm_vol: float, F: float, T: float, beta: float, rho: float, nu: float, shift: float = 0.0
) -> float:
    """The :math:`\\alpha` that reproduces a Black ATM vol in Hagan's formula.

    At the money the formula is a cubic in :math:`\\alpha`:
    :math:`\\frac{(1-\\beta)^2 T}{24 f^{2-2\\beta}}\\alpha^3 + \\frac{\\rho\\beta\\nu T}{4 f^{1-\\beta}}\\alpha^2
    + \\bigl(1 + \\frac{2-3\\rho^2}{24}\\nu^2 T\\bigr)\\alpha - \\sigma f^{1-\\beta} = 0`
    with :math:`f = F + s`. The smallest positive real root is returned,
    polished by Newton steps on the formula itself.
    """
    f = F + shift
    if f <= 0 or atm_vol <= 0:
        raise ValueError("need F + shift > 0 and a positive ATM vol")
    b = 1.0 - beta
    fb = f**b
    coeffs = [
        b * b * T / (24 * fb * fb),
        rho * beta * nu * T / (4 * fb),
        1 + (2 - 3 * rho * rho) / 24 * nu * nu * T,
        -atm_vol * fb,
    ]
    while coeffs[0] == 0 and len(coeffs) > 2:
        coeffs = coeffs[1:]
    roots = np.roots(coeffs)
    real = np.sort(
        [
            r.real
            for r in np.atleast_1d(roots)
            if abs(r.imag) < 1e-12 * max(1.0, abs(r)) and r.real > 0
        ]
    )
    if real.size == 0:
        raise ValueError("no positive alpha reproduces the ATM vol")
    a = float(real[0])
    # polish against the full formula
    for _ in range(3):
        g = sabr_lognormal_vol(F, F, T, a, beta, rho, nu, shift) - atm_vol
        h = 1e-7 * a
        dg = (
            sabr_lognormal_vol(F, F, T, a + h, beta, rho, nu, shift)
            - sabr_lognormal_vol(F, F, T, a - h, beta, rho, nu, shift)
        ) / (2 * h)
        a -= g / dg
    return a


@dataclass(frozen=True)
class SABRFit:
    """Result of :func:`calibrate_sabr`."""

    model: SABR
    errors: np.ndarray
    rmse: float


def calibrate_sabr(
    F: float,
    strikes: Any,
    vols: Any,
    T: float,
    beta: float,
    shift: float = 0.0,
    vol_type: str = "lognormal",
    method: str = "hagan",
    weights: Any = None,
    atm_vol: Any = None,
) -> SABRFit:
    """Fit :math:`(\\alpha, \\rho, \\nu)` of one smile with :math:`\\beta` fixed.

    ``vol_type`` says whether ``vols`` are Black (of the shifted rate) or
    normal volatilities. If ``atm_vol`` (Black, lognormal quotes only) is
    given, :math:`\\alpha` is tied to it through :func:`alpha_from_atm`, so
    the ATM quote is matched exactly and only :math:`\\rho, \\nu` are free.
    Otherwise all three are fitted. The least-squares fit uses the
    parametrisation :math:`\\rho = \\tanh a`, :math:`\\nu = e^b`.
    """
    from scipy.optimize import least_squares

    K = np.asarray(strikes, float)
    v = np.asarray(vols, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    if vol_type not in ("lognormal", "normal"):
        raise ValueError("vol_type must be 'lognormal' or 'normal'")
    if atm_vol is not None and vol_type != "lognormal":
        raise ValueError("atm_vol pins alpha through the Black ATM formula")

    def model_vols(alpha: float, rho: float, nu: float) -> np.ndarray:
        if vol_type == "normal":
            return np.asarray(sabr_normal_vol(F, K, T, alpha, beta, rho, nu, shift), float)
        return np.asarray(sabr_lognormal_vol(F, K, T, alpha, beta, rho, nu, shift, method), float)

    def unpack(x: np.ndarray) -> tuple[float, float, float]:
        rho, nu = float(np.tanh(x[0])), float(np.exp(x[1]))
        if atm_vol is not None:
            return alpha_from_atm(float(atm_vol), F, T, beta, rho, nu, shift), rho, nu
        return float(np.exp(x[2])), rho, nu

    def resid(x: np.ndarray) -> np.ndarray:
        return w * (model_vols(*unpack(x)) - v)

    f = F + shift
    i_atm = int(np.argmin(np.abs(K - F)))
    level = v[i_atm] * (f ** (1 - beta) if vol_type == "lognormal" else f ** (-beta))
    x0 = [0.0, np.log(0.3)] + ([] if atm_vol is not None else [np.log(level)])
    sol = least_squares(resid, np.array(x0), method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    alpha, rho, nu = unpack(sol.x)
    err = model_vols(alpha, rho, nu) - v
    return SABRFit(SABR(alpha, beta, rho, nu, shift), err, float(np.sqrt(np.mean(err * err))))
