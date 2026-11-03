r"""Implied volatility to machine precision.

Black implied volatility
------------------------
Following Jäckel ("Let's Be Rational", 2015) the problem is normalised:
with :math:`x = \ln(F/K)`, total volatility :math:`s = \sigma\sqrt T` and
normalised price :math:`\beta = V/(P\sqrt{FK})`,

.. math::
    b(x, s) = e^{x/2}\,\Phi\!\left(\tfrac{x}{s} + \tfrac{s}{2}\right)
             - e^{-x/2}\,\Phi\!\left(\tfrac{x}{s} - \tfrac{s}{2}\right)

is the normalised call price. In-the-money prices are reduced to
out-of-the-money ones by put–call parity, and calls and puts by the
symmetry :math:`b(x, s) = b(-x, s) + e^{x/2} - e^{-x/2}`, so only
:math:`x \le 0` calls are solved.

The direct formula suffers catastrophic cancellation for deep
out-of-the-money options at small :math:`s`. There the price is computed
from its derivative, :math:`\partial b/\partial s = e^{x/2}\varphi(x/s + s/2)`,
as the integral :math:`b(x, s) = \int_0^s e^{x/2}\varphi(x/u + u/2)\,du`
of a positive integrand, which has no cancellation. The equation
:math:`\log b(x, s) = \log\beta` is then solved by Newton's method on a
bracket that shrinks monotonically (bisection safeguard), giving relative
accuracy close to machine precision wherever :math:`\beta` is representable.

Normal (Bachelier) implied volatility
-------------------------------------
With :math:`d = (F-K)/(\sigma_N\sqrt T)` the out-of-the-money normalised
price is :math:`\psi(d) = d\,\Phi(d) + \varphi(d)` for :math:`d \le 0`,
with :math:`\psi'(d) = \Phi(d)`. The same cancellation-free evaluation
(:math:`\psi(d) = \int_{|d|}^\infty (z - |d|)\varphi(z)\,dz`) and safeguarded
Newton iteration are used (cf. Jäckel, "Implied Normal Volatility", 2017).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.special import erfcx, erfinv, log_ndtr

__all__ = [
    "implied_black_volatility",
    "implied_normal_volatility",
    "normalised_black_call",
]

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_HALF_PI = math.sqrt(0.5 * math.pi)
_EPS = np.finfo(float).eps
_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_GL_PANELS = 24
_LAG_X, _LAG_W = np.polynomial.laguerre.laggauss(60)


def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / _SQRT_2PI


def _log_phi(z: float) -> float:
    return -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)


# ----------------------------------------------------------------------
# Normalised Black price for x <= 0 (out-of-the-money call)
# ----------------------------------------------------------------------


def _log_b_otm(x: float, s: float) -> float:
    """``log b(x, s)`` for ``x <= 0``, ``s > 0``, accurate to a few ulps."""
    if s <= 0.0:
        return -math.inf
    if x == 0.0:
        return math.log(math.erf(s / (2.0 * math.sqrt(2.0))))
    h, t = x / s, 0.5 * s
    hp, hm = h + t, h - t
    # direct formula through the log-CDF: b = e^A (1 - e^{B - A})
    a = 0.5 * x + float(log_ndtr(hp))
    b = -0.5 * x + float(log_ndtr(hm))
    diff = b - a  # < 0
    if diff < -1e-3:
        # relative error of -expm1(diff) is ~ eps * (|A| + |B|) / |diff|
        if _EPS * (abs(a) + abs(b) + 1.0) / abs(diff) < 1e-15:
            return a + math.log(-math.expm1(diff))
    if hp < 0.0:
        # erfcx form: b = e^{x/2} phi(h+t) sqrt(pi/2) [erfcx(-hp/r2) - erfcx(-hm/r2)]
        u, v = -hp / math.sqrt(2.0), -hm / math.sqrt(2.0)
        eu, ev = float(erfcx(u)), float(erfcx(v))
        if (eu - ev) > 0.1 * eu:
            return 0.5 * x + _log_phi(hp) + math.log(_SQRT_HALF_PI * (eu - ev))
    return _log_b_integral(x, s)


def _log_b_integral(x: float, s: float) -> float:
    r"""``log b`` from :math:`b = \int_0^s e^{x/2}\varphi(x/u + u/2)\,du` (no cancellation).

    The integrand is scaled by its value at ``s`` so the quadrature works on
    O(1) numbers even when ``b`` is far below the smallest normal double.
    """
    ref = 0.5 * x + _log_phi(x / s + 0.5 * s)

    lo = max(_lower_cut(x, s, ref), 1e-12 * s)
    # composite Gauss-Legendre on geometrically spaced panels (smooth in log u)
    edges = np.geomspace(lo, s, _GL_PANELS + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
    half = 0.5 * (edges[1:] - edges[:-1])[:, None]
    u = mid + half * _GL_X[None, :]
    z = x / u + 0.5 * u
    vals = np.exp(0.5 * x - 0.5 * z * z - 0.5 * math.log(2.0 * math.pi) - ref)
    val = float(np.sum(half * _GL_W[None, :] * vals))
    return ref + math.log(val) if val > 0.0 else -math.inf


def _lower_cut(x: float, s: float, ref: float) -> float:
    def g(u: float) -> float:
        return 0.5 * x + _log_phi(x / u + 0.5 * u) - ref + 60.0

    if g(1e-300 + s * 1e-12) > 0:
        return 0.0
    lo, hi = s * 1e-12, s
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1 + 1e-6:
            break
    return lo


def normalised_black_call(x: Any, s: Any) -> Any:
    r"""Normalised Black call :math:`b(x, s)` (see module docstring), vectorised."""
    xa, sa = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(s, dtype=float))
    out = np.empty(xa.shape)
    for idx in np.ndindex(xa.shape):
        xi, si = float(xa[idx]), float(sa[idx])
        if si < 0:
            raise ValueError("total volatility must be non-negative")
        intrinsic = max(math.exp(0.5 * xi) - math.exp(-0.5 * xi), 0.0)
        otm = math.exp(_log_b_otm(-abs(xi), si)) if si > 0 else 0.0
        out[idx] = otm + intrinsic
    return float(out) if out.ndim == 0 else out


# ----------------------------------------------------------------------
# Safeguarded Newton on log price
# ----------------------------------------------------------------------


def _solve(
    log_target: float,
    log_f: Callable[[float], float],
    dlog_f: Callable[[float, float], float],
    s0: float,
    lo: float,
    hi: float,
) -> float:
    """Solve ``log_f(s) = log_target`` for increasing ``log_f`` on ``(lo, hi)``."""
    s = s0 if lo < s0 < hi else 0.5 * (lo + hi)
    for _ in range(200):
        val = log_f(s)
        err = val - log_target
        if err > 0:
            hi = s
        else:
            lo = s
        if abs(err) <= 2 * _EPS * max(1.0, abs(log_target)):
            return s
        step = -err / dlog_f(s, val)
        new = s + step
        if not (lo < new < hi) or not math.isfinite(new):
            new = 0.5 * (lo + hi) if math.isfinite(hi) else 2.0 * s
        if abs(new - s) <= 2 * _EPS * s:
            return new
        s = new
    return s


def _implied_black_scalar(beta: float, x: float) -> float:
    """Total implied volatility for an OTM (x <= 0) normalised call price."""
    if beta <= 0.0:
        return 0.0
    if beta >= math.exp(0.5 * x):
        return math.nan
    if x == 0.0:
        # b(0, s) = erf(s / (2 sqrt 2)) inverts in closed form
        return 2.0 * math.sqrt(2.0) * float(erfinv(beta))
    log_beta = math.log(beta)

    def dlog(s: float, log_val: float) -> float:
        return math.exp(0.5 * x + _log_phi(x / s + 0.5 * s) - log_val)

    # start at the inflection point s_c = sqrt(2|x|) of b(x, .) and bracket the root
    s0 = math.sqrt(2.0 * abs(x))
    lo, hi = 0.0, s0
    while _log_b_otm(x, hi) < log_beta:
        lo, hi = hi, 2.0 * hi
        if hi > 1e6:
            return math.nan
    return _solve(log_beta, lambda s: _log_b_otm(x, s), dlog, min(s0, hi), lo, hi)


def _as_eta(option: Any, shape: tuple[int, ...]) -> np.ndarray:
    from spxa.market.black import _eta

    return np.broadcast_to(_eta(option), shape).astype(float)


def implied_black_volatility(
    price: Any, F: Any, K: Any, T: Any, df: Any = 1.0, option: Any = "call", errors: str = "nan"
) -> Any:
    """Black implied volatility of option prices on a forward.

    Parameters
    ----------
    price, F, K, T, df : array-like
        Premium, forward, strike, expiry (years) and discount factor.
    option : str, bool array or +/-1 array
        ``"call"``/``"put"`` (broadcast with the other inputs).
    errors : {"nan", "raise"}
        Prices outside the no-arbitrage bounds
        :math:`[P\\max(\\eta(F-K), 0),\\ P F]` (calls) or
        :math:`[\\cdot,\\ P K]` (puts) give NaN, or raise ``ValueError``.

    Returns
    -------
    float or ndarray
        Implied volatility :math:`\\sigma`; 0 for a price equal to intrinsic.
    """
    if errors not in ("nan", "raise"):
        raise ValueError("errors must be 'nan' or 'raise'")
    arrs = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (price, F, K, T, df)))
    p, f, k, t, d = arrs
    eta = _as_eta(option, p.shape)
    out = np.empty(p.shape)
    for idx in np.ndindex(p.shape):
        pi, fi, ki, ti, di, ei = (float(a[idx]) for a in (p, f, k, t, d, eta))
        if fi <= 0 or ki <= 0 or ti <= 0 or di <= 0:
            raise ValueError("F, K, T and df must be positive")
        x = math.log(fi / ki)
        beta = pi / (di * math.sqrt(fi * ki))
        # convert to an OTM call with x <= 0: subtract intrinsic, use the symmetry
        intrinsic = max(ei * (math.exp(0.5 * x) - math.exp(-0.5 * x)), 0.0)
        otm = beta - intrinsic
        tol = 8 * _EPS * max(1.0, beta)
        cap = math.exp(0.5 * x) if ei > 0 else math.exp(-0.5 * x)
        if otm < -tol or beta > cap * (1 + 8 * _EPS):
            if errors == "raise":
                raise ValueError(f"price {pi} violates the no-arbitrage bounds")
            out[idx] = math.nan
            continue
        s = _implied_black_scalar(max(otm, 0.0), -abs(x))
        if math.isnan(s) and errors == "raise":
            raise ValueError(f"no implied volatility for price {pi}")
        out[idx] = s / math.sqrt(ti)
    return float(out) if out.ndim == 0 else out


# ----------------------------------------------------------------------
# Normal (Bachelier) implied volatility
# ----------------------------------------------------------------------


def _log_psi_neg(u: float) -> float:
    r"""``log psi(-u)`` for ``u >= 0`` with :math:`\psi(d) = d\Phi(d) + \varphi(d)`.

    For moderate ``u`` the direct form :math:`\varphi(u)(1 - uM(u))` with the
    Mills ratio :math:`M(u) = \Phi(-u)/\varphi(u)` is accurate. For large
    ``u`` it cancels, and the positive representation

    .. math::
        \psi(-u) = \frac{\varphi(u)}{u}\int_0^\infty
        M\!\left(u + \tfrac{w}{u}\right) e^{-w - w^2/(2u^2)}\,dw

    is evaluated by Gauss–Laguerre quadrature.
    """
    if u <= 4.0:
        mills = _SQRT_HALF_PI * float(erfcx(u / math.sqrt(2.0)))
        return _log_phi(u) + math.log(1.0 - u * mills)
    w = _LAG_X
    z = u + w / u
    vals = _SQRT_HALF_PI * erfcx(z / math.sqrt(2.0)) * np.exp(-0.5 * (w / u) ** 2)
    return _log_phi(u) - math.log(u) + math.log(float(np.dot(_LAG_W, vals)))


def _implied_normal_scalar(v: float, a: float) -> float:
    """Total normal volatility ``sd`` with ``sd * psi(-a/sd) = v`` (``a >= 0``)."""
    if v <= 0.0:
        return 0.0
    if a == 0.0:
        return v * _SQRT_2PI
    # in u = a / sd: g(u) = psi(-u) / u = v / a, strictly decreasing from +inf to 0
    log_target = math.log(v / a)

    def log_g(u: float) -> float:
        return _log_psi_neg(u) - math.log(u)

    def dlog_g(u: float, log_val: float) -> float:
        # d/du log psi(-u) = -Phi(-u)/psi(-u); Phi(-u) = phi(u) M(u)
        mills = _SQRT_HALF_PI * float(erfcx(u / math.sqrt(2.0)))
        return -math.exp(_log_phi(u) + math.log(mills) - _log_psi_neg(u)) - 1.0 / u

    # solve the increasing function -log g in u
    lo, hi = 0.0, 1.0
    while -log_g(hi) < -log_target:
        lo, hi = hi, 2.0 * hi
        if hi > 1e8:
            return math.nan
    u0 = 0.5 * (lo + hi) if lo > 0 else hi / 2
    u = _solve(-log_target, lambda x: -log_g(x), lambda x, val: -dlog_g(x, -val), u0, lo, hi)
    return a / u


def implied_normal_volatility(
    price: Any, F: Any, K: Any, T: Any, df: Any = 1.0, option: Any = "call", errors: str = "nan"
) -> Any:
    r"""Bachelier (normal) implied volatility.

    Parameters
    ----------
    price, F, K, T, df : array-like
        Premium, forward, strike (any sign), expiry and discount factor.
    option : str, bool array or +/-1 array
    errors : {"nan", "raise"}
        Prices below the discounted intrinsic value give NaN or raise.

    Returns
    -------
    float or ndarray
        :math:`\sigma_N` in absolute (price) units per square-root year.
    """
    if errors not in ("nan", "raise"):
        raise ValueError("errors must be 'nan' or 'raise'")
    arrs = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (price, F, K, T, df)))
    p, f, k, t, d = arrs
    eta = _as_eta(option, p.shape)
    out = np.empty(p.shape)
    for idx in np.ndindex(p.shape):
        pi, fi, ki, ti, di, ei = (float(a[idx]) for a in (p, f, k, t, d, eta))
        if ti <= 0 or di <= 0:
            raise ValueError("T and df must be positive")
        intrinsic = max(ei * (fi - ki), 0.0)
        v = pi / di - intrinsic
        if v < -8 * _EPS * max(1.0, abs(pi / di)):
            if errors == "raise":
                raise ValueError(f"price {pi} is below the intrinsic value")
            out[idx] = math.nan
            continue
        out[idx] = _implied_normal_scalar(max(v, 0.0), abs(fi - ki)) / math.sqrt(ti)
    return float(out) if out.ndim == 0 else out
