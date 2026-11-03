r"""Black deltas and strike-from-delta conversion.

Smiles are often quoted by delta rather than by strike, for example 25Δ
risk reversals or the ATM delta-neutral straddle. With
:math:`d_{1,2} = -k/\sqrt w \pm \sqrt w/2`, :math:`k = \log(K/F)`,
:math:`w = \sigma^2 T` and :math:`\eta = \pm 1` for calls and puts:

forward delta
    :math:`\Delta_F = \eta N(\eta d_1)`.
spot delta
    :math:`\Delta_S = Q\,\Delta_F` with :math:`Q = F D / S`, the carry
    discount (:math:`e^{-qT}` for a dividend yield).
premium-adjusted
    The premium is taken out of the hedge:
    :math:`\Delta^{pa}_F = \eta\,(K/F)\,N(\eta d_2)`, and
    :math:`\Delta^{pa}_S = Q\,\Delta^{pa}_F`.

Without premium adjustment and at a fixed volatility the inversion is
explicit:
:math:`k = -\eta\,N^{-1}(\eta\Delta_F)\sqrt w + w/2`. Premium-adjusted call
deltas rise and then fall in :math:`K`. Their inversion is therefore taken
on the branch above the strike of maximal delta, which is the market
convention. With a smile :math:`\sigma(k)` the equation is solved by
Brent's method.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Union

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtr, ndtri

__all__ = ["black_delta", "strike_from_delta", "atm_dns_strike"]

VolLike = Union[float, Callable[[float], float]]


def _eta(option: str) -> float:
    o = option.lower()
    if o in ("call", "c"):
        return 1.0
    if o in ("put", "p"):
        return -1.0
    raise ValueError("option must be 'call' or 'put'")


def _check_kind(kind: str) -> None:
    if kind not in ("forward", "spot"):
        raise ValueError("kind must be 'forward' or 'spot'")


def black_delta(
    F: Any,
    K: Any,
    T: float,
    sigma: Any,
    option: str = "call",
    kind: str = "forward",
    premium_adjusted: bool = False,
    carry: float = 1.0,
) -> Any:
    r"""Black delta in the requested convention.

    ``carry`` is :math:`Q = F D/S`, used only for ``kind="spot"``.
    """
    _check_kind(kind)
    eta = _eta(option)
    FF, KK, ss = (np.asarray(x, float) for x in (F, K, sigma))
    sw = ss * math.sqrt(T)
    k = np.log(KK / FF)
    d1 = -k / sw + 0.5 * sw
    if premium_adjusted:
        d = eta * (KK / FF) * ndtr(eta * (d1 - sw))
    else:
        d = eta * ndtr(eta * d1)
    if kind == "spot":
        d = d * carry
    return float(d) if np.ndim(d) == 0 else d


def _vol_at(sigma: VolLike, k: float) -> float:
    return float(sigma(k)) if callable(sigma) else float(sigma)


def strike_from_delta(
    delta: float,
    F: float,
    T: float,
    sigma: VolLike,
    option: str = "call",
    kind: str = "forward",
    premium_adjusted: bool = False,
    carry: float = 1.0,
) -> float:
    r"""Strike with the given Black delta.

    ``sigma`` is a volatility, or a smile :math:`k \mapsto \sigma(k)` in
    log-forward moneyness. ``delta`` follows the sign convention of
    :func:`black_delta` (negative for puts).
    """
    _check_kind(kind)
    eta = _eta(option)
    if T <= 0:
        raise ValueError("T must be positive")
    dF = delta / carry if kind == "spot" else delta
    if not 0 < eta * dF < 1:
        raise ValueError("delta out of range for this option type")
    sqT = math.sqrt(T)

    if not premium_adjusted and not callable(sigma):
        sw = float(sigma) * sqT
        return F * math.exp(-eta * float(ndtri(eta * dF)) * sw + 0.5 * sw * sw)

    def f(k: float) -> float:
        s = _vol_at(sigma, k)
        return float(black_delta(1.0, math.exp(k), T, s, option, "forward", premium_adjusted)) - dF

    s0 = _vol_at(sigma, 0.0) * sqT
    lo_k, hi_k = -12.0 * max(s0, 0.05), 12.0 * max(s0, 0.05)
    if premium_adjusted and eta > 0:
        # PA call delta rises then falls in k: use the branch above its maximum
        m = minimize_scalar(lambda k: -f(k), bounds=(lo_k, hi_k), method="bounded")
        if -m.fun < 0:
            raise ValueError("premium-adjusted call delta never reaches this value")
        lo_k = float(m.x)
    flo, fhi = f(lo_k), f(hi_k)
    if flo * fhi > 0:
        raise ValueError("could not bracket the strike for this delta")
    k = brentq(f, lo_k, hi_k, xtol=1e-15, rtol=8.9e-16, maxiter=200)
    return F * math.exp(k)


def atm_dns_strike(F: float, T: float, sigma: VolLike, premium_adjusted: bool = False) -> float:
    r"""ATM delta-neutral-straddle strike: call delta + put delta = 0.

    At a fixed volatility this is :math:`F e^{w/2}`, or :math:`F e^{-w/2}`
    with premium adjustment. With a smile the equation
    :math:`k = \pm\sigma(k)^2 T/2` is solved by Brent's method.
    """
    sgn = -1.0 if premium_adjusted else 1.0

    def g(k: float) -> float:
        return k - sgn * 0.5 * _vol_at(sigma, k) ** 2 * T

    if not callable(sigma):
        return F * math.exp(sgn * 0.5 * float(sigma) ** 2 * T)
    w0 = _vol_at(sigma, 0.0) ** 2 * T
    a, b = -2.0 * w0 - 1e-3, 2.0 * w0 + 1e-3
    return F * math.exp(brentq(g, a, b, xtol=1e-15, rtol=8.9e-16))
