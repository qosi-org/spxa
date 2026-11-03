r"""Black, Bachelier and shifted-Black option formulas.

All formulas price options on a forward :math:`F` with expiry :math:`T`,
strike :math:`K` and discount factor :math:`P`:

Black (lognormal)
    :math:`V = P\,\eta\,[F N(\eta d_1) - K N(\eta d_2)]`,
    :math:`d_{1,2} = \frac{\ln(F/K)}{\sigma\sqrt T} \pm \tfrac12\sigma\sqrt T`.
Bachelier (normal)
    :math:`V = P\,[\eta(F - K) N(\eta d) + \sigma_N\sqrt T\,n(d)]`,
    :math:`d = \frac{F - K}{\sigma_N\sqrt T}`.
Shifted Black
    Black with :math:`F + s` and :math:`K + s` (for negative rates).

with :math:`\eta = +1` for calls and :math:`-1` for puts. Inputs broadcast
like NumPy arrays; ``option`` may be ``"call"``/``"put"`` or a boolean
array (``True`` = call). Degenerate cases (:math:`T = 0` or zero
volatility) return the discounted intrinsic value.

Greeks are with respect to the forward (``delta``, ``gamma``), the
volatility (``vega``, ``volga``, ``vanna`` = :math:`\partial^2V/\partial F\partial\sigma`)
and the strike (``dual_delta``, ``dual_gamma``); ``theta`` is
:math:`-\partial V/\partial T` at fixed forward and discount factor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr

__all__ = [
    "bachelier_greeks",
    "bachelier_price",
    "black_greeks",
    "black_price",
    "black_scholes_greeks",
    "black_scholes_price",
    "shifted_black_greeks",
    "shifted_black_price",
]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _eta(option: Any) -> np.ndarray:
    if isinstance(option, str):
        o = option.lower()
        if o in ("call", "c"):
            return np.array(1.0)
        if o in ("put", "p"):
            return np.array(-1.0)
        raise ValueError("option must be 'call' or 'put'")
    arr = np.asarray(option)
    if arr.dtype == bool:
        return np.where(arr, 1.0, -1.0)
    if arr.dtype.kind in "OUS":
        return np.vectorize(lambda o: float(_eta(str(o))))(arr)
    if not np.all(np.isin(arr, (-1, 1))):
        raise ValueError("numeric option flags must be +1 (call) or -1 (put)")
    return arr.astype(float)


def _npdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _out(x: np.ndarray) -> Any:
    return float(x) if np.ndim(x) == 0 else x


def _black_parts(F: Any, K: Any, T: Any, sigma: Any) -> tuple[np.ndarray, ...]:
    F, K, T, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, sigma))
    if np.any(F <= 0):
        raise ValueError("Black requires a positive forward (use shifted_black for F <= 0)")
    if np.any(K < 0):
        raise ValueError("Black requires a non-negative strike")
    if np.any(T < 0) or np.any(sigma < 0):
        raise ValueError("expiry and volatility must be non-negative")
    sd = sigma * np.sqrt(T)
    live = (sd > 0) & (K > 0)
    sd_safe = np.where(live, sd, 1.0)
    with np.errstate(divide="ignore"):
        lm = np.log(np.where(K > 0, F / np.where(K > 0, K, 1.0), 1.0))
    d1 = np.where(live, lm / sd_safe + 0.5 * sd_safe, 0.0)
    d2 = d1 - np.where(live, sd_safe, 0.0)
    return F, K, T, sigma, sd, live, d1, d2


def black_price(F: Any, K: Any, T: Any, sigma: Any, df: Any = 1.0, option: Any = "call") -> Any:
    """Black (1976) price of a European option on a forward."""
    eta = _eta(option)
    F, K, T, sigma, sd, live, d1, d2 = _black_parts(F, K, T, sigma)
    df = np.asarray(df, dtype=float)
    val = eta * (F * ndtr(eta * d1) - K * ndtr(eta * d2))
    intrinsic = np.maximum(eta * (F - K), 0.0)
    val = _repair_cancellation(F, K, sd, live, d1, d2, eta, val)
    return _out(df * np.where(live, val, intrinsic))


# Relative accuracy lost by F N(.) - K N(.) is about (larger term) / (result);
# beyond this amplification the out-of-the-money value is recomputed with the
# cancellation-free normalised formulation of spxa.market.implied.
_MAX_AMPLIFICATION = 64.0


def _repair_cancellation(
    F: np.ndarray,
    K: np.ndarray,
    sd: np.ndarray,
    live: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    eta: np.ndarray,
    val: np.ndarray,
) -> np.ndarray:
    """Recompute prices whose closed form cancels catastrophically."""
    shape = np.broadcast_shapes(F.shape, K.shape, sd.shape, eta.shape, val.shape)
    Fb, Kb, sdb, liveb, d1b, d2b = np.broadcast_arrays(F, K, sd, live, d1, d2)
    eo = np.where(Kb >= Fb, 1.0, -1.0)  # out-of-the-money side
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        big = Fb * ndtr(eo * d1b)
        otm = eo * (big - Kb * ndtr(eo * d2b))
        bad = liveb & ~(otm * _MAX_AMPLIFICATION >= big)
    if not np.any(bad):
        return val
    from spxa.market.implied import normalised_black_call

    out = np.array(np.broadcast_to(val, shape), dtype=float)
    etab = np.broadcast_to(eta, shape)
    Fs, Ks, sds, es = Fb[bad], Kb[bad], sdb[bad], etab[bad]
    x = np.log1p((Fs - Ks) / Ks)  # F - K is exact near the money, log(F/K) is not
    # b(-|x|, s) is the normalised out-of-the-money value
    otm_fixed = np.sqrt(Fs * Ks) * np.atleast_1d(normalised_black_call(-np.abs(x), sds))
    out[bad] = otm_fixed + np.maximum(es * (Fs - Ks), 0.0)
    return out


def black_greeks(
    F: Any, K: Any, T: Any, sigma: Any, df: Any = 1.0, option: Any = "call"
) -> dict[str, Any]:
    """Black price and Greeks (see module docstring)."""
    eta = _eta(option)
    F, K, T, sigma, sd, live, d1, d2 = _black_parts(F, K, T, sigma)
    df = np.asarray(df, dtype=float)
    n1 = _npdf(d1)
    sqT = np.sqrt(T)
    sd_safe = np.where(live, sd, 1.0)
    sig_safe = np.where(live, sigma, 1.0)
    price = black_price(F, K, T, sigma, df, eta)
    itm = (eta * (F - K) > 0).astype(float)
    delta = np.where(live, eta * ndtr(eta * d1), eta * itm)
    gamma = np.where(live, n1 / (F * sd_safe), 0.0)
    vega = np.where(live, F * n1 * sqT, 0.0)
    volga = np.where(live, vega * d1 * d2 / sig_safe, 0.0)
    vanna = np.where(live, -n1 * d2 / sig_safe, 0.0)
    dual_delta = np.where(live, -eta * ndtr(eta * d2), -eta * itm)
    K_safe = np.where(K > 0, K, 1.0)
    dual_gamma = np.where(live, n1 * F / (K_safe * K_safe * sd_safe), 0.0)
    theta = np.where(live, -F * n1 * sig_safe / (2.0 * np.where(T > 0, sqT, 1.0)), 0.0)
    return {
        "price": price,
        "delta": _out(df * delta),
        "gamma": _out(df * gamma),
        "vega": _out(df * vega),
        "volga": _out(df * volga),
        "vanna": _out(df * vanna),
        "dual_delta": _out(df * dual_delta),
        "dual_gamma": _out(df * dual_gamma),
        "theta": _out(df * theta),
    }


def shifted_black_price(
    F: Any, K: Any, T: Any, sigma: Any, shift: Any, df: Any = 1.0, option: Any = "call"
) -> Any:
    """Shifted-lognormal (displaced-diffusion) price: Black on ``F + shift``, ``K + shift``."""
    s = np.asarray(shift, dtype=float)
    return black_price(np.asarray(F) + s, np.asarray(K) + s, T, sigma, df, option)


def shifted_black_greeks(
    F: Any, K: Any, T: Any, sigma: Any, shift: Any, df: Any = 1.0, option: Any = "call"
) -> dict[str, Any]:
    """Greeks of :func:`shifted_black_price` (derivatives in ``F`` and ``K`` are unchanged by the shift)."""
    s = np.asarray(shift, dtype=float)
    return black_greeks(np.asarray(F) + s, np.asarray(K) + s, T, sigma, df, option)


def bachelier_price(
    F: Any, K: Any, T: Any, sigma_n: Any, df: Any = 1.0, option: Any = "call"
) -> Any:
    """Bachelier (normal-model) price; ``sigma_n`` is the absolute (normal) volatility."""
    eta = _eta(option)
    F, K, T, sigma_n = (np.asarray(x, dtype=float) for x in (F, K, T, sigma_n))
    if np.any(T < 0) or np.any(sigma_n < 0):
        raise ValueError("expiry and volatility must be non-negative")
    df = np.asarray(df, dtype=float)
    sd = sigma_n * np.sqrt(T)
    live = sd > 0
    sd_safe = np.where(live, sd, 1.0)
    d = (F - K) / sd_safe
    val = eta * (F - K) * ndtr(eta * d) + sd_safe * _npdf(d)
    return _out(df * np.where(live, val, np.maximum(eta * (F - K), 0.0)))


def bachelier_greeks(
    F: Any, K: Any, T: Any, sigma_n: Any, df: Any = 1.0, option: Any = "call"
) -> dict[str, Any]:
    """Bachelier price and Greeks."""
    eta = _eta(option)
    F, K, T, sigma_n = (np.asarray(x, dtype=float) for x in (F, K, T, sigma_n))
    df = np.asarray(df, dtype=float)
    price = bachelier_price(F, K, T, sigma_n, df, eta)
    sqT = np.sqrt(T)
    sd = sigma_n * sqT
    live = sd > 0
    sd_safe = np.where(live, sd, 1.0)
    sig_safe = np.where(live, sigma_n, 1.0)
    d = (F - K) / sd_safe
    n = _npdf(d)
    itm = (eta * (F - K) > 0).astype(float)
    delta = np.where(live, eta * ndtr(eta * d), eta * itm)
    gamma = np.where(live, n / sd_safe, 0.0)
    vega = np.where(live, sqT * n, 0.0)
    volga = np.where(live, vega * d * d / sig_safe, 0.0)
    vanna = np.where(live, -d * n / sig_safe, 0.0)
    theta = np.where(live, -sigma_n * n / (2.0 * np.where(T > 0, sqT, 1.0)), 0.0)
    return {
        "price": price,
        "delta": _out(df * delta),
        "gamma": _out(df * gamma),
        "vega": _out(df * vega),
        "volga": _out(df * volga),
        "vanna": _out(df * vanna),
        "dual_delta": _out(-df * delta),
        "dual_gamma": _out(df * gamma),
        "theta": _out(df * theta),
    }


def black_scholes_price(
    S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any = 0.0, option: Any = "call"
) -> Any:
    """Black–Scholes–Merton price with continuous rate ``r`` and yield ``q``."""
    S, T, r, q = (np.asarray(x, dtype=float) for x in (S, T, r, q))
    F = S * np.exp((r - q) * T)
    return black_price(F, K, T, sigma, np.exp(-r * T), option)


def black_scholes_greeks(
    S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any = 0.0, option: Any = "call"
) -> dict[str, Any]:
    """Spot Greeks of :func:`black_scholes_price`.

    Returns ``price``, ``delta`` and ``gamma`` (in spot), ``vega``,
    ``theta`` (calendar decay :math:`-\\partial V/\\partial T` at fixed spot),
    ``rho`` (:math:`\\partial V/\\partial r`) and ``epsilon``
    (:math:`\\partial V/\\partial q`).
    """
    eta = _eta(option)
    S, K_, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    df, qf = np.exp(-r * T), np.exp(-q * T)
    F = S * qf / df
    g = black_greeks(F, K_, T, sigma, df, eta)
    price = np.asarray(g["price"])
    fwd_delta = np.asarray(g["delta"])  # dV/dF including df
    delta = fwd_delta * qf / df
    gamma = np.asarray(g["gamma"]) * (qf / df) ** 2
    # dV/dT at fixed S: Black theta (fixed F, df) plus carry terms
    theta = np.asarray(g["theta"]) + r * price - fwd_delta * F * (r - q)
    rho = T * (fwd_delta * F - price)
    epsilon = -T * fwd_delta * F
    return {
        "price": _out(price),
        "delta": _out(delta),
        "gamma": _out(gamma),
        "vega": g["vega"],
        "theta": _out(theta),
        "rho": _out(rho),
        "epsilon": _out(epsilon),
    }
