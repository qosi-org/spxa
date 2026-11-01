r"""Closed-form and semi-analytic reference prices used to verify the PDE solvers.

Every function here is independent of the finite-difference machinery, so
it can serve as an external benchmark:

* Black–Scholes prices and Greeks, digitals and asset-or-nothing options.
* Reiner–Rubinstein (1991) single-barrier formulas in Haug's notation,
  with rebates.
* Eigenfunction (sine-series) expansion for double-barrier knock-outs.
* Cox–Ross–Rubinstein and Leisen–Reimer binomial trees for American
  options, and the perpetual American put.
* Lewis (2001) Fourier formula for exponential Lévy and Heston models, and
  Merton's (1976) jump-diffusion series.

Conventions: ``S`` spot, ``K`` strike, ``T`` maturity in years, ``r``
continuously compounded rate, ``q`` continuous dividend yield, ``sigma``
volatility. Prices are per unit notional.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from spxa.pde._utils import check_option_type, check_positive

# =====================================================================================
# Black–Scholes
# =====================================================================================


def _d1d2(S: Any, K: Any, T: float, r: float, sigma: float, q: float) -> tuple[Any, Any]:
    sq = sigma * np.sqrt(T)
    d1 = (np.log(np.asarray(S, dtype=float) / K) + (r - q + 0.5 * sigma**2) * T) / sq
    return d1, d1 - sq


def bs_price(
    S: Any, K: Any, T: float, r: float, sigma: float, q: float = 0.0, option: str = "call"
) -> Any:
    """Black–Scholes–Merton price of a European call or put."""
    check_positive("T", T)
    check_positive("sigma", sigma)
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    S = np.asarray(S, dtype=float)
    if check_option_type(option) == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option: str = "call"
) -> dict[str, float]:
    """Price and Greeks (delta, gamma, vega, theta, rho) of a European option.

    ``theta`` is the calendar-time derivative per year and ``vega``/``rho``
    are per unit change of ``sigma``/``r``.
    """
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    pdf = float(norm.pdf(d1))
    gamma = dq * pdf / (S * sigma * math.sqrt(T))
    vega = S * dq * pdf * math.sqrt(T)
    decay = -S * dq * pdf * sigma / (2.0 * math.sqrt(T))
    if check_option_type(option) == "call":
        delta = dq * float(norm.cdf(d1))
        theta = decay - r * K * dr * float(norm.cdf(d2)) + q * S * dq * float(norm.cdf(d1))
        rho = K * T * dr * float(norm.cdf(d2))
    else:
        delta = -dq * float(norm.cdf(-d1))
        theta = decay + r * K * dr * float(norm.cdf(-d2)) - q * S * dq * float(norm.cdf(-d1))
        rho = -K * T * dr * float(norm.cdf(-d2))
    return {
        "price": float(bs_price(S, K, T, r, sigma, q, option)),
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def bs_digital_price(
    S: Any,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
    cash: float = 1.0,
) -> Any:
    """Cash-or-nothing digital paying ``cash`` at expiry."""
    _, d2 = _d1d2(S, K, T, r, sigma, q)
    sgn = 1.0 if check_option_type(option) == "call" else -1.0
    return cash * math.exp(-r * T) * norm.cdf(sgn * d2)


def bs_asset_or_nothing_price(
    S: Any, K: float, T: float, r: float, sigma: float, q: float = 0.0, option: str = "call"
) -> Any:
    """Asset-or-nothing digital paying ``S_T`` at expiry."""
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    sgn = 1.0 if check_option_type(option) == "call" else -1.0
    return np.asarray(S, dtype=float) * math.exp(-q * T) * norm.cdf(sgn * d1)


# =====================================================================================
# Reiner–Rubinstein single barriers
# =====================================================================================

_BARRIER_TYPES = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


def barrier_price(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
    barrier_type: str = "down-and-out",
    rebate: float = 0.0,
    rebate_timing: str = "hit",
) -> float:
    r"""Continuously monitored single-barrier option (Reiner & Rubinstein, 1991).

    Parameters
    ----------
    barrier_type : {'down-and-out', 'down-and-in', 'up-and-out', 'up-and-in'}
    rebate : float
        Cash rebate. Knock-in options pay it at expiry if the barrier was
        never hit. Knock-out options pay it when the barrier is hit
        (``rebate_timing='hit'``) or at expiry (``'expiry'``).

    Notes
    -----
    Uses the building blocks :math:`A, \dots, F` of Haug (2007), §4.17.1,
    with cost of carry :math:`b = r - q`. If the barrier has already been
    breached, knock-outs are worth their rebate and knock-ins the vanilla.
    """
    opt = check_option_type(option)
    bt = str(barrier_type).lower()
    if bt not in _BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {_BARRIER_TYPES}")
    if rebate_timing not in ("hit", "expiry"):
        raise ValueError("rebate_timing must be 'hit' or 'expiry'")
    for name, v in (("S", S), ("K", K), ("H", H), ("T", T), ("sigma", sigma)):
        check_positive(name, v)
    down = bt.startswith("down")
    knock_out = bt.endswith("out")
    breached = S <= H if down else S >= H
    vanilla_px = float(bs_price(S, K, T, r, sigma, q, opt))
    if breached:
        if knock_out:
            return rebate if rebate_timing == "hit" else rebate * math.exp(-r * T)
        return vanilla_px

    b = r - q
    sq = sigma * math.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    lam = math.sqrt(mu**2 + 2.0 * r / sigma**2)
    phi = 1.0 if opt == "call" else -1.0
    eta = 1.0 if down else -1.0
    x1 = math.log(S / K) / sq + (1 + mu) * sq
    x2 = math.log(S / H) / sq + (1 + mu) * sq
    y1 = math.log(H**2 / (S * K)) / sq + (1 + mu) * sq
    y2 = math.log(H / S) / sq + (1 + mu) * sq
    z = math.log(H / S) / sq + lam * sq
    N = norm.cdf
    carry, disc = math.exp((b - r) * T), math.exp(-r * T)
    hs = H / S
    A = phi * S * carry * N(phi * x1) - phi * K * disc * N(phi * x1 - phi * sq)
    B = phi * S * carry * N(phi * x2) - phi * K * disc * N(phi * x2 - phi * sq)
    C = phi * S * carry * hs ** (2 * (mu + 1)) * N(eta * y1) - phi * K * disc * hs ** (2 * mu) * N(
        eta * y1 - eta * sq
    )
    D = phi * S * carry * hs ** (2 * (mu + 1)) * N(eta * y2) - phi * K * disc * hs ** (2 * mu) * N(
        eta * y2 - eta * sq
    )
    # E: rebate at expiry if not hit; F: rebate at the hitting time
    E_unit = disc * (N(eta * x2 - eta * sq) - hs ** (2 * mu) * N(eta * y2 - eta * sq))
    F_unit = hs ** (mu + lam) * N(eta * z) + hs ** (mu - lam) * N(eta * z - 2 * eta * lam * sq)
    above = K > H
    if knock_out:
        if opt == "call":
            if down:
                core = A - C if above else B - D
            else:
                core = 0.0 if above else A - B + C - D
        else:
            if down:
                core = A - B + C - D if above else 0.0
            else:
                core = B - D if above else A - C
        if rebate_timing == "hit":
            reb = rebate * F_unit
        else:
            reb = rebate * (disc - E_unit)
        return float(core + reb)
    if opt == "call":
        if down:
            core = C if above else A - B + D
        else:
            core = A if above else B - C + D
    else:
        if down:
            core = B - C + D if above else A
        else:
            core = A - B + D if above else C
    return float(core + rebate * E_unit)


# =====================================================================================
# Double barrier knock-out: eigenfunction expansion
# =====================================================================================


def double_barrier_price(
    S: float,
    K: float,
    L: float,
    U: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
    tol: float = 1e-14,
    max_terms: int = 2000,
) -> float:
    r"""Double knock-out call or put (no rebate) by eigenfunction expansion.

    In :math:`x = \log S` with drift :math:`\nu = r - q - \tfrac12\sigma^2`,
    the transition density killed at :math:`l = \log L`, :math:`u = \log U`
    (:math:`W = u - l`) is

    .. math::
        p(x, y, T) = e^{\nu (y - x)/\sigma^2 - \nu^2 T / (2\sigma^2)}
        \frac{2}{W} \sum_{n \ge 1} e^{-\frac12 \sigma^2 (n\pi/W)^2 T}
        \sin\frac{n\pi (x - l)}{W} \sin\frac{n\pi (y - l)}{W},

    and the price is :math:`e^{-rT} \int_l^u p(x, y, T) g(e^y)\,dy`. The
    payoff integrals use high-order Gauss–Legendre quadrature on the smooth
    part of the payoff, and the series is summed until the terms fall below
    ``tol``.
    """
    opt = check_option_type(option)
    for name, v in (("S", S), ("K", K), ("L", L), ("U", U), ("T", T), ("sigma", sigma)):
        check_positive(name, v)
    if not L < U:
        raise ValueError("need L < U")
    if not (L < S < U):
        return 0.0
    lo, hi = math.log(L), math.log(U)
    W = hi - lo
    x = math.log(S)
    nu = r - q - 0.5 * sigma**2
    # payoff support inside the corridor
    k = math.log(K)
    a, b = (max(lo, k), hi) if opt == "call" else (lo, min(hi, k))
    if a >= b:
        return 0.0
    n_q = 400
    gx, gw = np.polynomial.legendre.leggauss(n_q)
    y = 0.5 * (b - a) * gx + 0.5 * (a + b)
    w = 0.5 * (b - a) * gw
    pay = np.maximum(np.exp(y) - K, 0.0) if opt == "call" else np.maximum(K - np.exp(y), 0.0)
    weight = w * pay * np.exp(nu * (y - x) / sigma**2)
    pref = math.exp(-r * T - nu**2 * T / (2 * sigma**2)) * 2.0 / W
    total = 0.0
    for n in range(1, max_terms + 1):
        kn = n * math.pi / W
        decay = math.exp(-0.5 * sigma**2 * kn**2 * T)
        term = decay * math.sin(kn * (x - lo)) * float(np.dot(weight, np.sin(kn * (y - lo))))
        total += term
        if decay * float(np.sum(np.abs(weight))) < tol and n > 3:
            break
    return float(pref * total)


# =====================================================================================
# Binomial trees and perpetual options
# =====================================================================================


def _peizer_pratt(z: float, n: int) -> float:
    return 0.5 + math.copysign(0.5, z) * math.sqrt(
        1.0 - math.exp(-((z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2) * (n + 1.0 / 6.0))
    )


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "put",
    american: bool = True,
    n_steps: int = 1001,
    tree: str = "lr",
) -> float:
    """European or American option on a binomial tree.

    Parameters
    ----------
    tree : {'lr', 'crr'}
        Leisen–Reimer (1996) tree with Peizer–Pratt inversion (method 2),
        which converges at second order for European options and very
        smoothly for American ones, or the Cox–Ross–Rubinstein tree.
        Leisen–Reimer requires an odd number of steps; even ``n_steps`` is
        increased by one.
    """
    opt = check_option_type(option)
    for name, v in (("S", S), ("K", K), ("T", T), ("sigma", sigma)):
        check_positive(name, v)
    n = int(n_steps)
    if n < 1:
        raise ValueError("n_steps must be positive")
    dt = T / n
    growth = math.exp((r - q) * dt)
    if tree == "crr":
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        p = (growth - d) / (u - d)
    elif tree == "lr":
        if n % 2 == 0:
            n += 1
            dt = T / n
            growth = math.exp((r - q) * dt)
        d1, d2 = _d1d2(S, K, T, r, sigma, q)
        p = _peizer_pratt(float(d2), n)
        pbar = _peizer_pratt(float(d1), n)
        u = growth * pbar / p
        d = (growth - p * u) / (1.0 - p)
    else:
        raise ValueError("tree must be 'lr' or 'crr'")
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability outside (0, 1); increase n_steps")
    disc = math.exp(-r * dt)
    j = np.arange(n + 1)
    s_t = S * u ** (n - j) * d**j
    pay = (
        (lambda s: np.maximum(s - K, 0.0)) if opt == "call" else (lambda s: np.maximum(K - s, 0.0))
    )
    v = pay(s_t)
    for step in range(n - 1, -1, -1):
        v = disc * (p * v[:-1] + (1.0 - p) * v[1:])
        if american:
            jj = np.arange(step + 1)
            v = np.maximum(v, pay(S * u ** (step - jj) * d**jj))
    return float(v[0])


def perpetual_american_put(S: float, K: float, r: float, sigma: float, q: float = 0.0) -> float:
    r"""Perpetual American put (McKean, 1965; Merton, 1973).

    With :math:`\beta < 0` the negative root of
    :math:`\tfrac12\sigma^2\beta(\beta - 1) + (r - q)\beta - r = 0`, the
    exercise boundary is :math:`S^* = K\beta/(\beta - 1)` and
    :math:`V = (K - S^*)(S/S^*)^\beta` for :math:`S > S^*`.
    """
    for name, v in (("S", S), ("K", K), ("r", r), ("sigma", sigma)):
        check_positive(name, v)
    s_star = perpetual_put_boundary(K, r, sigma, q)
    if S <= s_star:
        return K - S
    beta = _perpetual_beta(r, sigma, q)
    return (K - s_star) * (S / s_star) ** beta


def _perpetual_beta(r: float, sigma: float, q: float) -> float:
    m = r - q - 0.5 * sigma**2
    return (-m - math.sqrt(m**2 + 2.0 * r * sigma**2)) / sigma**2


def perpetual_put_boundary(K: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Optimal exercise level of the perpetual American put."""
    beta = _perpetual_beta(r, sigma, q)
    return K * beta / (beta - 1.0)


# =====================================================================================
# Fourier pricing (Lewis 2001) and Merton's jump-diffusion series
# =====================================================================================

CharFn = Callable[[complex], complex]


def lewis_call_price(
    cf: CharFn,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option: str = "call",
) -> float:
    r"""European option price from the characteristic function (Lewis, 2001).

    ``cf(u)`` must be the characteristic function
    :math:`\varphi(u) = \mathbb{E}[e^{iuX_T}]` of the *martingale-normalised*
    log-return, :math:`S_T = S e^{(r-q)T + X_T}` with
    :math:`\mathbb{E}[e^{X_T}] = 1`, analytic in the strip
    :math:`-1 < \operatorname{Im} u < 0`. Then

    .. math::
        C = S e^{-qT} - \frac{\sqrt{SK}\, e^{-(r+q)T/2}}{\pi}
            \int_0^\infty \operatorname{Re}\!\left[e^{iuk}
            \varphi\!\left(u - \tfrac{i}{2}\right)\right] \frac{du}{u^2 + 1/4},

    with :math:`k = \log(S/K) + (r - q)T`. Puts follow from parity.
    """
    opt = check_option_type(option)
    k = math.log(S / K) + (r - q) * T

    def integrand(u: float) -> float:
        return float((np.exp(1j * u * k) * cf(u - 0.5j)).real) / (u * u + 0.25)

    val, _ = quad(integrand, 0.0, np.inf, limit=1000, epsabs=1e-13, epsrel=1e-12)
    call = S * math.exp(-q * T) - math.sqrt(S * K) * math.exp(-0.5 * (r + q) * T) * val / math.pi
    if opt == "call":
        return float(call)
    return float(call - S * math.exp(-q * T) + K * math.exp(-r * T))


def bs_cf(sigma: float, T: float) -> CharFn:
    """Characteristic function of the normalised Black–Scholes log-return."""

    def cf(u: complex) -> complex:
        return complex(np.exp(-0.5 * sigma**2 * T * (u * u + 1j * u)))

    return cf


def heston_cf(T: float, v0: float, kappa: float, theta: float, sigma: float, rho: float) -> CharFn:
    r"""Heston characteristic function of :math:`X_T = \log(S_T/S_0) - (r - q)T`.

    Uses the formulation of Albrecher et al. (2007) ("little Heston trap"),
    which is continuous in :math:`u` along the integration path.
    """

    def cf(u: complex) -> complex:
        iu = 1j * u
        d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma**2 * (iu + u * u))
        g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
        e = np.exp(-d * T)
        C = (kappa * theta / sigma**2) * (
            (kappa - rho * sigma * iu - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g))
        )
        D = ((kappa - rho * sigma * iu - d) / sigma**2) * (1.0 - e) / (1.0 - g * e)
        return complex(np.exp(C + D * v0))

    return cf


def levy_cf(process: Any, T: float) -> CharFn:
    r"""Martingale-normalised characteristic function of a Lévy process.

    Returns :math:`u \mapsto \varphi_L(u, T) / \varphi_L(-i, T)^{iu}`, so
    that :math:`X_T = L_T - T\log\mathbb{E}[e^{L_1}]` satisfies
    :math:`\mathbb{E}[e^{X_T}] = 1`. The drift of the process cancels out,
    as it must under the risk-neutral measure.
    """
    log_m = complex(np.log(complex(process.char_func(-1j, t=1.0))))
    if abs(log_m.imag) > 1e-10:
        raise ValueError("process has no finite exponential moment E[exp(L_1)]")
    omega = log_m.real

    def cf(u: complex) -> complex:
        return complex(process.char_func(u, t=T)) * complex(np.exp(-1j * u * omega * T))

    return cf


def merton_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    jump_mean: float,
    jump_std: float,
    q: float = 0.0,
    option: str = "call",
    tol: float = 1e-15,
) -> float:
    r"""Merton (1976) jump-diffusion price as a Poisson mixture of Black–Scholes prices.

    Log-jumps are :math:`N(\mu_J, \delta^2)`, so
    :math:`\kappa = e^{\mu_J + \delta^2/2} - 1` and

    .. math::
        V = \sum_{n \ge 0} e^{-\lambda' T} \frac{(\lambda' T)^n}{n!}
            V_{BS}\!\left(\sigma_n, r_n\right), \quad
        \lambda' = \lambda(1 + \kappa),\;
        \sigma_n^2 = \sigma^2 + \frac{n\delta^2}{T},\;
        r_n = r - \lambda\kappa + \frac{n \log(1 + \kappa)}{T}.
    """
    kap = math.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    lam_p = lam * (1.0 + kap)
    total = 0.0
    weight_sum = 0.0
    n = 0
    log_w = -lam_p * T
    while True:
        w = math.exp(log_w)
        sig_n = math.sqrt(sigma**2 + n * jump_std**2 / T)
        r_n = r - lam * kap + n * math.log1p(kap) / T
        # with weights at intensity lambda' the n-th term is a plain BS price at rate r_n
        px = float(bs_price(S, K, T, r_n, sig_n, q, option))
        total += w * px
        weight_sum += w
        n += 1
        log_w += math.log(lam_p * T) - math.log(n) if lam_p > 0 else -np.inf
        if (1.0 - weight_sum) < tol or n > 500:
            break
    return float(total)


# =====================================================================================
# Short-rate models
# =====================================================================================


def _vasicek_ab(tau: float, kappa: float, theta: float, sigma: float) -> tuple[float, float]:
    B = (1.0 - math.exp(-kappa * tau)) / kappa
    lnA = (theta - sigma**2 / (2 * kappa**2)) * (B - tau) - sigma**2 * B**2 / (4 * kappa)
    return lnA, B


def vasicek_bond(r: Any, tau: float, kappa: float, theta: float, sigma: float) -> Any:
    r"""Vasicek zero-coupon bond :math:`P = A(\tau) e^{-B(\tau) r}`.

    :math:`B = (1 - e^{-\kappa\tau})/\kappa`,
    :math:`\ln A = (\theta - \sigma^2/2\kappa^2)(B - \tau) - \sigma^2 B^2/4\kappa`.
    """
    if tau <= 0.0:
        return np.ones_like(np.asarray(r, dtype=float)) if np.ndim(r) else 1.0
    lnA, B = _vasicek_ab(tau, kappa, theta, sigma)
    return np.exp(lnA - B * np.asarray(r, dtype=float))


def vasicek_bond_option(
    r: float,
    K: float,
    T: float,
    S: float,
    kappa: float,
    theta: float,
    sigma: float,
    option: str = "call",
) -> float:
    r"""Jamshidian (1989) option expiring at ``T`` on the bond maturing at ``S``.

    .. math::
        \sigma_p = \frac{\sigma}{\kappa}\bigl(1 - e^{-\kappa(S - T)}\bigr)
                   \sqrt{\frac{1 - e^{-2\kappa T}}{2\kappa}}, \quad
        h = \frac{1}{\sigma_p}\ln\frac{P(0,S)}{K P(0,T)} + \frac{\sigma_p}{2},

    call :math:`= P(0,S) N(h) - K P(0,T) N(h - \sigma_p)`.
    """
    opt = check_option_type(option)
    pS = float(vasicek_bond(r, S, kappa, theta, sigma))
    pT = float(vasicek_bond(r, T, kappa, theta, sigma))
    sp = sigma / kappa * (1 - math.exp(-kappa * (S - T)))
    sp *= math.sqrt((1 - math.exp(-2 * kappa * T)) / (2 * kappa))
    h = math.log(pS / (K * pT)) / sp + 0.5 * sp
    if opt == "call":
        return pS * norm.cdf(h) - K * pT * norm.cdf(h - sp)
    return K * pT * norm.cdf(-h + sp) - pS * norm.cdf(-h)


def _cir_ab(tau: float, kappa: float, theta: float, sigma: float) -> tuple[float, float]:
    g = math.sqrt(kappa**2 + 2 * sigma**2)
    e = math.expm1(g * tau)
    den = 2 * g + (kappa + g) * e
    B = 2 * e / den
    lnA = (2 * kappa * theta / sigma**2) * (
        math.log(2 * g) + 0.5 * (kappa + g) * tau - math.log(den)
    )
    return lnA, B


def cir_bond(r: Any, tau: float, kappa: float, theta: float, sigma: float) -> Any:
    r"""CIR (1985) zero-coupon bond :math:`P = A(\tau) e^{-B(\tau) r}` with
    :math:`\gamma = \sqrt{\kappa^2 + 2\sigma^2}`,
    :math:`B = 2(e^{\gamma\tau} - 1)/[2\gamma + (\kappa + \gamma)(e^{\gamma\tau} - 1)]`,
    :math:`A = [2\gamma e^{(\kappa + \gamma)\tau/2}/(\cdots)]^{2\kappa\theta/\sigma^2}`."""
    if tau <= 0.0:
        return np.ones_like(np.asarray(r, dtype=float)) if np.ndim(r) else 1.0
    lnA, B = _cir_ab(tau, kappa, theta, sigma)
    return np.exp(lnA - B * np.asarray(r, dtype=float))


def cir_bond_option(
    r: float,
    K: float,
    T: float,
    S: float,
    kappa: float,
    theta: float,
    sigma: float,
    option: str = "call",
) -> float:
    r"""CIR (1985) option on a zero-coupon bond via non-central chi-square laws.

    With :math:`\rho = 2\gamma/[\sigma^2(e^{\gamma T} - 1)]`,
    :math:`\psi = (\kappa + \gamma)/\sigma^2`, :math:`d = 4\kappa\theta/\sigma^2`
    and :math:`r^* = \ln(A(S - T)/K)/B(S - T)`,

    .. math::
        C = P(0,S)\,\chi^2\!\Bigl(2r^*[\rho + \psi + B];\, d,
              \tfrac{2\rho^2 r e^{\gamma T}}{\rho + \psi + B}\Bigr)
          - K P(0,T)\,\chi^2\!\Bigl(2r^*[\rho + \psi];\, d,
              \tfrac{2\rho^2 r e^{\gamma T}}{\rho + \psi}\Bigr)

    (Brigo & Mercurio, eq. 3.26); puts follow from parity.
    """
    from scipy.stats import ncx2

    opt = check_option_type(option)
    g = math.sqrt(kappa**2 + 2 * sigma**2)
    rho = 2 * g / (sigma**2 * math.expm1(g * T))
    psi = (kappa + g) / sigma**2
    lnA, B = _cir_ab(S - T, kappa, theta, sigma)
    r_star = (lnA - math.log(K)) / B
    d = 4 * kappa * theta / sigma**2
    pS = float(cir_bond(r, S, kappa, theta, sigma))
    pT = float(cir_bond(r, T, kappa, theta, sigma))
    e = math.exp(g * T)
    c = pS * ncx2.cdf(2 * r_star * (rho + psi + B), d, 2 * rho**2 * r * e / (rho + psi + B))
    c -= K * pT * ncx2.cdf(2 * r_star * (rho + psi), d, 2 * rho**2 * r * e / (rho + psi))
    if opt == "call":
        return float(c)
    return float(c - pS + K * pT)


def margrabe_price(
    S1: float,
    S2: float,
    T: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    q1: float = 0.0,
    q2: float = 0.0,
) -> float:
    r"""Option to exchange asset 2 for asset 1, :math:`(S_1 - S_2)^+` (Margrabe, 1978).

    .. math::
        V = S_1 e^{-q_1 T} N(d_1) - S_2 e^{-q_2 T} N(d_2), \qquad
        d_{1,2} = \frac{\log(S_1/S_2) + (q_2 - q_1 \pm \tfrac12\hat\sigma^2)T}{\hat\sigma\sqrt T},

    with :math:`\hat\sigma^2 = \sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2`.
    The interest rate cancels.
    """
    s = math.sqrt(sigma1**2 + sigma2**2 - 2.0 * rho * sigma1 * sigma2)
    d1 = (math.log(S1 / S2) + (q2 - q1 + 0.5 * s * s) * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    return float(S1 * math.exp(-q1 * T) * norm.cdf(d1) - S2 * math.exp(-q2 * T) * norm.cdf(d2))


def spread_option_price(
    S1: float,
    S2: float,
    K: float,
    T: float,
    r: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    q1: float = 0.0,
    q2: float = 0.0,
    n_nodes: int = 200,
) -> float:
    r"""Spread call :math:`(S_1 - S_2 - K)^+` by conditioning on :math:`S_2`.

    Given :math:`Z` driving asset 2, :math:`S_1(T)` is lognormal with forward
    :math:`S_1 e^{(r - q_1)T - \rho^2\sigma_1^2 T/2 + \rho\sigma_1\sqrt T Z}`
    and volatility :math:`\sigma_1\sqrt{1-\rho^2}`, so the price is the
    Gauss–Hermite average of Black prices with strike :math:`K + S_2(T)`.
    Converges spectrally; ``n_nodes = 200`` gives ~1e-12 for ordinary inputs.
    """
    z, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / math.sqrt(2.0 * math.pi)
    sT = math.sqrt(T)
    s2T = S2 * np.exp((r - q2 - 0.5 * sigma2**2) * T + sigma2 * sT * z)
    F1 = S1 * np.exp((r - q1) * T - 0.5 * rho**2 * sigma1**2 * T + rho * sigma1 * sT * z)
    vol = sigma1 * math.sqrt(1.0 - rho**2) * sT
    strike = K + s2T
    out = np.empty_like(z)
    pos = strike > 0.0
    d1 = (np.log(F1[pos] / strike[pos]) + 0.5 * vol**2) / vol
    out[pos] = F1[pos] * norm.cdf(d1) - strike[pos] * norm.cdf(d1 - vol)
    out[~pos] = F1[~pos] - strike[~pos]
    return float(math.exp(-r * T) * np.dot(w, out))
