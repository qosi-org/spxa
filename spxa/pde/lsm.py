r"""
Least-squares Monte Carlo (Longstaff & Schwartz, 2001) for early exercise.

An independent, simulation-based check of the PDE pricers for American and
Bermudan contracts, and a pricer for models whose PDE is not available.

Given simulated state paths :math:`X^{(m)}_{t_k}` at exercise dates
:math:`t_1 < \dots < t_N = T`, the cash flows are initialised with the
payoff at :math:`T` and moved backwards. At each date the continuation value
of the in-the-money paths is estimated by regressing the discounted future
cash flow on basis functions of the state; a path exercises when the
immediate payoff exceeds the fitted continuation value.

Two passes (``regression_paths``) remove the upward bias of re-using the
same paths for regression and valuation: the exercise rule is fitted on one
set of paths and applied to an independent set, so the estimate is a
genuine lower bound in expectation (the policy is sub-optimal), with a
standard error from the independent valuation paths.

Basis: weighted Laguerre polynomials
:math:`e^{-x/2}L_k(x)` of the normalised state :math:`x = S/K` as in the
original paper, or plain monomials. Regression uses a QR-based least
squares solve, which is stable for degrees up to about 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from spxa.pde._utils import check_option_type, check_positive

__all__ = ["LSMResult", "longstaff_schwartz", "gbm_paths", "lsm_american"]

PayoffFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class LSMResult:
    """Least-squares Monte Carlo estimate.

    Attributes
    ----------
    price : float
        Mean discounted cash flow under the fitted exercise policy
        (including exercise at ``t = 0`` when it is optimal).
    std_error : float
        Standard error of ``price`` (valuation paths).
    n_paths : int
        Number of valuation paths.
    exercise_probability : float
        Fraction of valuation paths exercised before maturity.
    boundary : ndarray
        For one-dimensional monotone payoffs, the largest (put) / smallest
        (call) exercised state at each date (NaN when nobody exercises).
    info : dict
        Basis, degree, number of regression paths and dates.
    """

    price: float
    std_error: float
    n_paths: int
    exercise_probability: float
    boundary: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.price)

    def __story__(self) -> str:
        """Plain-language summary of the estimate."""
        lo, hi = self.confidence_interval()
        passes = (
            "two-pass (independent regression paths)" if self.info.get("two_pass") else "one-pass"
        )
        return "\n".join(
            [
                f"Least-squares Monte Carlo, {passes}",
                f"  price = {self.price:.6f} +/- {self.std_error:.6f} (95% CI [{lo:.6f}, {hi:.6f}])",
                f"  {self.n_paths} valuation paths, {self.info.get('n_dates')} exercise dates, "
                f"{self.info.get('basis')} basis of degree {self.info.get('degree')}",
                f"  early exercise on {100 * self.exercise_probability:.1f}% of paths",
            ]
        )

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Normal confidence interval ``price -/+ z * std_error``."""
        return (self.price - z * self.std_error, self.price + z * self.std_error)


def _basis(x: np.ndarray, degree: int, kind: str) -> np.ndarray:
    """Constant plus ``degree`` basis functions (weighted Laguerre or monomials)."""
    cols = [np.ones_like(x)]
    if kind == "laguerre":
        w = np.exp(-0.5 * x)
        cols.append(w)  # e^{-x/2} L_0
        lk_prev, lk = np.ones_like(x), 1.0 - x
        for k in range(1, degree):
            cols.append(w * lk)
            lk_prev, lk = lk, ((2 * k + 1 - x) * lk - k * lk_prev) / (k + 1)
    else:
        for k in range(1, degree + 1):
            cols.append(x**k)
    return np.column_stack(cols)


def _fit_policy(
    paths: np.ndarray,
    disc: np.ndarray,
    payoff: PayoffFn,
    degree: int,
    basis: str,
    scale: float,
    itm_only: bool,
) -> list[Optional[np.ndarray]]:
    """Backward induction on the regression paths; returns coefficients per date."""
    n_dates = paths.shape[1] - 1
    cash = payoff(paths[:, -1]).astype(float)
    coefs: list[Optional[np.ndarray]] = [None] * (n_dates + 1)
    for k in range(n_dates - 1, 0, -1):
        cash = cash * disc[k]
        ex = payoff(paths[:, k])
        sel = ex > 0.0 if itm_only else np.ones_like(ex, dtype=bool)
        if np.count_nonzero(sel) <= degree + 1:
            continue
        A = _basis(paths[sel, k] / scale, degree, basis)
        beta, *_ = np.linalg.lstsq(A, cash[sel], rcond=None)
        coefs[k] = beta
        cont = A @ beta
        exercise = ex[sel] > cont
        idx = np.flatnonzero(sel)[exercise]
        cash[idx] = ex[sel][exercise]
    return coefs


def longstaff_schwartz(
    paths: np.ndarray,
    times: Sequence[float],
    payoff: PayoffFn,
    r: float,
    degree: int = 3,
    basis: str = "laguerre",
    regression_paths: Optional[np.ndarray] = None,
    scale: float = 1.0,
    itm_only: bool = True,
    side: Optional[str] = None,
) -> LSMResult:
    r"""Price an early-exercise contract from simulated paths.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_dates + 1)
        Valuation paths of the (one-dimensional) state at ``times``; column
        0 is the common initial state.
    times : sequence of float
        ``0 = t_0 < t_1 < ... < t_N = T``; exercise is allowed at
        :math:`t_1, \dots, t_N` and, if optimal, immediately at :math:`t_0`.
    payoff : callable
        Vectorised exercise value :math:`g(x)`.
    r : float
        Constant short rate.
    degree : int
        Number of non-constant basis functions.
    basis : {'laguerre', 'monomial'}
    regression_paths : ndarray, optional
        Independent paths for fitting the exercise rule (two-pass,
        low-biased estimator). When omitted the valuation paths are used
        (one-pass, as in Longstaff & Schwartz).
    scale : float
        The basis is evaluated at ``x / scale`` (use the strike).
    itm_only : bool
        Regress on in-the-money paths only (recommended).
    side : {'put', 'call'}, optional
        Orientation used to report the exercise boundary.
    """
    paths = np.asarray(paths, dtype=float)
    t = np.asarray(times, dtype=float)
    if paths.ndim != 2 or paths.shape[1] != t.size:
        raise ValueError("paths must have shape (n_paths, len(times))")
    if t.size < 2 or t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must start at 0 and increase strictly")
    if basis not in ("laguerre", "monomial"):
        raise ValueError("basis must be 'laguerre' or 'monomial'")
    if degree < 1:
        raise ValueError("degree must be at least 1")
    disc = np.ones(t.size)
    disc[:-1] = np.exp(-r * np.diff(t))  # disc[k]: from t_{k+1} back to t_k
    fit = paths if regression_paths is None else np.asarray(regression_paths, dtype=float)
    if fit.shape[1] != t.size:
        raise ValueError("regression_paths must have the same dates as paths")
    coefs = _fit_policy(fit, disc, payoff, degree, basis, scale, itm_only)

    n_paths, n_cols = paths.shape
    n_dates = n_cols - 1
    value = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    boundary = np.full(n_dates + 1, np.nan)
    discount_to_0 = np.exp(-r * t)
    n_early = 0
    for k in range(1, n_dates + 1):
        ex = payoff(paths[:, k])
        if k == n_dates:
            stop = alive & (ex > 0.0)
        else:
            beta = coefs[k]
            if beta is None:
                continue
            cand = alive & ((ex > 0.0) if itm_only else True)
            if not np.any(cand):
                continue
            cont = _basis(paths[cand, k] / scale, degree, basis) @ beta
            stop = np.zeros(n_paths, dtype=bool)
            stop[np.flatnonzero(cand)[ex[cand] > cont]] = True
            if side is not None and np.any(stop):
                s = paths[stop, k]
                boundary[k] = float(np.max(s) if side == "put" else np.min(s))
        value[stop] = ex[stop] * discount_to_0[k]
        alive &= ~stop
        if k < n_dates:
            n_early += int(np.count_nonzero(stop))

    mean = float(np.mean(value))
    se = float(np.std(value, ddof=1) / math.sqrt(n_paths))
    g0 = float(payoff(paths[:1, 0])[0])
    price = max(mean, g0)
    return LSMResult(
        price=price,
        std_error=se if mean >= g0 else 0.0,
        n_paths=n_paths,
        exercise_probability=n_early / n_paths,
        boundary=boundary,
        info={
            "basis": basis,
            "degree": degree,
            "n_dates": n_dates,
            "two_pass": regression_paths is not None,
            "n_regression_paths": int(fit.shape[0]),
            "immediate_exercise": mean < g0,
        },
    )


def gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact geometric Brownian motion paths on a uniform grid.

    Returns ``(times, paths)`` with ``paths.shape == (n_paths, n_steps + 1)``.
    With ``antithetic=True`` the second half of the paths mirrors the first
    (``n_paths`` must then be even).
    """
    S0 = check_positive("S0", S0)
    sigma = check_positive("sigma", sigma)
    T = check_positive("T", T)
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even with antithetic sampling")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((half, n_steps))
    if antithetic:
        z = np.vstack([z, -z])
    inc = (r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    logs = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)
    return np.linspace(0.0, T, n_steps + 1), S0 * np.exp(logs)


def lsm_american(
    K: float,
    T: float,
    spot: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "put",
    n_dates: int = 50,
    n_paths: int = 100_000,
    degree: int = 3,
    basis: str = "laguerre",
    seed: Optional[int] = 0,
    two_pass: bool = True,
) -> LSMResult:
    """Bermudan approximation of an American vanilla under Black–Scholes.

    ``n_dates`` equally spaced exercise dates; antithetic GBM paths; the
    regression set (two-pass) uses an independent seed.
    """
    opt = check_option_type(option)
    K = check_positive("K", K)
    sign = 1.0 if opt == "call" else -1.0

    def payoff(s: np.ndarray) -> np.ndarray:
        return np.maximum(sign * (s - K), 0.0)

    times, paths = gbm_paths(spot, r, sigma, T, n_dates, n_paths, q=q, seed=seed)
    reg = None
    if two_pass:
        reg_seed = None if seed is None else seed + 7919
        reg = gbm_paths(spot, r, sigma, T, n_dates, n_paths, q=q, seed=reg_seed)[1]
    return longstaff_schwartz(
        paths, times, payoff, r, degree, basis, regression_paths=reg, scale=K, side=opt
    )
