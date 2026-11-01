r"""
Dupire's forward equation and local-volatility extraction.

Under :math:`dS = (r - q) S\,dt + \sigma(S, t) S\,dW` the call prices
:math:`C(K, T)` seen from today, as functions of strike and maturity,
satisfy the forward equation (Dupire 1994)

.. math::
    C_T = \tfrac12 \sigma^2(K, T) K^2 C_{KK} - (r - q) K C_K - q C,
    \qquad C(K, 0) = (S_0 - K)^+,

so a single solve produces the whole call surface. At :math:`K = 0` the
equation degenerates to :math:`C_T = -qC` (giving :math:`S_0 e^{-qT}`) and is
used as the boundary condition; far out of the money :math:`C = 0`.

Conversely, Dupire's formula recovers the local volatility from a call
surface:

.. math::
    \sigma^2(K, T) = \frac{C_T + (r - q) K C_K + q C}{\tfrac12 K^2 C_{KK}}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from spxa.pde._utils import check_positive
from spxa.pde.boundary import Dirichlet, Natural
from spxa.pde.grid import Grid1D, TimeGrid
from spxa.pde.operators import DiscreteOperator1D, LinearPDE1D, derivative_matrices
from spxa.pde.payoffs import Payoff, project_payoff
from spxa.pde.reference import bs_price
from spxa.pde.theta import ThetaSolver1D

__all__ = ["DupirePDE", "DupireSurface", "local_vol_from_prices", "implied_volatility"]

VolSpec = Union[float, Callable[[np.ndarray, float], np.ndarray]]


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option: str = "call",
    tol: float = 1e-12,
) -> float:
    """Black–Scholes implied volatility by safeguarded Newton iteration (``nan`` if none)."""
    disc_s, disc_k = S * math.exp(-q * T), K * math.exp(-r * T)
    lower = max(disc_s - disc_k, 0.0) if option == "call" else max(disc_k - disc_s, 0.0)
    upper = disc_s if option == "call" else disc_k
    if not lower - 1e-14 <= price <= upper + 1e-14 or price - lower < 1e-14:
        return float("nan")
    lo, hi, vol = 1e-6, 10.0, 0.3
    for _ in range(100):
        p = float(bs_price(S, K, T, r, vol, q, option))
        diff = p - price
        if abs(diff) < tol:
            return vol
        if diff > 0:
            hi = vol
        else:
            lo = vol
        d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
        vega = disc_s * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(T)
        step = vol - diff / vega if vega > 1e-14 else 0.5 * (lo + hi)
        vol = step if lo < step < hi else 0.5 * (lo + hi)
    return vol


@dataclass
class DupireSurface:
    """Call prices :math:`C(K, T)` on a strike grid at the stored maturities."""

    strikes: np.ndarray
    maturities: np.ndarray
    prices: np.ndarray  # shape (n_maturities, n_strikes)
    spot: float
    r: float
    q: float
    info: dict[str, Any] = field(default_factory=dict)

    def _row(self, T: float) -> np.ndarray:
        j = int(np.argmin(np.abs(self.maturities - T)))
        if not math.isclose(self.maturities[j], T, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"maturity {T} was not stored; available: {self.maturities}")
        return self.prices[j]

    def call(self, K: Any, T: float) -> Any:
        """Call price(s) at strike(s) ``K`` for a stored maturity ``T``."""
        grid = Grid1D(self.strikes)
        k = np.atleast_1d(np.asarray(K, dtype=float))
        out = grid.interpolate(self._row(T), k)
        return float(out[0]) if np.ndim(K) == 0 else out

    def put(self, K: Any, T: float) -> Any:
        """Put prices by put–call parity."""
        c = self.call(K, T)
        return c - self.spot * math.exp(-self.q * T) + np.asarray(K) * math.exp(-self.r * T)

    def implied_vol(self, K: Any, T: float) -> Any:
        """Black–Scholes implied volatilities of the surface."""
        ks = np.atleast_1d(np.asarray(K, dtype=float))
        c = np.atleast_1d(self.call(ks, T))
        out = np.array(
            [implied_volatility(ci, self.spot, ki, T, self.r, self.q) for ci, ki in zip(c, ks)]
        )
        return float(out[0]) if np.ndim(K) == 0 else out

    def local_vol(self) -> np.ndarray:
        """Dupire local volatility recovered from the stored surface."""
        return local_vol_from_prices(self.strikes, self.maturities, self.prices, self.r, self.q)


def local_vol_from_prices(
    strikes: np.ndarray,
    maturities: np.ndarray,
    prices: np.ndarray,
    r: float,
    q: float = 0.0,
) -> np.ndarray:
    r"""Dupire's formula on a (maturity × strike) call-price grid.

    Derivatives use second-order three-point stencils on the (possibly
    non-uniform) grids; edge rows and columns are ``nan``. Points where the
    numerator or the convexity :math:`C_{KK}` is not positive (no
    arbitrage-free local volatility) are also ``nan``.
    """
    K = np.asarray(strikes, dtype=float)
    T = np.asarray(maturities, dtype=float)
    C = np.asarray(prices, dtype=float)
    if C.shape != (T.size, K.size):
        raise ValueError("prices must have shape (len(maturities), len(strikes))")
    if T.size < 3 or K.size < 3:
        raise ValueError("need at least three maturities and three strikes")
    D1, D2 = derivative_matrices(K)
    CK = (D1 @ C.T).T
    CKK = (D2 @ C.T).T
    Dt, _ = derivative_matrices(T)
    CT = Dt @ C
    num = CT + (r - q) * K[None, :] * CK + q * C
    den = 0.5 * K[None, :] ** 2 * CKK
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where((den > 0) & (num > 0), num / den, np.nan)
    out = np.sqrt(var)
    out[[0, -1], :] = np.nan
    out[:, [0, -1]] = np.nan
    return out


class DupirePDE:
    r"""Forward (Dupire) equation for the call surface of a local-volatility model.

    Parameters
    ----------
    r, q : float
        Rate and dividend yield.
    sigma : float or callable
        Local volatility :math:`\sigma(K, T)` (vectorised in ``K``).
    """

    def __init__(self, r: float, sigma: VolSpec, q: float = 0.0) -> None:
        self.r = float(r)
        self.q = float(q)
        if not callable(sigma):
            check_positive("sigma", sigma)
        self.sigma = sigma

    def __repr__(self) -> str:
        s = "callable" if callable(self.sigma) else f"{float(self.sigma):g}"
        return f"DupirePDE(r={self.r:g}, sigma={s}, q={self.q:g})"

    def _sig(self, k: np.ndarray, t: float) -> np.ndarray:
        if callable(self.sigma):
            return np.asarray(self.sigma(k, t), dtype=float) * np.ones_like(k)
        return np.full_like(k, float(self.sigma))

    def pde(self) -> LinearPDE1D:
        """Forward operator in ``(K, T)``."""
        mu = self.r - self.q

        def diffusion(k: np.ndarray, t: float) -> np.ndarray:
            return 0.5 * self._sig(k, t) ** 2 * k**2

        return LinearPDE1D(
            diffusion=diffusion,
            convection=lambda k, t: -mu * k,
            reaction=-self.q,
            time_homogeneous=not callable(self.sigma),
        )

    def solve(
        self,
        spot: float,
        T: float,
        maturities: Optional[Sequence[float]] = None,
        n_space: int = 401,
        n_time: int = 200,
        k_max: Optional[float] = None,
        concentration: float = 1.0,
        rannacher_steps: int = 2,
    ) -> DupireSurface:
        """Solve to ``T`` and store the surface at ``maturities`` (default: ``T`` only)."""
        spot = check_positive("spot", spot)
        T = check_positive("T", T)
        mats = sorted({float(m) for m in (maturities or [])} | {T})
        if mats[0] <= 0.0 or mats[-1] > T + 1e-14:
            raise ValueError("maturities must lie in (0, T]")
        if k_max is None:
            sig0 = float(self._sig(np.array([spot]), 0.0)[0])
            k_max = spot * math.exp(6.0 * sig0 * math.sqrt(T) + abs(self.r - self.q) * T)
        grid = Grid1D.sinh(0.0, k_max, n_space, center=spot, alpha=concentration * spot)
        op = DiscreteOperator1D(self.pde(), grid, Natural(), Dirichlet(0.0), "auto")
        payoff = Payoff(lambda k: np.maximum(spot - k, 0.0), (spot,), "(S0-K)+")
        # averaging only the kink cell keeps the linear wing exact, so C stays
        # convex down to K = 0 where the degenerate diffusion cannot smooth it
        u0 = project_payoff(payoff, grid, method="kink")
        tg = TimeGrid.uniform(T, n_time, [m for m in mats if m < T])
        stored: dict[float, np.ndarray] = {}

        def grab(k: int, t: float, u: np.ndarray) -> None:
            for m in mats:
                if abs(t - m) < 1e-12 * max(1.0, T):
                    stored[m] = u.copy()

        res = ThetaSolver1D(op, theta=0.5, rannacher_steps=rannacher_steps).solve(
            u0, tg, callback=grab
        )
        stored[T] = res.values.copy()
        return DupireSurface(
            strikes=grid.nodes.copy(),
            maturities=np.array(mats),
            prices=np.vstack([stored[m] for m in mats]),
            spot=spot,
            r=self.r,
            q=self.q,
            info={"time_steps": tg.n_steps if hasattr(tg, "n_steps") else len(tg.times) - 1},
        )
