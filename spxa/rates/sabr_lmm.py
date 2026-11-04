r"""
SABR-LMM: a LIBOR market model with SABR smiles.

Each forward follows a shifted CEV diffusion scaled by a stochastic
volatility factor shared by the whole curve (in the spirit of Rebonato,
McKay & White, *The SABR/LIBOR Market Model*, 2009),

.. math::
    dL_i = \alpha_i V_t (L_i + \delta)^\beta\,dW_i, \qquad
    dV_t = \nu V_t\,dZ_t, \quad V_0 = 1,

with :math:`d\langle W_i, W_j\rangle = \rho_{ij}dt` and
:math:`d\langle W_i, Z\rangle = \rho\,dt`. Under its own forward measure
each :math:`L_i` is then a SABR process with parameters
:math:`(\alpha_i, \beta, \rho, \nu)` up to the drift that the measure change
puts on :math:`V`, which is of order :math:`\nu\rho\alpha\tau L^\beta` and
vanishes for the first forward; caplets are therefore close to Hagan's
formula, and swaptions inherit a smile from the common factor.

The simulation runs under the spot LIBOR measure, where :math:`V` is
driftless (the numeraire's volatility over the current period is not
modelled) and the forward :math:`i` has drift
:math:`\alpha_i V^2 (L_i+\delta)^\beta\sum_{j=q(t)}^{i}\tau_j\rho_{ij}\alpha_j(L_j+\delta)^\beta/(1 + \tau_j L_j)`.
Forwards are stepped by Euler with absorption at :math:`-\delta` and
:math:`V` exactly.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from spxa.market.black import black_price
from spxa.market.curves import YieldCurve
from spxa.rates.lmm_mc import LMMPaths
from spxa.rates.sabr import SABR, sabr_lognormal_vol

__all__ = ["SABRLMM", "sabr_lmm_paths"]


class SABRLMM:
    """Forwards on ``times`` with SABR parameters ``alphas[i]``, ``beta``, ``rho``, ``nu``."""

    def __init__(
        self,
        curve: YieldCurve,
        times: Sequence[float],
        alphas: Sequence[float],
        beta: float,
        rho: float,
        nu: float,
        correlation: Optional[np.ndarray] = None,
        shift: float = 0.0,
    ) -> None:
        T = np.asarray(times, float)
        if T.ndim != 1 or T.size < 2 or np.any(np.diff(T) <= 0) or T[0] < 0:
            raise ValueError("times must be increasing, non-negative and at least two")
        self.times, self.tau, self.n = T, np.diff(T), T.size - 1
        self.alphas = np.asarray(alphas, float)
        if self.alphas.shape != (self.n,):
            raise ValueError("need one alpha per forward")
        self.beta, self.rho, self.nu, self.shift = float(beta), float(rho), float(nu), float(shift)
        self.correlation = (
            np.ones((self.n, self.n)) if correlation is None else np.asarray(correlation, float)
        )
        full = np.ones((self.n + 1, self.n + 1))
        full[: self.n, : self.n] = self.correlation
        full[: self.n, self.n] = full[self.n, : self.n] = self.rho
        w, V = np.linalg.eigh(full)
        if w.min() < -1e-10:
            raise ValueError(
                "the joint correlation of forwards and volatility is not positive semi-definite"
            )
        self.factor = V * np.sqrt(np.maximum(w, 0.0))
        P = np.array([float(curve.discount(t)) for t in T])
        self.discounts = P
        self.forwards = (P[:-1] / P[1:] - 1.0) / self.tau

    def sabr(self, i: int) -> SABR:
        """The SABR model of forward ``i`` (ignoring the measure-change drift of the volatility)."""
        return SABR(self.alphas[i], self.beta, self.rho, self.nu, self.shift)

    def caplet(self, i: int, K: Any, notional: float = 1.0, option: str = "call") -> Any:
        """Caplet by Hagan's formula on :meth:`sabr`."""
        F, T, d = self.forwards[i], self.times[i], self.shift
        vol = sabr_lognormal_vol(F, K, T, self.alphas[i], self.beta, self.rho, self.nu, d)
        df = self.discounts[i + 1] * self.tau[i] * notional
        return black_price(F + d, np.asarray(K, float) + d, T, vol, df, option)


def sabr_lmm_paths(
    model: SABRLMM,
    n_paths: int,
    steps_per_period: int = 8,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> LMMPaths:
    """Simulate the forwards to every fixing date (an :class:`LMMPaths`)."""
    m = model
    n, tau, d, b = m.n, m.tau, m.shift, m.beta
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2 if antithetic else n_paths
    S = np.tile(m.forwards + d, (n_paths, 1))  # shifted forwards
    V = np.ones(n_paths)
    out = np.empty((n, n_paths, n))
    grid = [0.0]
    for k in range(n):
        start, end = (0.0 if k == 0 else m.times[k - 1]), m.times[k]
        if end > start:
            n_sub = max(1, math.ceil(steps_per_period * (end - start) / float(np.mean(tau))))
            grid += list(np.linspace(start, end, n_sub + 1)[1:])
    k = 0
    for s in range(len(grid) - 1):
        t0, t1 = grid[s], grid[s + 1]
        dt = t1 - t0
        q = int(np.searchsorted(m.times[:-1], t0 + 1e-14, side="left"))
        z = rng.standard_normal((half, n + 1))
        if antithetic:
            z = np.concatenate([z, -z])[:n_paths]
        dB = (z @ m.factor.T) * math.sqrt(dt)  # columns: W_1..W_n, Z
        alive = np.arange(q, n)
        Sa = np.maximum(S[:, alive], 0.0)
        vol = m.alphas[alive] * V[:, None] * Sa**b  # alpha_i V (L_i + d)^beta
        g = tau[alive] * vol / (1.0 + tau[alive] * (Sa - d))
        rho = m.correlation[np.ix_(alive, alive)]
        drift = vol * ((g[:, None, :] * np.tril(rho)[None, :, :]).sum(axis=2))
        Sn = Sa + drift * dt + vol * dB[:, alive]
        S[:, alive] = np.where((Sa <= 0.0) | (Sn <= 0.0), 0.0, Sn)
        V = V * np.exp(m.nu * dB[:, n] - 0.5 * m.nu**2 * dt)
        while k < n and abs(t1 - m.times[k]) < 1e-12:
            out[k] = S - d
            k += 1
    return LMMPaths(m, out)  # type: ignore[arg-type]  # same n, tau, discounts
