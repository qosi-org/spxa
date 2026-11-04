r"""Monte Carlo for the Jarrow–Yildirim model.

The two short rates are simulated as :math:`n_t = x^n_t + \varphi_n(t)` and
:math:`r_t = x^r_t + \varphi_r(t)`. Here :math:`x^n` and :math:`x^r` are
Ornstein–Uhlenbeck processes, stepped exactly, with :math:`x^r` carrying the
drift :math:`-\rho_{rI}\sigma_r\sigma_I` of the nominal measure. The
deterministic shifts :math:`\varphi_x(t) = f_x(0,t) + \frac{\sigma_x^2}{2a_x^2}(1 - e^{-a_x t})^2`
fit each curve. The integrals of :math:`n` and :math:`r` use the trapezoidal
rule, and the log-index takes the matching drift and an exact Gaussian shock.
All drivers are antithetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from spxa.rates.jarrow_yildirim import JarrowYildirim

__all__ = ["JYPaths", "jy_paths"]


@dataclass(frozen=True)
class JYPaths:
    """Recorded times, discount factors :math:`e^{-\\int_0^t n}`, index levels and short rates."""

    times: np.ndarray
    discount: np.ndarray
    index: np.ndarray
    nominal_rate: np.ndarray
    real_rate: np.ndarray

    def column(self, t: float) -> int:
        k = int(np.argmin(np.abs(self.times - t)))
        if abs(self.times[k] - t) > 1e-10:
            raise KeyError(f"time {t} was not recorded")
        return k


def _phi(curve: object, a: float, sigma: float, t: np.ndarray) -> np.ndarray:
    f = np.array([float(curve.instantaneous_forward(float(max(s, 1e-6)))) for s in t])  # type: ignore[attr-defined]
    return f + 0.5 * (sigma / a) ** 2 * (-np.expm1(-a * t)) ** 2


def jy_paths(
    model: JarrowYildirim,
    T: float,
    n_steps: int,
    n_paths: int,
    record: Sequence[float] = (),
    seed: Optional[int] = None,
) -> JYPaths:
    """Simulate on ``n_steps`` equal steps plus the ``record`` times; returns values at ``record`` and ``T``."""
    grid = np.unique(
        np.round(np.r_[np.linspace(0.0, T, n_steps + 1), [t for t in record if 0 < t < T]], 14)
    )
    keep = sorted({float(t) for t in record if 0 < t <= T} | {float(T)})
    an, ar, sn, sr, si = model.a_n, model.a_r, model.sigma_n, model.sigma_r, model.sigma_i
    phi_n, phi_r = _phi(model.nominal, an, sn, grid), _phi(model.real, ar, sr, grid)
    lam, vec = np.linalg.eigh(model.correlation)
    L = vec * np.sqrt(np.maximum(lam, 0.0))
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2
    xn, xr = np.zeros(n_paths), np.zeros(n_paths)
    int_n, logi = np.zeros(n_paths), np.full(n_paths, math.log(model.cpi0))
    out = {k: np.empty((len(keep), n_paths)) for k in ("d", "i", "n", "r")}
    col = 0
    drift_r = -model.rho_ri * sr * si
    for k in range(1, grid.size):
        dt = grid[k] - grid[k - 1]
        z = rng.standard_normal((3, half))
        z = L @ z
        z = np.concatenate([z, -z], axis=1)[:, :n_paths]
        en, er = math.exp(-an * dt), math.exp(-ar * dt)
        # exact OU steps; the index shock is correlated with the rate shocks at the step level
        xn_new = xn * en + sn * math.sqrt(-math.expm1(-2 * an * dt) / (2 * an)) * z[0]
        xr_new = (
            xr * er
            + drift_r * (-math.expm1(-ar * dt)) / ar
            + sr * math.sqrt(-math.expm1(-2 * ar * dt) / (2 * ar)) * z[1]
        )
        n0, n1 = xn + phi_n[k - 1], xn_new + phi_n[k]
        r0, r1 = xr + phi_r[k - 1], xr_new + phi_r[k]
        step_n, step_r = 0.5 * (n0 + n1) * dt, 0.5 * (r0 + r1) * dt
        int_n += step_n
        logi += step_n - step_r - 0.5 * si * si * dt + si * math.sqrt(dt) * z[2]
        xn, xr = xn_new, xr_new
        if col < len(keep) and abs(grid[k] - keep[col]) < 1e-12:
            out["d"][col], out["i"][col] = np.exp(-int_n), np.exp(logi)
            out["n"][col], out["r"][col] = n1, r1
            col += 1
    return JYPaths(np.array(keep), out["d"].T, out["i"].T, out["n"].T, out["r"].T)
