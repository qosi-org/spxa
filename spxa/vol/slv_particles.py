r"""
Particle calibration of the SLV leverage (Guyon & Henry-Labordère, 2012).

Instead of solving the Fokker–Planck equation, the conditional expectation
:math:`E[v_t \mid y_t = y]` in

.. math:: L^2(t, y) = \frac{\sigma_{LV}^2(t, y)}{E[v_t \mid y_t = y]}

is estimated from the simulated particles themselves with a Nadaraya–Watson
kernel regression

.. math:: \widehat{E}[v_t \mid y] = \frac{\sum_i v_i\,K_h(y_i - y)}{\sum_i K_h(y_i - y)},
          \qquad K_h(u) = e^{-u^2 / (2h^2)},

on a grid of ``y`` nodes, with bandwidth
:math:`h = c\,\sigma\sqrt{\max(t, \Delta t)}\,N^{-1/5}`. The leverage is
then interpolated to the particles and the system is advanced one step
with the scheme of :func:`spxa.vol.slv_paths`, so that calibration and
pricing share their discretisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from spxa.vol.heston import Heston
from spxa.vol.slv import LeverageSurface
from spxa.vol.slv_fp import _local_variance

__all__ = ["ParticleCalibration", "calibrate_leverage_particles", "kernel_conditional_mean"]


def kernel_conditional_mean(
    x: np.ndarray, v: np.ndarray, nodes: np.ndarray, h: float, chunk: int = 20000
) -> np.ndarray:
    """Gaussian-kernel regression of ``v`` on ``x`` at ``nodes`` (bandwidth ``h``)."""
    num = np.zeros(nodes.size)
    den = np.zeros(nodes.size)
    for s in range(0, x.size, chunk):
        w = np.exp(-0.5 * ((x[s : s + chunk, None] - nodes[None, :]) / h) ** 2)
        num += v[s : s + chunk] @ w
        den += w.sum(axis=0)
    out = np.full(nodes.size, np.nan)
    ok = den > 1e-8 * max(den.max(), 1e-300)
    out[ok] = num[ok] / den[ok]
    idx = np.flatnonzero(ok)
    bad = np.flatnonzero(~ok)
    if bad.size:
        out[bad] = out[idx[np.abs(bad[:, None] - idx[None, :]).argmin(axis=1)]]
    return out


@dataclass(frozen=True)
class ParticleCalibration:
    model: Heston
    leverage: LeverageSurface
    bandwidths: np.ndarray


def calibrate_leverage_particles(
    local_vol: Any,
    model: Heston,
    T: float,
    n_steps: int = 100,
    n_particles: int = 50_000,
    nodes: Optional[np.ndarray] = None,
    bandwidth: float = 1.5,
    seed: Optional[int] = None,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> ParticleCalibration:
    """Leverage on ``nodes`` (default 81 points over 5 standard deviations) at each step start."""
    lv2 = _local_variance(local_vol)
    ts = np.linspace(0.0, T, int(n_steps) + 1)
    sig = float(np.sqrt(np.max(lv2(np.zeros(1), 0.5 * T))))
    if nodes is None:
        nodes = np.linspace(-5 * sig * np.sqrt(T), 5 * sig * np.sqrt(T), 81)
    rng = np.random.default_rng(seed)
    m = model
    sr = np.sqrt(1 - m.rho**2)
    N = int(n_particles)
    y = np.zeros(N)
    v = np.full(N, m.v0)
    rows, hs = [], []
    for n in range(ts.size - 1):
        t0, t1 = float(ts[n]), float(ts[n + 1])
        dt = t1 - t0
        vp = np.maximum(v, 0.0)
        if n == 0:
            ev = np.full(nodes.size, m.v0)
            h = 0.0
        else:
            h = bandwidth * sig * np.sqrt(t0) * N ** (-0.2)
            ev = kernel_conditional_mean(y, vp, nodes, h)
        lev = np.clip(np.sqrt(lv2(nodes, 0.5 * (t0 + t1)) / np.maximum(ev, 1e-12)), *bounds)
        rows.append(lev)
        hs.append(h)
        z = rng.standard_normal((2, N))
        a = np.interp(y, nodes, lev) * np.sqrt(vp)
        y = y - 0.5 * a * a * dt + a * np.sqrt(dt) * (m.rho * z[1] + sr * z[0])
        v = v + m.kappa * (m.theta - vp) * dt + m.xi * np.sqrt(vp * dt) * z[1]
    return ParticleCalibration(m, LeverageSurface(ts[:-1], nodes, np.vstack(rows)), np.array(hs))
