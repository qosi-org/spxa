r"""
Monte Carlo for stochastic local volatility.

Paths of the SLV model of :mod:`spxa.vol.slv` in forward log-moneyness:

.. math::
    y_{n+1} = y_n - \tfrac12 a_n^2\Delta t + a_n\sqrt{\Delta t}\,
        (\rho Z^2_n + \sqrt{1 - \rho^2}\,Z^1_n), \qquad a_n = L(y_n, t_n)\sqrt{v_n^+},

    v_{n+1} = v_n + \kappa(\theta - v_n^+)\Delta t + \xi\sqrt{v_n^+\Delta t}\,Z^2_n,

full-truncation Euler for the variance (Lord, Koekkoek & van Dijk 2010) and
a log-Euler step for :math:`y`, which keeps :math:`e^{y}` an exact
martingale step by step.

Knock-out barriers given as levels in :math:`y` (functions of time, so
that a fixed spot barrier :math:`H` enters as :math:`\log(H/F_t)`) are
monitored continuously with the Brownian-bridge correction: a path
surviving both ends of a step survives it with probability
:math:`1 - \exp(-2 d_0 d_1 / (a_n^2\Delta t))`, :math:`d_0, d_1` its
distances to the barrier, which is exact for a constant :math:`a_n` and
a barrier linear in time over the step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from spxa.vol.heston import Heston

__all__ = ["SLVPaths", "slv_paths"]

Level = Union[float, Callable[[float], float]]


@dataclass(frozen=True)
class SLVPaths:
    """Recorded ``y`` and ``v`` (rows are ``times``) and the knock-out survival weights."""

    times: np.ndarray
    y: np.ndarray
    v: np.ndarray
    survival: np.ndarray

    def _row(self, t: float) -> int:
        j = int(np.argmin(np.abs(self.times - t)))
        if abs(self.times[j] - t) > 1e-12:
            raise ValueError(f"{t} is not a recorded time")
        return j

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        j = self._row(t)
        return self.y[j], self.v[j]

    def price(
        self, payoff: Callable[[np.ndarray], np.ndarray], t: float, knock_out: bool = False
    ) -> tuple[float, float]:
        """Mean of ``payoff(e^{y_t})`` (times the survival weight if ``knock_out``) and its standard error."""
        y, _ = self.at(t)
        val = payoff(np.exp(y)) * (self.survival if knock_out else 1.0)
        return float(val.mean()), float(val.std(ddof=1) / np.sqrt(val.size))


def _level(b: Optional[Level]) -> Optional[Callable[[float], float]]:
    if b is None:
        return None
    if callable(b):
        return b
    c = float(b)
    return lambda t: c


def slv_paths(
    model: Heston,
    leverage: Callable[[np.ndarray, float], Any],
    T: float,
    n_steps: int,
    n_paths: int,
    record: Sequence[float] = (),
    seed: Optional[int] = None,
    antithetic: bool = True,
    lower: Optional[Level] = None,
    upper: Optional[Level] = None,
    bridge: bool = True,
) -> SLVPaths:
    """Simulate the SLV model; ``record`` times are added to the uniform step grid.

    ``lower`` / ``upper`` are knock-out levels in ``y``; ``survival`` is the
    probability-weighted indicator of not touching them before ``T``
    (discrete monitoring at the steps when ``bridge`` is false).
    """
    if T <= 0 or n_steps < 1 or n_paths < 2:
        raise ValueError("need T > 0, n_steps >= 1 and n_paths >= 2")
    ts = np.unique(
        np.round(np.r_[np.linspace(0.0, T, n_steps + 1), [t for t in record if 0 < t < T]], 14)
    )
    keep = sorted({float(t) for t in record if 0 < t <= T} | {float(ts[-1])})
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2 if antithetic else n_paths
    lo, hi = _level(lower), _level(upper)
    m = model
    sr = np.sqrt(1.0 - m.rho**2)
    y = np.zeros(n_paths)
    v = np.full(n_paths, m.v0)
    alive = np.ones(n_paths)
    rows_y, rows_v = [], []
    for n in range(ts.size - 1):
        t0, t1 = float(ts[n]), float(ts[n + 1])
        dt = t1 - t0
        z = rng.standard_normal((2, half))
        if antithetic:
            z = np.concatenate([z, -z], axis=1)[:, :n_paths]
        vp = np.maximum(v, 0.0)
        a = np.asarray(leverage(y, t0), float) * np.sqrt(vp)
        y_new = y - 0.5 * a * a * dt + a * np.sqrt(dt) * (m.rho * z[1] + sr * z[0])
        v = v + m.kappa * (m.theta - vp) * dt + m.xi * np.sqrt(vp * dt) * z[1]
        for level, sign in ((lo, 1.0), (hi, -1.0)):
            if level is None:
                continue
            d0 = sign * (y - level(t0))
            d1 = sign * (y_new - level(t1))
            out = (d0 <= 0) | (d1 <= 0)
            alive[out] = 0.0
            if bridge:
                with np.errstate(divide="ignore", over="ignore"):
                    p = np.exp(
                        -2.0
                        * np.maximum(d0, 0)
                        * np.maximum(d1, 0)
                        / np.maximum(a * a * dt, 1e-300)
                    )
                alive *= 1.0 - np.where(out, 0.0, p)
        y = y_new
        if any(abs(t1 - k) < 1e-12 for k in keep):
            rows_y.append(y.copy())
            rows_v.append(v.copy())
    return SLVPaths(np.array(keep), np.array(rows_y), np.array(rows_v), alive)
