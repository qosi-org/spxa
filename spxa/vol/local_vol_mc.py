r"""Monte Carlo under local volatility.

Paths of :math:`dS_t = (r - q)S_t\,dt + \sigma(S_t, t)S_t\,dW_t` are simulated
in log space,

.. math:: X_{n+1} = X_n + \bigl(r - q - \tfrac12\sigma_n^2\bigr)\Delta t
          + \sigma_n\sqrt{\Delta t}\,Z_n, \qquad \sigma_n = \sigma(e^{X_n}, t_n).

With constant :math:`\sigma` this scheme is exact. Otherwise it converges
weakly at first order. Because the drift correction keeps the
discounted spot an exact martingale step by step, forwards are matched
up to Monte Carlo noise for any local volatility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

__all__ = ["LocalVolPaths", "local_vol_paths"]


@dataclass(frozen=True)
class LocalVolPaths:
    """Simulated spot paths on ``times`` (``paths[i, j]`` is path ``i`` at ``times[j]``)."""

    times: np.ndarray
    paths: np.ndarray
    r: float
    q: float

    def at(self, t: float) -> np.ndarray:
        j = int(np.argmin(np.abs(self.times - t)))
        if abs(self.times[j] - t) > 1e-12:
            raise ValueError(f"{t} is not a simulation time")
        return self.paths[:, j]

    def price(self, payoff: Callable[[np.ndarray], np.ndarray], T: float) -> tuple[float, float]:
        """Discounted mean of ``payoff(S_T)`` and its standard error."""
        v = np.exp(-self.r * T) * payoff(self.at(T))
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def local_vol_paths(
    sigma: Callable[[Any, float], Any],
    spot: float,
    r: float,
    q: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 0,
    antithetic: bool = True,
    record: Optional[Sequence[float]] = None,
) -> LocalVolPaths:
    """Log-Euler paths with ``sigma(S, t)`` (vectorised in ``S``) on a uniform grid.

    Only the times in ``record`` (default: ``T`` alone) are stored; each must
    lie on the step grid ``k T / n_steps``. The start time 0 is always stored.
    """
    if n_steps < 1 or n_paths < 2 or T <= 0:
        raise ValueError("need n_steps >= 1, n_paths >= 2 and T > 0")
    dt = T / n_steps
    obs = [T] if record is None else sorted(float(t) for t in record)
    steps: dict[int, int] = {}
    for t in obs:
        k = round(t / dt)
        if k < 1 or k > n_steps or abs(k * dt - t) > 1e-9 * max(1.0, T):
            raise ValueError(f"record time {t} is not on the step grid")
        steps[k] = len(steps) + 1
    rng = np.random.default_rng(seed)
    m = (n_paths + 1) // 2 if antithetic else n_paths
    x = np.full(2 * m if antithetic else m, np.log(spot))
    out = np.empty((x.size, len(steps) + 1))
    out[:, 0] = spot
    for n in range(n_steps):
        t = n * dt
        z = rng.standard_normal(m)
        if antithetic:
            z = np.concatenate([z, -z])
        s = np.asarray(sigma(np.exp(x), t), float)
        x = x + (r - q - 0.5 * s * s) * dt + s * np.sqrt(dt) * z
        if n + 1 in steps:
            out[:, steps[n + 1]] = np.exp(x)
    times = np.array([0.0] + [k * dt for k in sorted(steps)])
    return LocalVolPaths(times, out[:n_paths], float(r), float(q))
