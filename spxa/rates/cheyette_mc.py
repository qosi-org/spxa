r"""Monte Carlo for the Cheyette model.

Over each step the local volatility :math:`\sigma_r(t_k, x_k)` and
:math:`y_k` in the drift of :math:`x` are frozen. The step is then solved
exactly:

.. math::
    x_{k+1} = e^{-\kappa h}x_k + y_k\frac{1 - e^{-\kappa h}}{\kappa}
        + \sigma_r\sqrt{\tfrac{1 - e^{-2\kappa h}}{2\kappa}}\,Z, \qquad
    y_{k+1} = e^{-2\kappa h}y_k + \sigma_r^2\,\frac{1 - e^{-2\kappa h}}{2\kappa}.

For :math:`\beta = 0` this is the exact Hull–White transition in
:math:`x`, with :math:`y` exact. The bank-account deflator is
:math:`D(t) = P(0, t)\,e^{-\int_0^t x_u du}`, with the integral by the
trapezoidal rule. Deflated bonds :math:`D(t)P(t, T | x_t, y_t)` then
reprice the curve up to the discretisation error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from spxa.rates.cheyette import Cheyette

__all__ = ["CheyettePaths", "cheyette_paths"]


@dataclass(frozen=True)
class CheyettePaths:
    model: Cheyette
    times: np.ndarray  # recorded times
    x: np.ndarray  # (n_times, n_paths)
    y: np.ndarray
    deflator: np.ndarray

    def _row(self, t: float) -> int:
        k = int(np.argmin(np.abs(self.times - t)))
        if abs(self.times[k] - t) > 1e-10:
            raise ValueError(f"time {t} was not recorded")
        return k

    def zero_bond(self, t: float, T: float) -> np.ndarray:
        k = self._row(t)
        return np.asarray(self.model.discount_bond(t, T, self.x[k], self.y[k]))

    def price(self, payoff: np.ndarray, t: float) -> tuple[float, float]:
        v = payoff * self.deflator[self._row(t)]
        return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size))


def cheyette_paths(
    model: Cheyette,
    horizon: float,
    n_steps: int,
    n_paths: int,
    record: Optional[Sequence[float]] = None,
    seed: int = 0,
    antithetic: bool = True,
) -> CheyettePaths:
    """Simulate to ``horizon`` in ``n_steps`` steps, storing the states at ``record`` (default: all steps)."""
    if horizon <= 0 or n_steps < 1:
        raise ValueError("need horizon > 0 and n_steps >= 1")
    h = horizon / n_steps
    grid = np.arange(n_steps + 1) * h
    rec = grid if record is None else np.asarray(record, float)
    idx = np.rint(rec / h).astype(int)
    if np.any(np.abs(idx * h - rec) > 1e-9 * max(1.0, horizon)) or np.any(
        (idx < 0) | (idx > n_steps)
    ):
        raise ValueError("record times must lie on the step grid")
    k = model.kappa
    e1, e2 = math.exp(-k * h), math.exp(-2 * k * h)
    b1, b2 = -math.expm1(-k * h) / k, -math.expm1(-2 * k * h) / (2 * k)
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    n = 2 * half if antithetic else n_paths
    x, y, ix = np.zeros(n), np.zeros(n), np.zeros(n)
    out = {name: np.empty((rec.size, n)) for name in ("x", "y", "d")}
    where = {int(i): j for j, i in enumerate(idx)}

    def store(step: int) -> None:
        if step in where:
            j = where[step]
            out["x"][j], out["y"][j] = x, y
            out["d"][j] = float(model.curve.discount(step * h)) * np.exp(-ix)

    store(0)
    for step in range(n_steps):
        z = rng.standard_normal(half)
        if antithetic:
            z = np.concatenate([z, -z])
        s = model.local_vol(step * h, x)
        xn = e1 * x + b1 * y + s * math.sqrt(b2) * z
        y = e2 * y + s * s * b2
        ix += 0.5 * (x + xn) * h
        x = xn
        store(step + 1)
    return CheyettePaths(model, rec, out["x"], out["y"], out["d"])
