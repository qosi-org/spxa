r"""Monte Carlo for Gaussian HJM on a maturity grid.

The state is the vector of deflated bonds :math:`Z(t, T_j) = P(t, T_j)/B(t)`
on a fixed maturity grid. Under the risk-neutral measure

.. math:: d\log Z(t, T) = -\tfrac12|\Sigma(t, T)|^2\,dt - \Sigma(t, T)\cdot dW_t,

and each step uses :math:`\Sigma` frozen at the left end. The Gaussian
increment then has
:math:`E[e^{-\frac12|\Sigma|^2\Delta t - \Sigma\cdot\Delta W}] = 1` exactly,
so every :math:`Z(\cdot, T_j)` is a discrete martingale: the paths
reprice the curve with no discretisation bias. The deflator is
:math:`1/B(t_k) = Z(t_k, t_k)`, and the bond prices are
:math:`P(t_k, T_j) = Z(t_k, T_j)/Z(t_k, t_k)`. Option prices carry an
:math:`O(\Delta t)` bias from the left-point variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from spxa.rates.hjm import GaussianHJM

__all__ = ["HJMPaths", "hjm_paths"]


@dataclass(frozen=True)
class HJMPaths:
    model: GaussianHJM
    maturities: np.ndarray  # stored maturities
    record: np.ndarray  # recorded times (on the grid)
    Z: np.ndarray  # (n_record, n_stored, n_paths)

    def _row(self, t: float) -> int:
        k = int(np.argmin(np.abs(self.record - t)))
        if abs(self.record[k] - t) > 1e-10:
            raise ValueError(f"time {t} was not recorded")
        return k

    def _col(self, T: float) -> int:
        j = int(np.argmin(np.abs(self.maturities - T)))
        if abs(self.maturities[j] - T) > 1e-10:
            raise ValueError(f"maturity {T} was not stored")
        return j

    def deflated(self, t: float, T: float) -> np.ndarray:
        """:math:`Z(t, T) = P(t, T)/B(t)`."""
        return self.Z[self._row(t), self._col(T)]

    def deflator(self, t: float) -> np.ndarray:
        return self.deflated(t, t)

    def zero_bond(self, t: float, T: float) -> np.ndarray:
        return self.deflated(t, T) / self.deflator(t)

    def price(self, payoff: np.ndarray, t: float) -> tuple[float, float]:
        v = payoff * self.deflator(t)
        return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size))


def hjm_paths(
    model: GaussianHJM,
    maturities: Sequence[float],
    n_paths: int,
    record: Optional[Sequence[float]] = None,
    keep: Optional[Sequence[float]] = None,
    seed: int = 0,
    antithetic: bool = True,
) -> HJMPaths:
    """Simulate on the grid ``maturities`` (also the time steps).

    ``Z(t, T)`` is stored for ``t`` in ``record`` and ``T`` in ``keep``
    (both default to the whole grid; both must lie on it).
    """
    mats = np.asarray(maturities, float)
    if mats[0] != 0.0 or np.any(np.diff(mats) <= 0):
        raise ValueError("maturities must start at 0 and increase")

    def on_grid(xs: np.ndarray, what: str) -> list[int]:
        idx = [int(np.argmin(np.abs(mats - x))) for x in xs]
        if any(abs(mats[i] - x) > 1e-10 for i, x in zip(idx, xs)):
            raise ValueError(f"{what} must lie on the maturity grid")
        return idx

    rec = mats if record is None else np.asarray(record, float)
    kept = mats if keep is None else np.asarray(keep, float)
    rec_idx, keep_idx = on_grid(rec, "record times"), on_grid(kept, "kept maturities")
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    n = half * 2 if antithetic else n_paths
    logZ = np.repeat(np.log(np.asarray(model.curve.discount(mats), float))[:, None], n, axis=1)
    out = np.empty((len(rec), len(keep_idx), n))
    k_rec = {i: r for r, i in enumerate(rec_idx)}
    if 0 in k_rec:
        out[k_rec[0]] = np.exp(logZ[keep_idx])
    for k in range(mats.size - 1):
        t, dt = mats[k], mats[k + 1] - mats[k]
        live = slice(k + 1, None)  # maturities after t
        Sig = model.Sigma(t, mats[live])  # (n_factors, n_live)
        eps = rng.standard_normal((model.n_factors, half))
        if antithetic:
            eps = np.concatenate([eps, -eps], axis=1)
        dW = math.sqrt(dt) * eps
        logZ[live] += -0.5 * np.sum(Sig**2, axis=0)[:, None] * dt - Sig.T @ dW
        # maturities reached keep their value: Z(t, T) for t >= T is frozen at 1/B(T)
        if k + 1 in k_rec:
            out[k_rec[k + 1]] = np.exp(logZ[keep_idx])
    return HJMPaths(model, kept, rec, out)
