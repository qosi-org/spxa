r"""
Calibration of rough Bergomi to a smile with common random numbers.

Rough Bergomi has no closed form, so each objective evaluation is a Monte
Carlo run. Reusing the same random numbers for every parameter set (common
random numbers) makes the simulated implied volatilities a smooth,
deterministic function of :math:`(H, \eta, \rho, \xi_0)`. A gradient-based
least-squares solver then works as for a closed-form model, and the fit is
exact for the simulated model: calibrating to prices generated with the
same seed returns the true parameters to solver precision. The remaining
error against market data is the Monte Carlo error of the fixed sample.
The conditional estimator of :mod:`spxa.vol.rbergomi` keeps that error
small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from spxa.vol.rbergomi import rbergomi_paths

__all__ = ["RBergomiFit", "calibrate_rbergomi"]

_BOUNDS = {"hurst": (0.02, 0.45), "eta": (0.05, 5.0), "rho": (-0.99, 0.5), "xi0": (1e-4, 1.0)}


@dataclass(frozen=True)
class RBergomiFit:
    hurst: float
    eta: float
    rho: float
    xi0: float
    vols: np.ndarray
    targets: np.ndarray
    nfev: int

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean((self.vols - self.targets) ** 2)))


def calibrate_rbergomi(
    T: float,
    strikes: Sequence[float],
    vols: Sequence[float],
    initial: dict[str, float],
    free: Sequence[str] = ("hurst", "eta", "rho"),
    n_steps: int = 50,
    n_paths: int = 10_000,
    seed: int = 0,
) -> RBergomiFit:
    """Fit the ``free`` entries of ``initial`` (keys ``hurst, eta, rho, xi0``) to a smile at ``T`` (unit forward)."""
    if set(initial) != set(_BOUNDS) or not set(free) <= set(_BOUNDS):
        raise ValueError(f"initial needs the keys {sorted(_BOUNDS)}; free must be a subset")
    K = np.asarray(strikes, float)
    target = np.asarray(vols, float)
    if K.shape != target.shape:
        raise ValueError("strikes and vols must have the same shape")

    def smile(p: dict[str, float]) -> np.ndarray:
        paths = rbergomi_paths(
            p["hurst"], p["eta"], p["rho"], p["xi0"], T, n_steps, n_paths, seed=seed
        )
        return paths.implied_vol(K)

    def params(x: np.ndarray) -> dict[str, float]:
        d = dict(initial)
        d.update(zip(free, map(float, x)))
        return d

    def resid(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(smile(params(x)) - target, nan=1.0)

    lo = [_BOUNDS[f][0] for f in free]
    hi = [_BOUNDS[f][1] for f in free]
    x0 = np.clip([initial[f] for f in free], lo, hi)
    res = least_squares(
        resid,
        x0,
        bounds=(lo, hi),
        x_scale="jac",
        diff_step=1e-6,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    p = params(res.x)
    return RBergomiFit(p["hurst"], p["eta"], p["rho"], p["xi0"], smile(p), target, int(res.nfev))
