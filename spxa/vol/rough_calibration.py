r"""
Calibration of rough Heston to implied-volatility smiles.

The parameters are fitted by least squares on implied volatilities,

.. math:: \min_p \sum_j w_j\bigl(\sigma^{\text{model}}_j(p) - \sigma^{\text{mkt}}_j\bigr)^2,

over a chosen subset of :math:`(V_0, \theta, \lambda, \nu, \rho, H)`; the
others stay at their initial values. Each maturity needs one fractional
Riccati solve for all its strikes, so the cost is a handful of solves per
function evaluation. The mean-reversion speed :math:`\lambda` is weakly
identified by a few maturities and is held fixed by default, as is common
practice (Gatheral, Jaisson & Rosenbaum, 2018).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from spxa.vol.rough_heston import RoughHeston

__all__ = ["RoughHestonFit", "calibrate_rough_heston"]

_NAMES = ("v0", "theta", "lam", "nu", "rho", "hurst")
_BOUNDS = {
    "v0": (1e-4, 1.0),
    "theta": (1e-4, 1.0),
    "lam": (0.0, 10.0),
    "nu": (1e-3, 2.0),
    "rho": (-0.999, 0.999),
    "hurst": (0.01, 0.5),
}


@dataclass(frozen=True)
class RoughHestonFit:
    model: RoughHeston
    vols: np.ndarray
    targets: np.ndarray
    nfev: int

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean((self.vols - self.targets) ** 2)))


def _model_vols(
    m: RoughHeston, smiles: Sequence[tuple[float, np.ndarray]], F: Sequence[float], **kw: int
) -> np.ndarray:
    return np.concatenate(
        [np.atleast_1d(m.implied_vol(K, T, f, **kw)) for (T, K), f in zip(smiles, F)]
    )


def calibrate_rough_heston(
    maturities: Sequence[float],
    strikes: Sequence[Sequence[float]],
    vols: Sequence[Sequence[float]],
    initial: RoughHeston,
    free: Sequence[str] = ("v0", "theta", "nu", "rho", "hurst"),
    forwards: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[Sequence[float]]] = None,
    n_steps: int = 100,
) -> RoughHestonFit:
    """Fit the ``free`` parameters of ``initial`` to smiles ``vols[j]`` at ``strikes[j]``, ``maturities[j]``."""
    bad = set(free) - set(_NAMES)
    if bad:
        raise ValueError(f"unknown parameters {sorted(bad)}")
    smiles = [(float(T), np.asarray(K, float)) for T, K in zip(maturities, strikes)]
    F = [1.0] * len(smiles) if forwards is None else [float(f) for f in forwards]
    target = np.concatenate([np.asarray(v, float) for v in vols])
    w = (
        np.ones_like(target)
        if weights is None
        else np.concatenate([np.asarray(x, float) for x in weights])
    )
    if target.size != sum(K.size for _, K in smiles) or w.size != target.size:
        raise ValueError("strikes, vols and weights must have matching shapes")
    base = {n: getattr(initial, n) for n in _NAMES}

    def build(p: np.ndarray) -> RoughHeston:
        d = dict(base)
        d.update(zip(free, p))
        return RoughHeston(**d)

    def resid(p: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            mv = _model_vols(build(p), smiles, F, n_steps=n_steps)
        return np.sqrt(w) * np.nan_to_num(mv - target, nan=1.0)

    lo = [_BOUNDS[n][0] for n in free]
    hi = [_BOUNDS[n][1] for n in free]
    x0 = np.clip([base[n] for n in free], lo, hi)
    res = least_squares(
        resid, x0, bounds=(lo, hi), x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12
    )
    model = build(res.x)
    return RoughHestonFit(
        model, _model_vols(model, smiles, F, n_steps=n_steps), target, int(res.nfev)
    )
