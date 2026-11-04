r"""
Joint calibration of Heston to SPX and VIX quotes.

SPX smiles constrain the risk-neutral law of the integrated variance, and
VIX futures and options constrain the law of the forward variance at a
future date. A model that fits one need not fit the other. The joint
objective stacks

* SPX implied-volatility errors,
* VIX futures errors (in VIX points), and
* VIX option implied-volatility errors (Black on the futures),

with user weights. The model is Heston (:class:`~spxa.vol.vix.HestonVIX`).
Heston can match SPX skew and VIX futures, but its VIX smile always slopes
down, while the market's slopes up. The fitted residuals measure this
SPX/VIX tension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from spxa.vol.heston import Heston
from spxa.vol.vix import HestonVIX

__all__ = ["JointFit", "SmileQuote", "VIXQuote", "calibrate_heston_joint"]

_NAMES = ("v0", "kappa", "theta", "xi", "rho")
_LO = np.array([1e-4, 0.05, 1e-4, 0.05, -0.999])
_HI = np.array([1.0, 20.0, 1.0, 5.0, 0.999])


@dataclass(frozen=True)
class SmileQuote:
    """SPX implied volatilities at ``T`` (unit forward)."""

    T: float
    strikes: np.ndarray
    vols: np.ndarray


@dataclass(frozen=True)
class VIXQuote:
    """VIX futures at ``T`` and, optionally, VIX option implied volatilities."""

    T: float
    futures: float
    strikes: np.ndarray = field(default_factory=lambda: np.empty(0))
    vols: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass(frozen=True)
class JointFit:
    model: Heston
    spx_errors: np.ndarray
    futures_errors: np.ndarray
    vix_errors: np.ndarray
    nfev: int

    def rmse(self, part: str) -> float:
        e = {"spx": self.spx_errors, "futures": self.futures_errors, "vix": self.vix_errors}[part]
        return float(np.sqrt(np.mean(e**2))) if e.size else 0.0


def _errors(
    m: Heston, spx: Sequence[SmileQuote], vix: Sequence[VIXQuote]
) -> tuple[np.ndarray, ...]:
    e_spx = [np.asarray(m.implied_vol(q.strikes, q.T), float) - q.vols for q in spx]
    hv = HestonVIX(m)
    e_fut = [hv.futures(q.T) - q.futures for q in vix]
    e_vix = [hv.implied_vols(q.T, q.strikes) - q.vols for q in vix if len(q.strikes)]
    cat = lambda xs: np.concatenate(xs) if xs else np.empty(0)  # noqa: E731
    return cat(e_spx), np.asarray(e_fut, float), cat(e_vix)


def calibrate_heston_joint(
    spx: Sequence[SmileQuote],
    vix: Sequence[VIXQuote],
    initial: Heston,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> JointFit:
    """Fit all five Heston parameters to SPX smiles and VIX quotes.

    ``weights`` scales the (SPX vol, VIX futures, VIX vol) residual blocks. Use a zero
    weight to drop a block, e.g. ``(1, 0, 0)`` for an SPX-only fit.
    """
    if not spx and not vix:
        raise ValueError("no quotes")
    w = np.asarray(weights, float)

    def model(x: np.ndarray) -> Heston:
        return Heston(*map(float, x))

    def resid(x: np.ndarray) -> np.ndarray:
        parts = _errors(model(x), spx, vix)
        r = np.concatenate([wi * p for wi, p in zip(w, parts) if wi > 0])
        return np.nan_to_num(r, nan=1.0)

    x0 = np.clip([getattr(initial, n) for n in _NAMES], _LO, _HI)
    res = least_squares(
        resid, x0, bounds=(_LO, _HI), x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12
    )
    m = model(res.x)
    return JointFit(m, *_errors(m, spx, vix), nfev=int(res.nfev))
