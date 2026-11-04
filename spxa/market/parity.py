r"""Forwards and discount factors implied by put–call parity.

For European options on one expiry,

.. math:: C(K) - P(K) = D\,(F - K),

so call-minus-put prices are linear in the strike with slope :math:`-D` and
intercept :math:`DF`. :func:`implied_forward` fits that line by weighted
least squares (weights from the synthetic forwards' bid–ask spreads),
optionally restricted to the strikes nearest the money and with iterative
rejection of outliers. With a known discount factor only the forward is
fitted.

The relation is model-free but assumes European exercise; American puts
on dividend-paying stocks carry an early-exercise premium that biases the
implied forward upward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from spxa.market.quotes import OptionChain, OptionSurfaceQuotes

__all__ = ["ParityFit", "implied_forward", "implied_forwards"]


@dataclass(frozen=True)
class ParityFit:
    """Result of a put–call-parity fit.

    Attributes
    ----------
    forward, discount : float
    strikes : ndarray
        Strikes used in the final fit.
    residuals : ndarray
        ``(C - P) - D (F - K)`` at those strikes.
    rejected : ndarray
        Strikes discarded as outliers.
    discount_fitted : bool
        Whether ``discount`` was estimated (``False`` when it was given).
    """

    forward: float
    discount: float
    strikes: np.ndarray
    residuals: np.ndarray
    rejected: np.ndarray
    discount_fitted: bool

    @property
    def n(self) -> int:
        return int(self.strikes.size)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2))) if self.n else math.nan


def _synthetic(chain: OptionChain) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strikes, mid of C - P and the bid-ask width of C - P (NaN if one-sided)."""
    calls = {q.strike: q for q in chain.calls() if q.mid is not None}
    puts = {q.strike: q for q in chain.puts() if q.mid is not None}
    ks = sorted(set(calls) & set(puts))
    y, width = [], []
    for k in ks:
        c, p = calls[k], puts[k]
        cm, pm, cs, ps = c.mid, p.mid, c.spread, p.spread
        assert cm is not None and pm is not None
        y.append(cm - pm)
        width.append(cs + ps if cs is not None and ps is not None else math.nan)
    return np.asarray(ks, float), np.asarray(y, float), np.asarray(width, float)


def _wls(K: np.ndarray, y: np.ndarray, w: np.ndarray, D: Optional[float]) -> tuple[float, float]:
    """Weighted fit of y = D F - D K; returns (F, D)."""
    if D is not None:
        return float(np.sum(w * (K + y / D)) / np.sum(w)), float(D)
    sw = np.sum(w)
    km, ym = np.sum(w * K) / sw, np.sum(w * y) / sw
    sxx = np.sum(w * (K - km) ** 2)
    if sxx <= 0:
        raise ValueError("need at least two distinct strikes to fit the discount factor")
    slope = np.sum(w * (K - km) * (y - ym)) / sxx
    Dfit = -float(slope)
    if Dfit <= 0:
        raise ValueError(f"parity regression gives a non-positive discount factor ({Dfit:.6g})")
    return float(ym / Dfit + km), Dfit


def implied_forward(
    chain: OptionChain,
    discount: Optional[float] = None,
    n_strikes: Optional[int] = None,
    weights: str = "spread",
    outlier_threshold: Optional[float] = 4.0,
    max_iter: int = 10,
) -> ParityFit:
    """Forward (and discount factor) implied by put–call parity.

    Parameters
    ----------
    chain : OptionChain
        Needs calls and puts at common strikes.
    discount : float, optional
        Known discount factor; if omitted it is fitted from the slope.
    n_strikes : int, optional
        Use only the ``n_strikes`` strikes where :math:`|C-P|` is smallest
        (closest to the forward), where parity quotes are most liquid.
    weights : {"spread", "equal"}
        ``"spread"`` weighs each strike by the inverse square of the
        synthetic forward's bid–ask width (equal weights if any width is
        missing or zero).
    outlier_threshold : float or None
        Reject strikes whose residual exceeds this many robust standard
        deviations (1.4826 × median absolute deviation), refitting until
        nothing more is rejected. ``None`` disables rejection.
    """
    if weights not in ("spread", "equal"):
        raise ValueError("weights must be 'spread' or 'equal'")
    K, y, width = _synthetic(chain)
    if n_strikes is not None:
        if n_strikes < 1:
            raise ValueError("n_strikes must be positive")
        keep = np.sort(np.argsort(np.abs(y), kind="stable")[:n_strikes])
        K, y, width = K[keep], y[keep], width[keep]
    need = 1 if discount is not None else 2
    if K.size < need:
        raise ValueError(f"need at least {need} call/put pairs at common strikes, got {K.size}")
    if weights == "spread" and np.all(np.isfinite(width)) and np.all(width > 0):
        w = 1.0 / width**2
    else:
        w = np.ones_like(K)

    active = np.ones(K.size, bool)
    F, D = _wls(K, y, w, discount)
    thr = math.inf if outlier_threshold is None else float(outlier_threshold)
    for _ in range(max_iter if outlier_threshold is not None else 0):
        r = y - D * (F - K)
        ra = r[active]
        # robust scale, floored so that rounding noise in exact prices is never "outlying"
        scale = max(
            1.4826 * np.median(np.abs(ra - np.median(ra))), 1e-11 * float(np.max(np.abs(K)))
        )
        new = active & (np.abs(r - np.median(ra)) <= thr * scale)
        if new.sum() < need + 1 or np.array_equal(new, active):
            break
        active = new
        F, D = _wls(K[active], y[active], w[active], discount)
    if F <= 0:
        raise ValueError(f"parity implies a non-positive forward ({F:.6g})")
    res = y[active] - D * (F - K[active])
    return ParityFit(F, D, K[active], res, K[~active], discount is None)


def implied_forwards(
    surface: OptionSurfaceQuotes,
    discounts: Optional[list[Optional[float]]] = None,
    **kwargs: object,
) -> list[ParityFit]:
    """:func:`implied_forward` for every chain of a surface (same keyword arguments)."""
    ds = discounts if discounts is not None else [None] * len(surface)
    if len(ds) != len(surface):
        raise ValueError("discounts must match the number of chains")
    return [implied_forward(c, d, **kwargs) for c, d in zip(surface, ds)]
