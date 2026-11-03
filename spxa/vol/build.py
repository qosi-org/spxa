r"""From option quotes to an arbitrage-free :class:`~spxa.vol.surface.VolSurface`.

:func:`build_vol_surface` runs the standard pipeline on
:class:`~spxa.market.quotes.OptionSurfaceQuotes`:

1. **Forwards and discount factors.** Use the chain's own values if it has
   them. Otherwise regress put-call parity, :math:`C - P = D(F - K)`, with
   :func:`~spxa.market.parity.implied_forward`.
2. **Implied total variances.** Take out-of-the-money mid prices, convert
   them to Black volatilities and record :math:`(k, w) = (\log(K/F), \sigma^2 T)`.
3. **Fit.** Use calendar-free raw SVI (``"svi"``), eSSVI (``"essvi"``) or
   SSVI (``"ssvi"``).
4. **Surface.** SVI slices are joined in time by Gatheral–Jacquier price
   interpolation, while (e)SSVI is used as is. The forwards and discount
   factors become a :class:`ForwardTermStructure`.

Weights (SVI only) multiply squared total-variance residuals:

``"uniform"``
    all ones.
``"vega"``
    :math:`(\partial c/\partial w)^2 = \varphi(d_1)^2/(4w)`, so that the fit
    minimises normalised price errors to first order.
``"spread"``
    :math:`(\partial c/\partial w \cdot DF/\text{spread})^2`, so that each
    residual is measured in units of its quote's bid-ask spread.

Both non-uniform choices are normalised to mean one per expiry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from spxa.market.parity import ParityFit, implied_forward
from spxa.market.quotes import OptionChain, OptionSurfaceQuotes
from spxa.vol.calibration import calibrate_ssvi, calibrate_svi_surface
from spxa.vol.essvi import calibrate_essvi
from spxa.vol.interpolation import PriceInterpolatedSurface
from spxa.vol.surface import VolSurface

__all__ = ["ForwardTermStructure", "SurfaceBuild", "build_vol_surface", "chain_total_variance"]


class ForwardTermStructure:
    r"""Forwards and discount factors interpolated between option expiries.

    :math:`\log F` and :math:`\log D` are linear in :math:`t` between nodes,
    which means constant carry and flat discount forwards. The node
    :math:`(0, \log S)` is used when ``spot`` is given; otherwise the
    forward is flat before the first expiry. :math:`\log D` always starts
    at :math:`(0, 0)`. Both extrapolate the last segment beyond the last
    expiry.
    """

    def __init__(
        self,
        expiries: Sequence[float],
        forwards: Sequence[float],
        discounts: Sequence[float],
        spot: Optional[float] = None,
    ) -> None:
        T = np.asarray(expiries, float)
        F = np.asarray(forwards, float)
        D = np.asarray(discounts, float)
        if T.ndim != 1 or T.size == 0 or T[0] <= 0 or np.any(np.diff(T) <= 0):
            raise ValueError("expiries must be positive and strictly increasing")
        if F.shape != T.shape or D.shape != T.shape or np.any(F <= 0) or np.any(D <= 0):
            raise ValueError("need one positive forward and discount factor per expiry")
        self.expiries, self.forwards, self.discounts = T, F, D
        self.spot = None if spot is None else float(spot)
        if self.spot is not None:
            self._tf = np.concatenate([[0.0], T])
            self._lf = np.log(np.concatenate([[self.spot], F]))
        else:
            self._tf, self._lf = T, np.log(F)
        self._td = np.concatenate([[0.0], T])
        self._ld = np.concatenate([[0.0], np.log(D)])

    @staticmethod
    def _interp(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.interp(t, x, y)
        if x.size > 1:
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            out = np.where(t > x[-1], y[-1] + slope * (t - x[-1]), out)
        return out

    def forward(self, t: Any) -> Any:
        tt = np.asarray(t, float)
        if np.any(tt < 0):
            raise ValueError("negative time")
        out = np.exp(self._interp(tt, self._tf, self._lf))
        return float(out) if out.ndim == 0 else out

    def discount(self, t: Any) -> Any:
        tt = np.asarray(t, float)
        if np.any(tt < 0):
            raise ValueError("negative time")
        out = np.exp(self._interp(tt, self._td, self._ld))
        return float(out) if out.ndim == 0 else out


def _dc_dw(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    d1 = -k / sw + 0.5 * sw
    return np.exp(-0.5 * d1 * d1) / (math.sqrt(2 * math.pi) * 2 * sw)


def chain_total_variance(
    chain: OptionChain,
    forward: float,
    discount: float,
    weights: str = "uniform",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""``(k, w, weights)`` from the OTM mid quotes of one chain.

    Quotes without a mid, or whose price breaks the arbitrage bounds, are
    dropped.
    """
    if weights not in ("uniform", "vega", "spread"):
        raise ValueError("weights must be 'uniform', 'vega' or 'spread'")
    K, vol, _ = chain.implied_vols(forward, discount)
    otm = chain.otm(forward)
    spreads = {q.strike: q.spread for q in otm.quotes if q.mid is not None}
    ok = np.isfinite(vol) & (vol > 0)
    K, vol = K[ok], vol[ok]
    k = np.log(K / forward)
    w = vol * vol * chain.T
    if weights == "uniform" or k.size == 0:
        wt = np.ones_like(k)
    else:
        g = _dc_dw(k, w)
        if weights == "vega":
            wt = g * g
        else:
            sp = np.array([spreads.get(float(x)) or np.nan for x in K])
            if np.any(~np.isfinite(sp) | (sp <= 0)):
                raise ValueError("spread weights need two-sided quotes with positive spreads")
            wt = (g * discount * forward / sp) ** 2
        wt = wt / np.mean(wt)
    return k, w, wt


@dataclass
class SurfaceBuild:
    """Result of :func:`build_vol_surface`."""

    surface: VolSurface
    method: str
    expiries: np.ndarray
    forwards: np.ndarray
    discounts: np.ndarray
    ks: list[np.ndarray]
    ws: list[np.ndarray]
    weights: list[np.ndarray]
    fit: Any
    parity: list[Optional[ParityFit]] = field(default_factory=list)

    def model_total_variance(self, i: int) -> np.ndarray:
        """Fitted total variance at the data of expiry ``i``."""
        return np.asarray(self.surface.total_variance(self.ks[i], self.expiries[i]), float)

    @property
    def vol_rmse(self) -> np.ndarray:
        """Root-mean-square implied-volatility error per expiry."""
        out = []
        for i, T in enumerate(self.expiries):
            m = np.sqrt(self.model_total_variance(i) / T)
            out.append(float(np.sqrt(np.mean((m - np.sqrt(self.ws[i] / T)) ** 2))))
        return np.array(out)


def build_vol_surface(
    quotes: OptionSurfaceQuotes,
    method: str = "svi",
    weights: str = "uniform",
    spot: Optional[float] = None,
    min_quotes: int = 5,
    **fit_kwargs: Any,
) -> SurfaceBuild:
    r"""Fit an arbitrage-free implied-volatility surface to option quotes.

    Parameters
    ----------
    quotes
        Chains with bid/ask quotes. A chain's ``forward`` and ``discount``
        are used if set, otherwise they are implied from put-call parity.
    method : {"svi", "essvi", "ssvi"}
    weights : {"uniform", "vega", "spread"}
        Residual weights. Only ``"svi"`` accepts non-uniform weights.
    spot
        Anchors the forward curve at :math:`t = 0` (defaults to
        ``quotes.spot``).
    min_quotes
        Chains with fewer usable quotes are skipped.
    fit_kwargs
        Passed to the calibrator.
    """
    if method not in ("svi", "essvi", "ssvi"):
        raise ValueError("method must be 'svi', 'essvi' or 'ssvi'")
    if weights != "uniform" and method != "svi":
        raise ValueError("non-uniform weights are only supported by method='svi'")
    T, F, D, ks, ws, wts, par = [], [], [], [], [], [], []
    for chain in quotes:
        fit: Optional[ParityFit] = None
        if chain.forward is not None and chain.discount is not None:
            f, d = float(chain.forward), float(chain.discount)
        else:
            fit = implied_forward(chain, chain.discount)
            f, d = fit.forward, fit.discount
        k, w, wt = chain_total_variance(chain, f, d, weights)
        if k.size < max(min_quotes, 5):
            continue
        T.append(chain.T)
        F.append(f)
        D.append(d)
        ks.append(k)
        ws.append(w)
        wts.append(wt)
        par.append(fit)
    if not T:
        raise ValueError("no chain has enough usable quotes")
    TT = np.array(T)
    model: Any
    if method == "svi":
        res: Any = calibrate_svi_surface(
            TT, ks, ws, None if weights == "uniform" else wts, **fit_kwargs
        )
        model = PriceInterpolatedSurface(TT, res.slices)
    elif method == "essvi":
        res = calibrate_essvi(TT, ks, ws, **fit_kwargs)
        model = res
    else:
        res = calibrate_ssvi(TT, ks, ws, **fit_kwargs)
        model = res.surface
    s0 = spot if spot is not None else quotes.spot
    fwd = ForwardTermStructure(TT, F, D, s0)
    return SurfaceBuild(
        VolSurface(model, fwd), method, TT, np.array(F), np.array(D), ks, ws, wts, res, par
    )
