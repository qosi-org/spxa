r"""Nelson–Siegel and Svensson parametric curves.

Svensson (1994) extends Nelson & Siegel (1987) by a second hump. With
:math:`x_k = t/\tau_k` and :math:`L(x) = (1 - e^{-x})/x`, the continuously
compounded zero rate and instantaneous forward are

.. math::

    z(t) &= \beta_0 + \beta_1 L(x_1) + \beta_2 \big(L(x_1) - e^{-x_1}\big)
            + \beta_3 \big(L(x_2) - e^{-x_2}\big), \\
    f(t) &= \beta_0 + \beta_1 e^{-x_1} + \beta_2 x_1 e^{-x_1}
            + \beta_3 x_2 e^{-x_2},

with :math:`f = \partial_t (t z)`. Both start at :math:`\beta_0 + \beta_1`
at :math:`t = 0` and tend to :math:`\beta_0`. Nelson–Siegel is the case
:math:`\beta_3 = 0`.

Fitting is linear in the betas once the decay times are fixed.
:func:`fit_nss_zero_rates` therefore uses *variable projection*: an
outer search over :math:`\log \tau` (a grid, then Nelder–Mead), with a
weighted linear least-squares solve for the betas inside.
:func:`fit_nss` fits to instrument helpers instead. It starts from the
zero-rate fit to a log-linear bootstrap and then minimises the helper
residuals over all parameters with Levenberg–Marquardt.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize

from spxa.market.curves import TimeLike, YieldCurve
from spxa.market.dates import DateLike, to_date
from spxa.market.daycount import DayCount
from spxa.rates.helpers import CurveHelper

__all__ = [
    "NelsonSiegelSvensson",
    "nelson_siegel",
    "nss_loadings",
    "fit_nss_zero_rates",
    "fit_nss",
    "NSSFit",
]


def _L(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x)
    nz = x > 1e-12
    out[nz] = -np.expm1(-x[nz]) / x[nz]
    small = ~nz
    out[small] = 1 - x[small] / 2
    return out


def nss_loadings(
    t: np.ndarray, tau1: float, tau2: Optional[float] = None, forward: bool = False
) -> np.ndarray:
    """Factor loadings: rows are times, columns the betas (3 or 4).

    These are zero-rate loadings, or forward loadings when ``forward``.
    """
    t = np.asarray(t, dtype=float)
    x1 = t / tau1
    e1 = np.exp(-x1)
    if forward:
        cols = [np.ones_like(t), e1, x1 * e1]
    else:
        L1 = _L(x1)
        cols = [np.ones_like(t), L1, L1 - e1]
    if tau2 is not None:
        x2 = t / tau2
        e2 = np.exp(-x2)
        cols.append(x2 * e2 if forward else _L(x2) - e2)
    return np.stack(cols, axis=-1)


class NelsonSiegelSvensson(YieldCurve):
    """Svensson zero curve, or Nelson–Siegel when ``beta3 = 0`` and ``tau2 is None``."""

    def __init__(
        self,
        beta0: float,
        beta1: float,
        beta2: float,
        beta3: float = 0.0,
        tau1: float = 1.0,
        tau2: Optional[float] = None,
        reference_date: Optional[DateLike] = None,
        day_count: "str | DayCount" = "ACT/365F",
    ) -> None:
        super().__init__(reference_date, day_count)
        if tau1 <= 0 or (tau2 is not None and tau2 <= 0):
            raise ValueError("decay times must be positive")
        if tau2 is None and beta3 != 0.0:
            raise ValueError("beta3 needs tau2")
        self.betas = np.array([beta0, beta1, beta2] + ([beta3] if tau2 is not None else []), float)
        self.tau1, self.tau2 = float(tau1), None if tau2 is None else float(tau2)

    @property
    def is_svensson(self) -> bool:
        return self.tau2 is not None

    def zero(self, t: np.ndarray) -> np.ndarray:
        return nss_loadings(t, self.tau1, self.tau2) @ self.betas

    def _discount(self, t: np.ndarray) -> np.ndarray:
        return np.exp(-self.zero(t) * t)

    def instantaneous_forward(self, t: TimeLike, h: float = 1e-5) -> Any:
        """Analytic forward (``h`` is ignored)."""
        tt, scalar = self._eval(t)
        out = nss_loadings(np.atleast_1d(tt), self.tau1, self.tau2, forward=True) @ self.betas
        return float(out[0]) if scalar else out.reshape(tt.shape)

    @property
    def params(self) -> dict[str, float]:
        names = ["beta0", "beta1", "beta2", "beta3"][: self.betas.size]
        out = dict(zip(names, map(float, self.betas)))
        out["tau1"] = self.tau1
        if self.tau2 is not None:
            out["tau2"] = self.tau2
        return out

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v:.6g}" for k, v in self.params.items())
        return f"NelsonSiegelSvensson({body})"


def nelson_siegel(
    beta0: float, beta1: float, beta2: float, tau: float, **kw: object
) -> NelsonSiegelSvensson:
    """Three-factor Nelson–Siegel curve."""
    return NelsonSiegelSvensson(beta0, beta1, beta2, 0.0, tau, None, **kw)


@dataclass
class NSSFit:
    """A fitted parametric curve and its residuals (zero rates or helper quotes)."""

    curve: NelsonSiegelSvensson
    residuals: np.ndarray
    helpers: list[CurveHelper] = field(default_factory=list)
    name: str = "discount"
    known: dict[str, YieldCurve] = field(default_factory=dict)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2)))

    @property
    def max_residual(self) -> float:
        return float(np.max(np.abs(self.residuals)))

    @property
    def curves(self) -> dict[str, YieldCurve]:
        return {**self.known, self.name: self.curve}


def _betas(
    t: np.ndarray, z: np.ndarray, sw: np.ndarray, tau1: float, tau2: Optional[float]
) -> tuple[np.ndarray, float]:
    A = nss_loadings(t, tau1, tau2) * sw[:, None]
    b, *_ = np.linalg.lstsq(A, z * sw, rcond=None)
    r = A @ b - z * sw
    return b, float(r @ r)


def fit_nss_zero_rates(
    times: Sequence[float],
    zero_rates: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    svensson: bool = True,
    tau_bounds: tuple[float, float] = (0.05, 30.0),
    reference_date: Optional[DateLike] = None,
    day_count: "str | DayCount" = "ACT/365F",
) -> NSSFit:
    """Weighted least-squares fit to continuously compounded zero rates.

    Decay times are kept inside ``tau_bounds`` and, for Svensson, ordered
    as :math:`\\tau_1 < \\tau_2`, which removes the label symmetry.
    """
    t = np.asarray(times, dtype=float)
    z = np.asarray(zero_rates, dtype=float)
    n_par = 4 if svensson else 3
    if (
        t.shape != z.shape
        or t.ndim != 1
        or t.size < n_par + (2 if svensson else 1)
        or np.any(t <= 0)
    ):
        raise ValueError("need positive times and enough zero rates for the parameters")
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != t.shape or np.any(w <= 0):
        raise ValueError("need one positive weight per rate")
    sw = np.sqrt(w)
    lo, hi = np.log(tau_bounds[0]), np.log(tau_bounds[1])

    def taus(v: np.ndarray) -> tuple[float, Optional[float]]:
        v = np.clip(v, lo, hi)
        return (float(np.exp(v[0])), float(np.exp(v[1])) if svensson else None)

    def sse(v: np.ndarray) -> float:
        if svensson and v[1] <= v[0] + 1e-6:
            return np.inf
        return _betas(t, z, sw, *taus(v))[1]

    grid = np.linspace(lo, hi, 25)
    starts = itertools.combinations(grid, 2) if svensson else ((g,) for g in grid)
    best = min((np.array(s) for s in starts), key=sse)
    res = minimize(
        sse, best, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-30, "maxiter": 4000}
    )
    v = res.x if res.fun <= sse(best) else best
    tau1, tau2 = taus(v)
    b, _ = _betas(t, z, sw, tau1, tau2)
    curve = NelsonSiegelSvensson(
        *b[:3], b[3] if svensson else 0.0, tau1, tau2, reference_date, day_count
    )
    return NSSFit(curve, curve.zero(t) - z)


def fit_nss(
    helpers: Sequence[CurveHelper],
    reference_date: DateLike,
    svensson: bool = True,
    name: str = "discount",
    known: Optional[Mapping[str, YieldCurve]] = None,
    weights: Optional[Sequence[float]] = None,
    day_count: str = "ACT/365F",
) -> NSSFit:
    """Least-squares Nelson–Siegel(–Svensson) fit to instrument quotes.

    The start is the zero-rate fit at the pillars of a log-linear
    bootstrap of the same helpers. The decay times are optimised in logs.
    """
    from spxa.rates.bootstrap import bootstrap_curve

    ref = to_date(reference_date)
    known = dict(known or {})
    hs = sorted(helpers, key=lambda h: h.pillar)
    w = np.ones(len(hs)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(hs),) or np.any(w <= 0):
        raise ValueError("need one positive weight per helper")
    boot = bootstrap_curve(hs, ref, name=name, known=known, day_count=day_count)
    tp = boot.base.node_times
    start = fit_nss_zero_rates(tp, boot.base.node_zero_rates, svensson=svensson)
    p = start.curve
    x0 = np.concatenate([p.betas, [np.log(p.tau1)], [np.log(p.tau2)] if svensson else []])
    quotes = np.array([h.quote for h in hs])
    sw = np.sqrt(w)

    def build(x: np.ndarray) -> NelsonSiegelSvensson:
        if svensson:
            return NelsonSiegelSvensson(*x[:4], np.exp(x[4]), np.exp(x[5]), ref, day_count)
        return NelsonSiegelSvensson(*x[:3], 0.0, np.exp(x[3]), None, ref, day_count)

    def resid(x: np.ndarray) -> np.ndarray:
        cs = {**known, name: build(x)}
        return sw * (np.array([h.implied_quote(cs) for h in hs]) - quotes)

    method = "lm" if len(hs) >= x0.size else "trf"
    sol = least_squares(resid, x0, method=method, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    curve = build(sol.x)
    return NSSFit(curve, sol.fun / sw, hs, name, known)
