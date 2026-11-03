r"""Policy-meeting step curves.

Overnight rates move when a central bank meets and stay put in between.
A *meeting-date curve* has one flat instantaneous forward per
inter-meeting period: for step dates :math:`s_1 < \dots < s_n` (the first
day each new policy rate applies) and segments
:math:`[s_{j-1}, s_j)` with :math:`s_0` the reference date,

.. math::

    f(t) = r_j \quad (t \in [s_{j-1}, s_j)),

which is a :class:`~spxa.rates.interpolation.PiecewiseCurve` under
:class:`~spxa.rates.interpolation.LogLinearDiscount` with nodes at the
step dates. :func:`meeting_date_curve` solves for the :math:`r_j` from
any helpers: exactly when there is one per segment, and by Gauss–Newton
least squares when the helpers over-determine the steps (for example
1M futures, several of which can fall between two meetings).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from spxa.market.curves import YieldCurve
from spxa.market.dates import DateLike, to_date
from spxa.market.daycount import DayCount
from spxa.rates.helpers import CurveHelper
from spxa.rates.interpolation import LogLinearDiscount, PiecewiseCurve

__all__ = ["flat_forward_curve", "meeting_date_curve", "MeetingCurveResult"]


def flat_forward_curve(
    reference_date: DateLike,
    dates: Sequence[DateLike],
    rates: Sequence[float],
    day_count: "str | DayCount" = "ACT/365F",
) -> PiecewiseCurve:
    """Curve with continuous forward ``rates[j]`` on ``[dates[j-1], dates[j])``.

    ``dates[-1]`` ends the last segment, and the reference date starts
    the first. Beyond the last date the final rate continues flat.
    """
    ref = to_date(reference_date)
    ds = [to_date(d) for d in dates]
    r = np.asarray(rates, dtype=float)
    if r.shape != (len(ds),) or not ds:
        raise ValueError("need one rate per date")
    if ds[0] <= ref or any(b <= a for a, b in zip(ds, ds[1:])):
        raise ValueError("dates must be after the reference date and strictly increasing")
    dc = DayCount(day_count) if isinstance(day_count, str) else day_count
    t = np.array([dc.year_fraction(ref, d) for d in ds])
    x = np.cumsum(r * np.diff(np.concatenate([[0.0], t])))
    return PiecewiseCurve(t, np.exp(-x), LogLinearDiscount(), reference_date=ref, day_count=dc)


@dataclass
class MeetingCurveResult:
    """Output of :func:`meeting_date_curve`."""

    curve: PiecewiseCurve
    #: end date of each segment: the step dates in range, then the last pillar
    dates: list[_dt.date]
    #: continuous forward on each segment
    rates: np.ndarray
    helpers: list[CurveHelper]
    residuals: np.ndarray
    iterations: int
    name: str = "discount"
    known: dict[str, YieldCurve] = field(default_factory=dict)

    @property
    def max_residual(self) -> float:
        return float(np.max(np.abs(self.residuals)))

    @property
    def curves(self) -> dict[str, YieldCurve]:
        return {**self.known, self.name: self.curve}


def meeting_date_curve(
    helpers: Sequence[CurveHelper],
    reference_date: DateLike,
    step_dates: Sequence[DateLike],
    name: str = "discount",
    known: Optional[Mapping[str, YieldCurve]] = None,
    day_count: str = "ACT/365F",
    weights: Optional[Sequence[float]] = None,
    max_iter: int = 20,
    tol: float = 1e-14,
) -> MeetingCurveResult:
    r"""Fit one flat forward per inter-meeting period to the helpers.

    Parameters
    ----------
    helpers
        Instruments reading the curve ``name`` (and curves in ``known``).
    step_dates
        First day of each new policy rate. Steps on or before the
        reference date, or on or after the last pillar, are ignored; the
        last segment ends at the last pillar.
    weights
        Residual weights for the least-squares case (default 1).

    The unknowns are the segment rates :math:`r_j`. Each Gauss–Newton step
    solves :math:`\min \|W(q(r) - q^{mkt}) + WJ\,\delta\|` by least squares
    with a finite-difference Jacobian. A segment that no helper depends on
    makes :math:`J` rank-deficient, and that raises ``ValueError``.
    """
    if not helpers:
        raise ValueError("need at least one helper")
    ref = to_date(reference_date)
    known = dict(known or {})
    hs = list(helpers)
    last = max(h.pillar for h in hs)
    if last <= ref:
        raise ValueError("pillars must be after the reference date")
    steps = sorted({to_date(d) for d in step_dates})
    dates = [d for d in steps if ref < d < last] + [last]
    n, m = len(dates), len(hs)
    if m < n:
        raise ValueError(f"{n} segments but only {m} helpers")
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (m,) or np.any(w <= 0):
        raise ValueError("need one positive weight per helper")
    quotes = np.array([h.quote for h in hs])

    def curve(r: np.ndarray) -> PiecewiseCurve:
        return flat_forward_curve(ref, dates, r, day_count)

    def resid(r: np.ndarray) -> np.ndarray:
        cs = {**known, name: curve(r)}
        return w * (np.array([h.implied_quote(cs) for h in hs]) - quotes)

    def jac(r: np.ndarray) -> np.ndarray:
        J = np.empty((m, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1e-6
            J[:, j] = (resid(r + e) - resid(r - e)) / 2e-6
        return J

    def solve(J: np.ndarray, f: np.ndarray) -> np.ndarray:
        step, _, rank, sv = np.linalg.lstsq(J, -f, rcond=None)
        # a column with no dependence is pure rounding noise in the finite differences
        if rank < n or sv[-1] < 1e-7 * sv[0]:
            raise ValueError("a segment rate is not identified by the helpers")
        return step

    # Quotes are close to linear in the segment rates, so the Jacobian is kept
    # while steps contract and is refreshed to confirm convergence: the
    # result solves the normal equations of the true Jacobian.
    r = np.full(n, 0.02)
    f = resid(r)
    J, fresh, prev = jac(r), True, np.inf
    it = 0
    for it in range(1, max_iter + 1):
        step = solve(J, f)
        r = r + step
        f = resid(r)
        size = float(np.max(np.abs(step)))
        if size < tol and fresh:
            break
        if size < tol or size > 0.25 * prev:
            J, fresh = jac(r), True
        else:
            fresh = False
        prev = size
    return MeetingCurveResult(curve(r), dates, r, hs, f / w, it, name, known)
