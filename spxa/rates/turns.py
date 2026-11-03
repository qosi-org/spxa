r"""Turn-of-year and other deterministic forward-rate spreads.

Funding gets more expensive over balance-sheet dates such as the year end.
In a curve this shows up as a spike in the overnight forward that no
smooth interpolation can produce. A :class:`ForwardSpread` adds a constant
spread :math:`s` to the instantaneous forward on :math:`[t_a, t_b)`:

.. math:: P(t) = P_{\text{base}}(t)\,\exp\Bigl(-s\,\bigl|[t_a, t_b) \cap [0, t)\bigr|\Bigr).

:class:`SpreadAdjustedCurve` applies any number of such spreads to a base
curve. To bootstrap around a known turn, pass the spreads to
:func:`~spxa.rates.bootstrap.bootstrap_curve` or
:class:`~spxa.rates.bootstrap.CurveSpec`. The smooth base curve is then
solved for, with the turn on top.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from spxa.market.curves import YieldCurve

__all__ = ["ForwardSpread", "SpreadAdjustedCurve", "turn_of_year"]


@dataclass(frozen=True)
class ForwardSpread:
    """Spread ``spread`` (continuous) on the instantaneous forward over ``[start, end)``."""

    start: _dt.date
    end: _dt.date
    spread: float

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("end must be after start")


def turn_of_year(year: int, spread: float, calendar: Any = "TARGET") -> ForwardSpread:
    """Spread over the last business day of ``year`` to the first business day of ``year + 1``."""
    from spxa.market.calendars import get_calendar

    cal = get_calendar(calendar)
    last = cal.previous_business_day(_dt.date(year, 12, 31))
    first = cal.next_business_day(_dt.date(year + 1, 1, 1))
    return ForwardSpread(last, first, spread)


class SpreadAdjustedCurve(YieldCurve):
    """``base`` with deterministic forward spreads added."""

    def __init__(self, base: YieldCurve, spreads: Sequence[ForwardSpread]) -> None:
        super().__init__(base.reference_date, base.day_count)
        if spreads and base.reference_date is None:
            raise ValueError("forward spreads need a base curve with a reference date")
        self.base = base
        self.spreads = list(spreads)
        self._a = np.array([float(base.time(s.start)) for s in self.spreads])
        self._b = np.array([float(base.time(s.end)) for s in self.spreads])
        self._s = np.array([s.spread for s in self.spreads])

    def log_adjustment(self, t: np.ndarray) -> np.ndarray:
        r"""Added :math:`-\log P` at times ``t``: :math:`\sum s_j |[a_j, b_j) \cap [0, t)|`."""
        t = np.asarray(t, float)
        if not self.spreads:
            return np.zeros_like(t)
        a, b = np.maximum(self._a, 0.0), np.maximum(self._b, 0.0)
        overlap = np.clip(t[..., None], a, b) - a
        return np.sum(self._s * overlap, axis=-1)

    def _discount(self, t: np.ndarray) -> np.ndarray:
        return np.asarray(self.base.discount(t), float) * np.exp(-self.log_adjustment(t))

    def instantaneous_forward(self, t: Any, h: float = 1e-5) -> Any:
        tt, scalar = self._eval(t)
        base = np.asarray(self.base.instantaneous_forward(tt, h), float)
        inside = (tt[..., None] >= self._a) & (tt[..., None] < self._b) if self.spreads else None
        add = np.sum(self._s * inside, axis=-1) if inside is not None else 0.0
        out = base + add
        return float(out) if scalar else out

    def __repr__(self) -> str:
        return f"SpreadAdjustedCurve({self.base!r}, spreads={len(self.spreads)})"
