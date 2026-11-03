r"""Compounded and averaged overnight rates (ISDA 2021 conventions).

For a calculation period :math:`[t_s, t_e)` and an overnight index with
business days :math:`d_0 < \dots < d_{N-1}` in the period, the
compounded rate is

.. math::

    R = \frac{1}{\tau}\Bigl[\prod_{i=0}^{N-1}\bigl(1 + r_i\,\tau_i\bigr) - 1\Bigr],

where :math:`\tau_i` is the index day-count fraction of the days on which
:math:`r_i` applies and :math:`\tau = \sum_i \tau_i`. The averaged rate is
:math:`\sum_i r_i \tau_i / \tau`. ISDA 2021 defines four ways to choose
:math:`r_i` and :math:`\tau_i` so that the rate is known before payment:

``lookback = p`` (without observation shift)
    :math:`r_i` is the fixing ``p`` business days before :math:`d_i`; the
    weights :math:`\tau_i` come from the calculation period.
``observation_shift = True`` (with ``lookback = p``)
    The whole observation period is shifted back ``p`` business days, and
    both the rates and the weights come from the shifted period.
``lockout = s``
    The final ``s`` business days of the period all use the fixing of the
    *lockout date*, ``s`` business days before the period end; ``s <= 1``
    means no lockout. Lockout is not combined with a lookback.
payment delay
    Moves only the payment date and is handled by the coupon.

If the period starts on a non-business day, the fixing of the preceding
business day applies until the first business day of the period.

Spreads can be added after compounding (``"exclusive"``, the ISDA default)
or inside each daily factor (``"inclusive"``).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Union

import numpy as np

from spxa.market.dates import DateLike, to_date
from spxa.rates.indices import FixingHistory, OvernightIndex

__all__ = ["CompoundedRate", "compounded_overnight_rate", "observation_schedule"]

RateSource = Union[FixingHistory, Mapping[_dt.date, float], None]


@dataclass(frozen=True)
class CompoundedRate:
    """Result of :func:`compounded_overnight_rate`.

    Attributes
    ----------
    rate : float
        Compounded (or averaged) rate including any spread.
    accrual : float
        :math:`\\tau`, the day-count fraction the rate is quoted over.
    accrual_dates : list of date
        Start of each daily sub-period (in the shifted observation period
        when ``observation_shift`` is used).
    observation_dates : list of date
        Fixing date whose rate applies to each sub-period.
    rates, weights : ndarray
        :math:`r_i` and :math:`\\tau_i`.
    n_fixed : int
        Number of sub-periods whose rate came from published fixings.
    """

    rate: float
    accrual: float
    accrual_dates: list[_dt.date]
    observation_dates: list[_dt.date]
    rates: np.ndarray
    weights: np.ndarray
    n_fixed: int

    @property
    def growth_factor(self) -> float:
        """:math:`1 + R\\tau`."""
        return 1.0 + self.rate * self.accrual

    @property
    def fully_fixed(self) -> bool:
        return self.n_fixed == len(self.rates)


def observation_schedule(
    index: OvernightIndex,
    start: DateLike,
    end: DateLike,
    lookback: int = 0,
    lockout: int = 0,
    observation_shift: bool = False,
) -> tuple[list[_dt.date], list[_dt.date], np.ndarray, float]:
    """Accrual dates, observation dates, weights and total accrual of a period.

    See the module docstring for the conventions.
    """
    s, e = to_date(start), to_date(end)
    if e <= s:
        raise ValueError("end must be after start")
    if lookback < 0 or lockout < 0:
        raise ValueError("lookback and lockout must be non-negative")
    cal, dc = index.cal, index.dc

    def subperiods(
        a: _dt.date, b: _dt.date
    ) -> tuple[list[_dt.date], list[_dt.date], list[_dt.date]]:
        """Fixing dates applying in [a, b), the start and end of their sub-periods."""
        days = index.fixing_dates(a, b)
        fix, lo, hi = [], [], []
        if not days or days[0] > a:  # starts on a holiday: previous fixing applies
            fix.append(cal.previous_business_day(a))
            lo.append(a)
            hi.append(days[0] if days else b)
        for j, d in enumerate(days):
            fix.append(d)
            lo.append(d)
            hi.append(days[j + 1] if j + 1 < len(days) else b)
        return fix, lo, hi

    if lockout > 1 and (lookback or observation_shift):
        raise ValueError("lockout cannot be combined with a lookback or observation shift")
    if observation_shift:
        a = cal.add_business_days(s, -lookback) if lookback else s
        b = cal.add_business_days(e, -lookback) if lookback else e
        obs, lo, hi = subperiods(a, b)
    else:
        a, b = s, e
        fix, lo, hi = subperiods(s, e)
        obs = [cal.add_business_days(d, -lookback) if lookback else d for d in fix]
    weights = np.array([dc.year_fraction(x, y) for x, y in zip(lo, hi)])
    total = dc.year_fraction(a, b)
    if lockout > 1:
        lock_date = cal.add_business_days(e, -lockout)
        obs = [o if o <= lock_date else lock_date for o in obs]
    accrual_dates = lo
    return accrual_dates, obs, weights, float(total)


def compounded_overnight_rate(
    index: OvernightIndex,
    start: DateLike,
    end: DateLike,
    fixings: RateSource = None,
    projection: Optional[Callable[[_dt.date], float]] = None,
    lookback: int = 0,
    lockout: int = 0,
    observation_shift: bool = False,
    spread: float = 0.0,
    spread_compounding: str = "exclusive",
    method: str = "compounded",
) -> CompoundedRate:
    """Compounded or averaged overnight rate over ``[start, end)``.

    Parameters
    ----------
    index : OvernightIndex
    start, end : date-like
    fixings : FixingHistory or mapping, optional
        Published fixings by fixing date.
    projection : callable, optional
        ``projection(fixing_date) -> rate`` for dates without a published
        fixing (e.g. forward overnight rates from a curve).
    lookback, lockout, observation_shift
        ISDA conventions (module docstring).
    spread : float
    spread_compounding : {"exclusive", "inclusive"}
    method : {"compounded", "averaged"}
    """
    if method not in ("compounded", "averaged"):
        raise ValueError("method must be 'compounded' or 'averaged'")
    if spread_compounding not in ("exclusive", "inclusive"):
        raise ValueError("spread_compounding must be 'exclusive' or 'inclusive'")
    acc, obs, w, total = observation_schedule(
        index, start, end, lookback, lockout, observation_shift
    )
    rates = np.empty(len(obs))
    n_fixed = 0
    getter = fixings.get if fixings is not None else (lambda d: None)
    for j, d in enumerate(obs):
        v = getter(d)
        if v is not None:
            rates[j] = v
            n_fixed += 1
        elif projection is not None:
            rates[j] = projection(d)
        else:
            raise KeyError(f"no {index.name} fixing for {d} and no projection given")
    inclusive = spread_compounding == "inclusive"
    r_eff = rates + spread if inclusive else rates
    if method == "compounded":
        growth = float(np.prod(1.0 + r_eff * w))
        rate = (growth - 1.0) / total
    else:
        rate = float(np.sum(r_eff * w)) / total
    if not inclusive:
        rate += spread
    return CompoundedRate(float(rate), total, acc, obs, rates, w, n_fixed)
