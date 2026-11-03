r"""
spxa.rates — interest-rate instruments, curves and models.

Indices
    :class:`OvernightIndex` (SOFR, €STR, SONIA), :class:`IborIndex`
    (EURIBOR 1M–12M), :class:`FixingHistory`, :func:`get_index`.
Overnight compounding
    :func:`compounded_overnight_rate` with ISDA 2021 lookback, observation
    shift and lockout; :func:`observation_schedule`.
"""

from spxa.rates.indices import (
    ESTR,
    EURIBOR1M,
    EURIBOR3M,
    EURIBOR6M,
    EURIBOR12M,
    SOFR,
    SONIA,
    FixingHistory,
    IborIndex,
    OvernightIndex,
    available_indices,
    get_index,
)
from spxa.rates.overnight import (
    CompoundedRate,
    compounded_overnight_rate,
    observation_schedule,
)

__all__ = [
    # indices
    "ESTR",
    "EURIBOR1M",
    "EURIBOR3M",
    "EURIBOR6M",
    "EURIBOR12M",
    "SOFR",
    "SONIA",
    "FixingHistory",
    "IborIndex",
    "OvernightIndex",
    "available_indices",
    "get_index",
    # overnight compounding
    "CompoundedRate",
    "compounded_overnight_rate",
    "observation_schedule",
]
