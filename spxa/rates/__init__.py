r"""
spxa.rates — interest-rate instruments, curves and models.

Indices
    :class:`OvernightIndex` (SOFR, €STR, SONIA), :class:`IborIndex`
    (EURIBOR 1M–12M), :class:`FixingHistory`, :func:`get_index`.
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
]
