r"""
spxa.rates — interest-rate instruments, curves and models.

Indices
    :class:`OvernightIndex` (SOFR, €STR, SONIA), :class:`IborIndex`
    (EURIBOR 1M–12M), :class:`FixingHistory`, :func:`get_index`.
Overnight compounding
    :func:`compounded_overnight_rate` with ISDA 2021 lookback, observation
    shift and lockout; :func:`observation_schedule`; curve projection with
    :func:`forward_overnight_rate` and :func:`projected_overnight_rate`.
Cashflows and legs
    :class:`FixedCoupon`, :class:`IborCoupon`, :class:`OvernightCoupon`,
    :class:`SimpleCashflow`; :func:`fixed_leg`, :func:`ibor_leg`,
    :func:`overnight_leg`.
Money-market instruments
    :class:`Deposit`, :class:`FRA`.
Futures
    :func:`imm_date`, :func:`next_imm_date`, :func:`imm_code`,
    :func:`imm_from_code`, :class:`RateFuture` (term rates) and
    :class:`OvernightFuture` (compounded 3M / averaged 1M SOFR), with exact
    Hull–White convexity.
Swaps
    :class:`Swap`, :func:`ois_swap`, :func:`vanilla_swap`,
    :class:`BasisSwap`, :func:`basis_swap`.
Bonds
    :class:`FixedRateBond` (street yield, duration, convexity, z-spread).
Pricing
    :func:`leg_pv`, :func:`leg_annuity`, :func:`cashflow_table` on
    separate discount and projection curves.
"""

from spxa.rates.bonds import BondCashflow, FixedRateBond
from spxa.rates.cashflows import (
    Cashflow,
    FixedCoupon,
    IborCoupon,
    Leg,
    OvernightCoupon,
    SimpleCashflow,
    fixed_leg,
    ibor_leg,
    overnight_leg,
)
from spxa.rates.futures import (
    OvernightFuture,
    RateFuture,
    hull_white_compounded_convexity,
    hull_white_compounding_covariance,
    hull_white_discrete_compounded_convexity,
    hull_white_futures_convexity,
    hull_white_rate_convexity,
    imm_code,
    imm_date,
    imm_from_code,
    is_imm_date,
    next_imm_date,
)
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
from spxa.rates.pricing import CashflowRow, cashflow_table, leg_annuity, leg_pv
from spxa.rates.instruments import FRA, Deposit
from spxa.rates.swaps import BasisSwap, Swap, basis_swap, ois_swap, vanilla_swap
from spxa.rates.overnight import (
    CompoundedRate,
    compounded_overnight_rate,
    curve_projection,
    forward_overnight_rate,
    observation_schedule,
    projected_overnight_rate,
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
    "curve_projection",
    "forward_overnight_rate",
    "observation_schedule",
    "projected_overnight_rate",
    # cashflows
    "Cashflow",
    "FixedCoupon",
    "IborCoupon",
    "Leg",
    "OvernightCoupon",
    "SimpleCashflow",
    "fixed_leg",
    "ibor_leg",
    "overnight_leg",
    # instruments
    "Deposit",
    "FRA",
    # futures
    "OvernightFuture",
    "RateFuture",
    "hull_white_compounded_convexity",
    "hull_white_compounding_covariance",
    "hull_white_discrete_compounded_convexity",
    "hull_white_futures_convexity",
    "hull_white_rate_convexity",
    "imm_code",
    "imm_date",
    "imm_from_code",
    "is_imm_date",
    "next_imm_date",
    # swaps
    "BasisSwap",
    "Swap",
    "basis_swap",
    "ois_swap",
    "vanilla_swap",
    # bonds
    "BondCashflow",
    "FixedRateBond",
    # pricing
    "CashflowRow",
    "cashflow_table",
    "leg_annuity",
    "leg_pv",
]
