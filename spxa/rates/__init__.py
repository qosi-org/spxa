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
    :class:`BasisSwap`, :func:`basis_swap`; amortising notionals with
    :func:`step_notional` and :func:`linear_notional`.
Bonds
    :class:`FixedRateBond` (street yield, duration, convexity, z-spread).
Pricing
    :func:`leg_pv`, :func:`leg_annuity`, :func:`cashflow_table` on
    separate discount and projection curves.
Curve construction
    :class:`PiecewiseCurve` on discount-factor nodes with the
    interpolation schemes :class:`LogLinearDiscount`, :class:`LinearZero`,
    :class:`CubicSplineRT`, :class:`CubicZero` and
    :class:`MonotoneCubicRT` (Hyman-filtered, :func:`hyman_slopes`) and
    Hagan–West :class:`MonotoneConvex`.
Calibration helpers
    :class:`DepositHelper`, :class:`FRAHelper`, :class:`FutureHelper`,
    :class:`SwapHelper`, :class:`BasisSwapHelper` (quote, pillar, curve
    names and model quote on a mapping of named curves);
    :func:`quarterly_futures` builds a consecutive IMM futures strip;
    :class:`BondHelper` quotes a fixed-rate bond by clean price or yield.
Bootstrapping
    :func:`bootstrap_curve` (:class:`BootstrapResult`): pillar-by-pillar
    Brent solves, exact in one pass for local schemes and iterated for
    non-local ones; other curves are passed as ``known``.
    :func:`fit_curves` (:class:`CurveSpec`, :class:`GlobalFitResult`)
    solves several curves at once by Newton's method.
Scheme diagnostics
    :func:`compare_schemes` (:class:`SchemeDiagnostics`),
    :func:`bump_locality` and :func:`forward_roughness`.
Curve risk
    :func:`node_deltas` (:class:`NodeRisk`), :func:`quote_deltas`
    (:class:`QuoteRisk`), :func:`quote_jacobian`, :func:`curve_nodes` and
    :func:`with_node_values`; key-rate deltas :func:`key_rate_deltas`
    (:class:`KeyRateRisk`, :class:`KeyRateShiftedCurve`, :func:`key_rate_hats`).
    Hedging: :func:`hedge_notionals` and :func:`par_hedge` (:class:`HedgeResult`).
    P&L explain: :func:`pnl_explain` (:class:`PnLExplain`) with :func:`node_gamma`.
    Principal components of curve moves: :func:`curve_pca` (:class:`CurvePCA`).
Curve sets
    :class:`CurveSet`: named curves with a discount role and index
    projections; prices swaps and basis swaps by leg index.
Parametric curves
    :class:`NelsonSiegelSvensson`, :func:`nelson_siegel`,
    :func:`fit_nss_zero_rates` (variable projection) and :func:`fit_nss`
    (helpers), returning :class:`NSSFit`.
Meeting-date curves
    :func:`flat_forward_curve`, :func:`meeting_date_curve`
    (:class:`MeetingCurveResult`): one flat forward per policy period.
Forward spreads
    :class:`ForwardSpread`, :func:`turn_of_year`,
    :class:`SpreadAdjustedCurve`; bootstrapping takes ``spreads=`` to fit
    the smooth curve under a known turn.
Caps and floors
    :class:`Caplet`, :class:`CapFloor`, :func:`cap_floor` (Black, Bachelier or shifted Black; flat implied vol).
Swaptions
    :class:`Swaption` (physical or cash settled), :func:`swaption`, :func:`cash_annuity`.
SABR
    :class:`SABR`, :func:`sabr_lognormal_vol`, :func:`sabr_normal_vol` (Hagan 2002 or Obłój, optional shift);
    :func:`calibrate_sabr` and :func:`alpha_from_atm`.
Arbitrage-free SABR
    :class:`ArbitrageFreeSABR` (Hagan et al. 2014 density PDE, Lawson–Swayne steps) and :class:`SABRDensity`.
SABR Monte Carlo
    :func:`sabr_paths` (exact volatility, Euler forward, absorption with a Brownian bridge for beta = 0).
Caplet stripping
    :func:`strip_caplet_vols` bootstraps a piecewise-constant :class:`CapletVolCurve` from flat cap quotes.
"""

from spxa.rates.stripping import CapletVolCurve, strip_caplet_vols

from spxa.rates.sabr_mc import SABRPaths, sabr_paths

from spxa.rates.sabr_pde import ArbitrageFreeSABR, SABRDensity

from spxa.rates.sabr import (
    SABR,
    SABRFit,
    alpha_from_atm,
    calibrate_sabr,
    sabr_lognormal_vol,
    sabr_normal_vol,
    z_over_chi,
)

from spxa.rates.swaptions import Swaption, cash_annuity, swaption

from spxa.rates.capfloor import CapFloor, Caplet, cap_floor

from spxa.rates.bonds import BondCashflow, FixedRateBond
from spxa.rates.bootstrap import (
    BootstrapResult,
    CurveSpec,
    GlobalFitResult,
    bootstrap_curve,
    fit_curves,
)
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
from spxa.rates.helpers import (
    BasisSwapHelper,
    CurveHelper,
    DepositHelper,
    FRAHelper,
    FutureHelper,
    SwapHelper,
    last_curve_date,
    quarterly_futures,
    BondHelper,
)
from spxa.rates.interpolation import (
    CubicSplineRT,
    CubicZero,
    FittedInterpolation,
    InterpolationScheme,
    LinearZero,
    LogLinearDiscount,
    MonotoneConvex,
    MonotoneCubicRT,
    PiecewiseCurve,
    discrete_forwards,
    hyman_slopes,
)
from spxa.rates.pricing import CashflowRow, cashflow_table, leg_annuity, leg_pv
from spxa.rates.instruments import FRA, Deposit
from spxa.rates.turns import ForwardSpread, SpreadAdjustedCurve, turn_of_year
from spxa.rates.meetings import MeetingCurveResult, flat_forward_curve, meeting_date_curve
from spxa.rates.parametric import (
    NelsonSiegelSvensson,
    NSSFit,
    fit_nss,
    fit_nss_zero_rates,
    nelson_siegel,
    nss_loadings,
)
from spxa.rates.curveset import CurveSet
from spxa.rates.diagnostics import (
    SchemeDiagnostics,
    bump_locality,
    compare_schemes,
    forward_roughness,
)
from spxa.rates.pca import CurvePCA, curve_pca
from spxa.rates.risk import (
    PnLExplain,
    node_gamma,
    pnl_explain,
    HedgeResult,
    hedge_notionals,
    par_hedge,
    KeyRateRisk,
    KeyRateShiftedCurve,
    key_rate_deltas,
    key_rate_hats,
    NodeRisk,
    QuoteRisk,
    curve_nodes,
    node_deltas,
    quote_deltas,
    quote_jacobian,
    with_node_values,
)
from spxa.rates.swaps import (
    BasisSwap,
    NotionalLike,
    Swap,
    basis_swap,
    linear_notional,
    ois_swap,
    step_notional,
    vanilla_swap,
)
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
    "NotionalLike",
    "linear_notional",
    "step_notional",
    # bonds
    "BondCashflow",
    "FixedRateBond",
    # pricing
    "CashflowRow",
    "cashflow_table",
    "leg_annuity",
    "leg_pv",
    # curve construction
    "PiecewiseCurve",
    "InterpolationScheme",
    "FittedInterpolation",
    "LogLinearDiscount",
    "LinearZero",
    "CubicSplineRT",
    "CubicZero",
    "MonotoneCubicRT",
    "MonotoneConvex",
    "hyman_slopes",
    "discrete_forwards",
    # calibration helpers
    "CurveHelper",
    "DepositHelper",
    "FRAHelper",
    "FutureHelper",
    "SwapHelper",
    "BasisSwapHelper",
    "last_curve_date",
    "quarterly_futures",
    "BondHelper",
    # bootstrapping
    "bootstrap_curve",
    "BootstrapResult",
    "fit_curves",
    "CurveSpec",
    "GlobalFitResult",
    # forward spreads
    "ForwardSpread",
    "SpreadAdjustedCurve",
    "turn_of_year",
    # meeting-date curves
    "flat_forward_curve",
    "meeting_date_curve",
    "MeetingCurveResult",
    # parametric curves
    "NelsonSiegelSvensson",
    "nelson_siegel",
    "nss_loadings",
    "fit_nss_zero_rates",
    "fit_nss",
    "NSSFit",
    # curve sets
    "CurveSet",
    # scheme diagnostics
    "compare_schemes",
    "SchemeDiagnostics",
    "bump_locality",
    "forward_roughness",
    # curve risk
    "NodeRisk",
    "QuoteRisk",
    "curve_nodes",
    "with_node_values",
    "node_deltas",
    "quote_jacobian",
    "quote_deltas",
    "KeyRateRisk",
    "KeyRateShiftedCurve",
    "key_rate_hats",
    "key_rate_deltas",
    "CurvePCA",
    "curve_pca",
    "HedgeResult",
    "hedge_notionals",
    "par_hedge",
    "PnLExplain",
    "node_gamma",
    "pnl_explain",
    "CapFloor",
    "Caplet",
    "cap_floor",
    "Swaption",
    "cash_annuity",
    "swaption",
    "SABR",
    "sabr_lognormal_vol",
    "sabr_normal_vol",
    "z_over_chi",
    "SABRFit",
    "alpha_from_atm",
    "calibrate_sabr",
    "ArbitrageFreeSABR",
    "SABRDensity",
    "SABRPaths",
    "sabr_paths",
    "CapletVolCurve",
    "strip_caplet_vols",
]
