r"""
spxa.vol — implied and local volatility surfaces, stochastic-local
volatility and model-free volatility tools.

Coordinates
    :func:`log_moneyness`, :func:`total_variance`,
    :func:`implied_vol_from_total_variance`,
    :func:`normalised_call_from_total_variance`,
    :func:`total_variance_from_price`.
SVI
    :class:`SVIRaw`, :class:`SVINatural`, :class:`SVIJumpWings`.
SSVI
    :class:`SSVI` with :class:`HestonLikePhi`, :class:`PowerLawPhi` and
    :class:`ATMVarianceCurve`; Theorems 4.1 and 4.2 of Gatheral–Jacquier.
eSSVI
    :class:`ESSVISlice`, :class:`ESSVI`, :func:`essvi_calendar_free` and
    :func:`calibrate_essvi`
    (Hendriks–Martini; arbitrage-free interpolation of Corbetta et al.).
Calibration
    :func:`calibrate_svi` (quasi-explicit, Zeliade), :func:`svi_inner_fit`,
    :func:`calibrate_svi_butterfly_free`, :func:`calibrate_svi_surface`,
    :func:`calibrate_ssvi`,
    :func:`isotonic_increasing`.
No-arbitrage diagnostics
    :func:`durrleman_g`, :func:`risk_neutral_density`,
    :func:`check_butterfly`, :func:`check_calendar`.
Wings
    :func:`wing_asymptotics`, :class:`WingAsymptotics`,
    :func:`lee_moment_from_slope`, :func:`lee_slope_from_moment`,
    :func:`rogers_tehranchi_slope`.
Interpolation in time
    :class:`PriceInterpolatedSurface` (Gatheral–Jacquier price interpolation),
    :class:`VarianceInterpolatedSurface` (fixed-k or ATM-normalised variance).
Surfaces
    :class:`VolSurface`: implied volatilities and Black prices in strike
    space from a total-variance model and a forward curve.
    :func:`build_vol_surface` runs quotes -> parity forwards -> OTM
    total variances -> SVI/eSSVI/SSVI fit -> surface
    (:class:`SurfaceBuild`, :class:`ForwardTermStructure`,
    :func:`chain_total_variance`); :meth:`SurfaceBuild.quality` gives a
    :class:`SurfaceQuality` report (fit errors, butterfly, calendar).
Deltas
    :func:`black_delta` (forward/spot, premium-adjusted),
    :func:`strike_from_delta` and :func:`atm_dns_strike`, also as
    :meth:`VolSurface.delta` and :meth:`VolSurface.strike_from_delta`.
Serialisation
    :func:`to_dict`/:func:`from_dict` and :func:`dumps`/:func:`loads`:
    bit-exact JSON round trips of surfaces, models, forwards and curves.
Bumps
    :meth:`VolSurface.bumped` with :func:`parallel_bump`, :func:`skew_bump`
    and :func:`expiry_buckets`; :func:`bucketed_vega`.
Spline smoothing
    :func:`fengler_smooth` (:class:`FenglerSmile`): Fengler's convex,
    monotone natural cubic call-price spline; :func:`fengler_surface`
    (:class:`FenglerSurface`) adds the calendar condition across expiries.
Arbitrage repair
    :func:`call_arbitrage_constraints` and :func:`repair_call_prices`:
    minimal weighted l1 repair of a call grid by linear programming
    (Cohen–Reisinger–Wang), with optional free moves inside the spread.
Implied distribution
    :class:`ImpliedDistribution`: density, cdf, digitals, moments, quantiles
    and samples of :math:`\log(S_T/F_T)` from a slice.
Local volatility
    :class:`LocalVolSurface` and :func:`dupire_local_variance`: Dupire's formula in total-variance form;
    :class:`LocalVolGrid` tabulates it for fast lookups; :func:`bbf_implied_vol`
    gives the short-expiry implied vol as a harmonic mean.
Andreasen–Huge
    :class:`AndreasenHugeSurface`: one implicit Dupire step per expiry, arbitrage-free by construction;
    :func:`calibrate_andreasen_huge` fits the levels expiry by expiry.
Local-vol Monte Carlo
    :func:`local_vol_paths` (log-Euler) and :class:`LocalVolPaths`.
"""

from spxa.vol.local_vol_mc import LocalVolPaths, local_vol_paths

from spxa.vol.andreasen_huge import (
    AndreasenHugeFit,
    AndreasenHugeSurface,
    calibrate_andreasen_huge,
    andreasen_huge_grid,
    piecewise_constant_vol,
)

from spxa.vol.local_vol import (
    LocalVolGrid,
    LocalVolSurface,
    bbf_implied_vol,
    dupire_local_variance,
)

from spxa.vol.arbitrage import (
    ButterflyCheck,
    CalendarCheck,
    check_butterfly,
    check_calendar,
    durrleman_g,
    risk_neutral_density,
)
from spxa.vol.calibration import (
    SSVIFit,
    SVIFit,
    SVISurfaceFit,
    calibrate_ssvi,
    calibrate_svi,
    calibrate_svi_butterfly_free,
    calibrate_svi_surface,
    isotonic_increasing,
    svi_inner_fit,
)
from spxa.vol.conventions import (
    implied_vol_from_total_variance,
    log_moneyness,
    normalised_call_from_total_variance,
    strike_from_log_moneyness,
    total_variance,
    total_variance_from_price,
)
from spxa.vol.bumps import (
    BumpedModel,
    bucketed_vega,
    expiry_buckets,
    parallel_bump,
    skew_bump,
)
from spxa.vol.delta import atm_dns_strike, black_delta, strike_from_delta
from spxa.vol.serialization import dumps, from_dict, loads, to_dict
from spxa.vol.fengler import (
    FenglerSmile,
    FenglerSurface,
    fengler_smooth,
    fengler_surface,
    spline_eval_matrix,
    spline_matrices,
)
from spxa.vol.density import ImpliedDistribution
from spxa.vol.interpolation import PriceInterpolatedSurface, VarianceInterpolatedSurface
from spxa.vol.essvi import ESSVI, ESSVISlice, anchored_theta, calibrate_essvi, essvi_calendar_free
from spxa.vol.ssvi import SSVI, ATMVarianceCurve, HestonLikePhi, PowerLawPhi
from spxa.vol.svi import SVIJumpWings, SVINatural, SVIRaw
from spxa.vol.build import (
    ForwardTermStructure,
    SurfaceBuild,
    SurfaceQuality,
    ExpiryQuality,
    build_vol_surface,
    chain_total_variance,
)
from spxa.vol.repair import (
    ArbitrageConstraints,
    RepairResult,
    call_arbitrage_constraints,
    repair_call_prices,
)
from spxa.vol.surface import ForwardLike, TotalVarianceModel, VolSurface
from spxa.vol.wings import (
    WingAsymptotics,
    lee_moment_from_slope,
    lee_slope_from_moment,
    rogers_tehranchi_slope,
    wing_asymptotics,
)

__all__ = [
    # coordinates
    "implied_vol_from_total_variance",
    "log_moneyness",
    "normalised_call_from_total_variance",
    "strike_from_log_moneyness",
    "total_variance",
    "total_variance_from_price",
    # SVI
    "SVIJumpWings",
    "SVINatural",
    "SVIRaw",
    # SSVI
    "SSVI",
    "ATMVarianceCurve",
    "HestonLikePhi",
    "PowerLawPhi",
    # eSSVI
    "ESSVI",
    "ESSVISlice",
    "anchored_theta",
    "calibrate_essvi",
    "essvi_calendar_free",
    # calibration
    "SSVIFit",
    "SVIFit",
    "calibrate_ssvi",
    "calibrate_svi",
    "isotonic_increasing",
    "calibrate_svi_butterfly_free",
    "SVISurfaceFit",
    "calibrate_svi_surface",
    "svi_inner_fit",
    # arbitrage
    "ButterflyCheck",
    "CalendarCheck",
    "check_butterfly",
    "check_calendar",
    "durrleman_g",
    "risk_neutral_density",
    # implied distribution
    "ImpliedDistribution",
    # interpolation
    "PriceInterpolatedSurface",
    "VarianceInterpolatedSurface",
    # wings
    "WingAsymptotics",
    "lee_moment_from_slope",
    "lee_slope_from_moment",
    "rogers_tehranchi_slope",
    "wing_asymptotics",
    # surfaces
    "VolSurface",
    "TotalVarianceModel",
    "ForwardLike",
    "build_vol_surface",
    "SurfaceBuild",
    "SurfaceQuality",
    "ExpiryQuality",
    "ForwardTermStructure",
    "chain_total_variance",
    # arbitrage repair
    "call_arbitrage_constraints",
    "repair_call_prices",
    "ArbitrageConstraints",
    "RepairResult",
    # deltas
    "black_delta",
    "strike_from_delta",
    "atm_dns_strike",
    # spline smoothing
    "fengler_smooth",
    "FenglerSmile",
    "spline_matrices",
    "fengler_surface",
    "FenglerSurface",
    "spline_eval_matrix",
    # bumps
    "BumpedModel",
    "parallel_bump",
    "skew_bump",
    "expiry_buckets",
    "bucketed_vega",
    # serialisation
    "to_dict",
    "from_dict",
    "dumps",
    "loads",
    "LocalVolGrid",
    "LocalVolSurface",
    "dupire_local_variance",
    "bbf_implied_vol",
    "AndreasenHugeSurface",
    "andreasen_huge_grid",
    "piecewise_constant_vol",
    "AndreasenHugeFit",
    "calibrate_andreasen_huge",
    "LocalVolPaths",
    "local_vol_paths",
]
