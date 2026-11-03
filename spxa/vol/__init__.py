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
Implied distribution
    :class:`ImpliedDistribution`: density, cdf, digitals, moments, quantiles
    and samples of :math:`\log(S_T/F_T)` from a slice.
"""

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
from spxa.vol.density import ImpliedDistribution
from spxa.vol.essvi import ESSVI, ESSVISlice, anchored_theta, calibrate_essvi, essvi_calendar_free
from spxa.vol.ssvi import SSVI, ATMVarianceCurve, HestonLikePhi, PowerLawPhi
from spxa.vol.svi import SVIJumpWings, SVINatural, SVIRaw
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
    # wings
    "WingAsymptotics",
    "lee_moment_from_slope",
    "lee_slope_from_moment",
    "rogers_tehranchi_slope",
    "wing_asymptotics",
]
