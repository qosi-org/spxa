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
    :class:`ESSVISlice`, :class:`ESSVI` and :func:`essvi_calendar_free`
    (Hendriks–Martini; arbitrage-free interpolation of Corbetta et al.).
Calibration
    :func:`calibrate_svi` (quasi-explicit, Zeliade), :func:`svi_inner_fit`,
    :func:`calibrate_svi_butterfly_free`, :func:`calibrate_ssvi`,
    :func:`isotonic_increasing`.
No-arbitrage diagnostics
    :func:`durrleman_g`, :func:`risk_neutral_density`,
    :func:`check_butterfly`, :func:`check_calendar`.
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
    calibrate_ssvi,
    calibrate_svi,
    calibrate_svi_butterfly_free,
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
from spxa.vol.essvi import ESSVI, ESSVISlice, essvi_calendar_free
from spxa.vol.ssvi import SSVI, ATMVarianceCurve, HestonLikePhi, PowerLawPhi
from spxa.vol.svi import SVIJumpWings, SVINatural, SVIRaw

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
    "essvi_calendar_free",
    # calibration
    "SSVIFit",
    "SVIFit",
    "calibrate_ssvi",
    "calibrate_svi",
    "isotonic_increasing",
    "calibrate_svi_butterfly_free",
    "svi_inner_fit",
    # arbitrage
    "ButterflyCheck",
    "CalendarCheck",
    "check_butterfly",
    "check_calendar",
    "durrleman_g",
    "risk_neutral_density",
]
