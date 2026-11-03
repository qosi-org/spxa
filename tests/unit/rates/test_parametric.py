"""Nelson-Siegel and Svensson curves and fits."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from spxa.rates import ESTR, Deposit, ois_swap
from spxa.rates.helpers import DepositHelper, SwapHelper
from spxa.rates.interpolation import MonotoneConvex, PiecewiseCurve
from spxa.rates.parametric import NelsonSiegelSvensson, fit_nss, fit_nss_zero_rates, nelson_siegel

REF = dt.date(2024, 1, 15)
TRUE = NelsonSiegelSvensson(0.04, -0.02, 0.015, -0.01, 1.5, 8.0, reference_date=REF)
T = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30], float)


def _helpers(curve: object) -> list:
    hs: list = []
    for t in ("1M", "3M", "6M"):
        d = Deposit.from_tenor(REF, t, 0.0)
        hs.append(DepositHelper(d, quote=d.implied_rate(curve)))  # type: ignore[arg-type]
    for t in ("1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"):
        s = ois_swap(REF, t, 0.0, ESTR)
        hs.append(SwapHelper(s, quote=s.par_rate(curve)))  # type: ignore[arg-type]
    return hs


def test_closed_forms() -> None:
    t = np.linspace(0.01, 40, 400)
    h = 1e-6
    tz = lambda s: TRUE.zero(s) * s  # noqa: E731
    assert TRUE.instantaneous_forward(t) == pytest.approx(
        (tz(t + h) - tz(t - h)) / (2 * h), abs=1e-9
    )
    assert TRUE.zero(np.array([1e-14]))[0] == pytest.approx(0.02, abs=1e-14)  # beta0 + beta1
    assert TRUE.instantaneous_forward(0.0) == pytest.approx(0.02, abs=1e-15)
    assert TRUE.zero(np.array([1e5]))[0] == pytest.approx(0.04, abs=1e-5)
    assert TRUE.discount(5.0) == pytest.approx(
        np.exp(-5 * TRUE.zero(np.array([5.0]))[0]), rel=1e-15
    )
    ns = nelson_siegel(0.04, -0.02, 0.015, 2.0)
    assert not ns.is_svensson and set(ns.params) == {"beta0", "beta1", "beta2", "tau1"}
    with pytest.raises(ValueError):
        NelsonSiegelSvensson(0.04, 0, 0, 0.01, 1.0, None)
    with pytest.raises(ValueError):
        nelson_siegel(0.04, 0, 0, -1.0)


def test_zero_rate_fit_recovers_parameters() -> None:
    fit = fit_nss_zero_rates(T, TRUE.zero(T))
    assert fit.max_residual < 1e-12
    p, q = fit.curve.params, TRUE.params
    for k in q:
        assert p[k] == pytest.approx(q[k], rel=1e-6, abs=1e-9)
    ns_true = nelson_siegel(0.035, -0.015, 0.02, 2.5)
    ns = fit_nss_zero_rates(T, ns_true.zero(T), svensson=False)
    assert ns.curve.params == pytest.approx(ns_true.params, rel=1e-7)


def test_helper_fit_reprices_nss_market_exactly() -> None:
    fit = fit_nss(_helpers(TRUE), REF)
    assert fit.max_residual < 1e-12
    t = np.linspace(0.1, 30, 200)
    assert fit.curve.zero_rate(t) == pytest.approx(TRUE.zero_rate(t), abs=1e-9)


def test_helper_fit_to_non_parametric_market() -> None:
    market = PiecewiseCurve.from_zero_rates(
        [0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30],
        [0.039, 0.0385, 0.037, 0.034, 0.032, 0.0305, 0.030, 0.0302, 0.031, 0.0308, 0.029],
        MonotoneConvex(),
        reference_date=REF,
    )
    sv = fit_nss(_helpers(market), REF)
    ns = fit_nss(_helpers(market), REF, svensson=False)
    assert sv.rmse <= ns.rmse  # the extra hump can only help
    assert sv.max_residual < 1.5e-3  # a smooth fit, not an interpolation: within 15bp
    t = np.linspace(0, 30, 301)
    f = sv.curve.instantaneous_forward(t)
    g = market.instantaneous_forward(t)
    # smoother forwards than the curve that generated the quotes
    assert np.sum(np.diff(f, 2) ** 2) < 0.5 * np.sum(np.diff(g, 2) ** 2)


def test_validation() -> None:
    with pytest.raises(ValueError):
        fit_nss_zero_rates([1.0, 2.0], [0.01, 0.02])
    with pytest.raises(ValueError):
        fit_nss_zero_rates(T, TRUE.zero(T), weights=-np.ones_like(T))
