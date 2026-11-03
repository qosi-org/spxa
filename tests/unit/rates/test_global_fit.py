"""Simultaneous multi-curve calibration by Newton's method."""

from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pytest

from spxa.rates import EURIBOR6M, ESTR, FRA, basis_swap, ois_swap, vanilla_swap
from spxa.rates.bootstrap import CurveSpec, bootstrap_curve, fit_curves
from spxa.rates.helpers import BasisSwapHelper, FRAHelper, SwapHelper
from spxa.rates.interpolation import LogLinearDiscount, MonotoneConvex, PiecewiseCurve
from tests.unit.rates.test_bootstrap import MARKET, REF, _estr_helpers

PROJ = PiecewiseCurve.from_zero_rates(
    [0.5, 1, 2, 5, 10, 20],
    [0.041, 0.040, 0.038, 0.036, 0.0365, 0.036],
    MonotoneConvex(),
    reference_date=REF,
)
TRUE = {"discount": MARKET, "EUR-6M": PROJ}


def test_single_curve_matches_bootstrap() -> None:
    hs = _estr_helpers(MARKET)
    seq = bootstrap_curve(hs, REF)
    glob = fit_curves([CurveSpec("discount", hs)], REF)
    assert glob.max_residual < 1e-14
    assert glob.iterations <= 6
    c = glob.curves["discount"]
    assert c.node_discounts == pytest.approx(seq.curve.node_discounts, rel=1e-13)  # type: ignore[attr-defined]


def test_non_local_scheme_faster_than_passes() -> None:
    hs = _estr_helpers(MARKET)
    t0 = time.perf_counter()
    seq = bootstrap_curve(hs, REF, MonotoneConvex())
    t_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    glob = fit_curves([CurveSpec("discount", hs, MonotoneConvex())], REF)
    t_glob = time.perf_counter() - t0
    assert glob.max_residual < 1e-14
    got = glob.curves["discount"].node_discounts  # type: ignore[attr-defined]
    assert got == pytest.approx(seq.curve.node_discounts, rel=1e-12)
    assert t_glob < t_seq


def _projection_helpers() -> list:
    hs: list = []
    for st in ("1M", "3M"):
        fra = FRA.from_index(EURIBOR6M, REF, st, 0.0)
        hs.append(FRAHelper(fra, quote=fra.forward_rate(PROJ), curve="EUR-6M"))
    for t in ("2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y"):
        h = SwapHelper(vanilla_swap(REF, t, 0.0, EURIBOR6M), forward="EUR-6M")
        h.quote = h.implied_quote(TRUE)
        hs.append(h)
    return hs


def test_joint_ois_and_projection() -> None:
    ois_h, proj_h = _estr_helpers(MARKET), _projection_helpers()
    res = fit_curves(
        [CurveSpec("discount", ois_h), CurveSpec("EUR-6M", proj_h, MonotoneConvex())], REF
    )
    assert res.max_residual < 1e-14
    assert len(res.nodes) == len(ois_h) + len(proj_h)
    # the same answer as bootstrapping OIS first and the projection curve on top
    ois = bootstrap_curve(ois_h, REF).curve
    proj = bootstrap_curve(proj_h, REF, MonotoneConvex(), name="EUR-6M", known={"discount": ois})
    got = res.curves["EUR-6M"].node_discounts  # type: ignore[attr-defined]
    assert got == pytest.approx(proj.curve.node_discounts, rel=1e-12)
    # block structure: OIS helpers do not read the projection curve
    n_ois = len(ois_h)
    assert np.all(res.jacobian[:n_ois, n_ois:] == 0.0)
    assert np.any(res.jacobian[n_ois:, :n_ois] != 0.0)


def test_coupled_through_basis_swaps() -> None:
    """Long OIS points come only from ESTR/6M basis swaps: the curves must be solved together."""
    ois_h = [h for h in _estr_helpers(MARKET) if h.pillar < dt.date(2032, 1, 1)]
    long_basis = []
    for t in ("10Y", "15Y", "20Y"):
        bs = basis_swap(REF, t, ESTR, EURIBOR6M)
        par = bs.par_spread(MARKET, MARKET, PROJ)
        long_basis.append(BasisSwapHelper(bs, par, first="discount", second="EUR-6M"))
    proj_h = _projection_helpers()
    res = fit_curves([CurveSpec("discount", ois_h + long_basis), CurveSpec("EUR-6M", proj_h)], REF)
    assert res.max_residual < 1e-13
    for h in ois_h + long_basis + proj_h:
        assert h.residual(res.curves) == pytest.approx(0.0, abs=1e-13)
    with pytest.raises(ValueError):
        bootstrap_curve(ois_h + long_basis, REF)  # basis swaps also need EUR-6M


def test_validation() -> None:
    hs = _estr_helpers(MARKET)
    with pytest.raises(ValueError, match="distinct"):
        fit_curves([CurveSpec("discount", hs), CurveSpec("discount", hs)], REF)
    with pytest.raises(ValueError, match="known and fitted"):
        fit_curves([CurveSpec("discount", hs)], REF, known={"discount": MARKET})
    with pytest.raises(ValueError, match="no helpers"):
        fit_curves([CurveSpec("discount", [])], REF)
    with pytest.raises(ValueError, match="unknown curves"):
        fit_curves([CurveSpec("EUR-6M", _projection_helpers())], REF)
    with pytest.raises(ValueError, match="share a pillar"):
        fit_curves([CurveSpec("discount", hs + [hs[0]])], REF)
    assert isinstance(CurveSpec("x", hs).scheme, LogLinearDiscount)
    ois = ois_swap(REF, "1Y", 0.03, ESTR)
    assert SwapHelper(ois).pillar > REF
