"""Bond-curve bootstrapping and fitted (NSS) bond curves."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from spxa.rates.bonds import FixedRateBond
from spxa.rates.bootstrap import bootstrap_curve
from spxa.rates.helpers import BondHelper
from spxa.rates.interpolation import MonotoneConvex
from spxa.rates.parametric import NelsonSiegelSvensson, fit_nss

TRADE = dt.date(2024, 3, 1)
TRUE = NelsonSiegelSvensson(0.042, 0.01, -0.012, 0.008, 1.2, 9.0, reference_date=TRADE)
# (years to maturity, coupon): an on-the-run style ladder issued at par-ish coupons
LADDER = [
    (1, 0.05),
    (2, 0.0475),
    (3, 0.0425),
    (5, 0.04),
    (7, 0.04125),
    (10, 0.0425),
    (20, 0.045),
    (30, 0.04375),
]


def _helpers(quote_type: str = "clean") -> list[BondHelper]:
    hs = []
    for years, cpn in LADDER:
        b = FixedRateBond(dt.date(2024 - 1, 2, 15), dt.date(2024 + years, 2, 15), cpn)
        settle = b.settlement_date(TRADE)
        clean = b.dirty_price_from_curve(TRUE, settle) - b.accrued_interest(settle)
        q = clean if quote_type == "clean" else b.yield_from_clean(clean, settle)
        hs.append(BondHelper(b, q, settle, quote_type=quote_type))
    return hs


def test_helper_quotes() -> None:
    h = _helpers()[3]
    assert h.residual({"discount": TRUE}) == pytest.approx(0.0, abs=1e-12)
    y = _helpers("yield")[3]
    assert y.bond.clean_price(y.quote, y.settle) == pytest.approx(h.quote, abs=1e-10)
    assert h.pillar == dt.date(2029, 2, 15)
    with pytest.raises(ValueError):
        BondHelper(h.bond, 100.0, h.settle, quote_type="dirty")


@pytest.mark.parametrize("quote_type", ["clean", "yield"])
def test_bootstrap_reprices_bonds(quote_type: str) -> None:
    res = bootstrap_curve(_helpers(quote_type), TRADE, MonotoneConvex())
    assert res.max_residual < (1e-10 if quote_type == "clean" else 1e-12)
    t = np.linspace(1, 30, 60)
    assert np.max(np.abs(res.curve.zero_rate(t) - TRUE.zero_rate(t))) < 2e-3


def test_fitted_bond_curve_recovers_svensson() -> None:
    fit = fit_nss(_helpers(), TRADE)
    assert fit.max_residual < 1e-9  # price units per 100 face
    t = np.linspace(0.5, 30, 60)
    assert fit.curve.zero_rate(t) == pytest.approx(TRUE.zero_rate(t), abs=1e-9)
