"""Curve calibration helpers."""

from __future__ import annotations

import datetime as dt

import pytest

from spxa.market.curves import FlatCurve
from spxa.rates import (
    ESTR,
    EURIBOR3M,
    EURIBOR6M,
    FRA,
    SOFR,
    Deposit,
    OvernightFuture,
    RateFuture,
    basis_swap,
    imm_date,
    ois_swap,
    vanilla_swap,
)
from spxa.rates.helpers import (
    BasisSwapHelper,
    DepositHelper,
    FRAHelper,
    FutureHelper,
    SwapHelper,
    last_curve_date,
)

REF = dt.date(2024, 1, 15)
OIS = FlatCurve(0.030, reference_date=REF)
E6 = FlatCurve(0.034, reference_date=REF)
E3 = FlatCurve(0.033, reference_date=REF)
CURVES = {"discount": OIS, "EUR-6M": E6, "EUR-3M": E3}


def test_deposit_and_fra() -> None:
    dep = Deposit.from_tenor(REF, "6M", 0.031)
    h = DepositHelper(dep)
    assert h.quote == 0.031 and h.pillar == dep.end and h.curves == ("discount",)
    assert h.implied_quote(CURVES) == pytest.approx(dep.implied_rate(OIS), rel=1e-15)
    h2 = DepositHelper(dep, quote=dep.implied_rate(OIS))
    assert h2.residual(CURVES) == pytest.approx(0.0, abs=1e-16)
    fra = FRA.from_index(EURIBOR6M, REF, "3M", 0.035)
    fh = FRAHelper(fra, curve="EUR-6M")
    assert fh.quote == 0.035 and fh.pillar == fra.end
    assert fh.implied_quote(CURVES) == pytest.approx(fra.forward_rate(E6), rel=1e-15)
    with pytest.raises(KeyError, match="EUR-6M"):
        fh.implied_quote({"discount": OIS})


def test_futures() -> None:
    imm = imm_date(2024, 6)
    rf = RateFuture.from_imm(EURIBOR3M, imm, 96.5)
    h = FutureHelper(rf, curve="EUR-3M", a=0.03, sigma=0.01)
    assert h.quote == 96.5 and h.pillar == rf.end
    assert h.implied_quote(CURVES) == pytest.approx(rf.implied_price(E3, 0.03, 0.01), rel=1e-15)
    of = OvernightFuture.from_imm(imm, 96.0)
    oh = FutureHelper(of)
    assert oh.pillar == of.end == imm_date(2024, 9)
    assert oh.residual(CURVES) == pytest.approx(of.implied_price(OIS) - 96.0, rel=1e-14)


def test_swaps_and_pillars() -> None:
    ois = ois_swap(REF, "5Y", 0.03, ESTR)
    h = SwapHelper(ois)
    assert h.quote == 0.03 and h.curves == ("discount",)
    assert h.implied_quote(CURVES) == pytest.approx(ois.par_rate(OIS), rel=1e-15)
    assert h.pillar == ois.maturity  # payment lag beyond the last accrual end
    irs = vanilla_swap(REF, "10Y", 0.034, EURIBOR6M)
    sh = SwapHelper(irs, discount="discount", forward="EUR-6M")
    assert sh.curves == ("discount", "EUR-6M")
    assert sh.implied_quote(CURVES) == pytest.approx(irs.par_rate(OIS, E6), rel=1e-15)
    # index periods can end after the last payment (index dates)
    last = irs.floating[-1]
    idx_end = EURIBOR6M.maturity_date(EURIBOR6M.value_date(last.fixing))
    assert sh.pillar == max(irs.maturity, idx_end) == last_curve_date(irs.floating)
    assert SwapHelper(irs, forward="discount").curves == ("discount",)


def test_basis_swap() -> None:
    bs0 = basis_swap(REF, "5Y", EURIBOR3M, EURIBOR6M)
    par = bs0.par_spread(OIS, E3, E6)
    bs = basis_swap(REF, "5Y", EURIBOR3M, EURIBOR6M, spread=par)
    h = BasisSwapHelper(bs, par, first="EUR-3M", second="EUR-6M")
    assert h.curves == ("discount", "EUR-3M", "EUR-6M")
    assert h.implied_quote(CURVES) == pytest.approx(par, abs=1e-15)
    assert h.residual(CURVES) == pytest.approx(0.0, abs=1e-15)
    off = BasisSwapHelper(bs, par + 0.001, first="EUR-3M", second="EUR-6M")
    assert off.residual(CURVES) == pytest.approx(-0.001, abs=1e-15)
    plain = BasisSwapHelper(bs0, par, first="EUR-3M", second="EUR-6M")
    assert plain.leg_spread == 0.0 and h.leg_spread == par
    assert plain.implied_quote(CURVES) == pytest.approx(par, abs=1e-15)
    assert h.pillar >= bs.maturity


def test_sofr_swap_quote_default() -> None:
    s = ois_swap(REF, "2Y", 0.041, SOFR)
    assert SwapHelper(s).quote == 0.041
    assert SwapHelper(s, quote=0.05).quote == 0.05
