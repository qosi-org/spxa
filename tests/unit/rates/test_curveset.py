"""Named curve sets with discount and projection roles."""

from __future__ import annotations

import datetime as dt

import pytest

from spxa.market.curves import FlatCurve
from spxa.rates import ESTR, EURIBOR3M, EURIBOR6M, basis_swap, ois_swap, vanilla_swap
from spxa.rates.bootstrap import bootstrap_curve
from spxa.rates.curveset import CurveSet
from spxa.rates.helpers import SwapHelper

REF = dt.date(2024, 1, 15)
OIS = FlatCurve(0.03, reference_date=REF)
E6 = FlatCurve(0.034, reference_date=REF)
E3 = FlatCurve(0.0325, reference_date=REF)
CS = CurveSet(
    {"ESTR": OIS, "EUR-6M": E6, "EUR-3M": E3},
    "ESTR",
    {"EURIBOR6M": "EUR-6M", "EURIBOR3M": "EUR-3M"},
)


def test_roles_and_mapping() -> None:
    assert CS.discount_curve is OIS and CS["EUR-6M"] is E6 and len(CS) == 3
    assert CS.projection(EURIBOR6M) is E6 and CS.projection("EURIBOR3M") is E3
    assert CS.projection(ESTR) is OIS  # same name as the index
    with pytest.raises(KeyError):
        CS.projection("SOFR")
    with pytest.raises(KeyError):
        CurveSet({"a": OIS}, "b")
    with pytest.raises(KeyError):
        CurveSet({"a": OIS}, "a", {"EURIBOR6M": "missing"})
    assert "ESTR" in repr(CS)


def test_pv_dispatches_by_leg_index() -> None:
    ois = ois_swap(REF, "5Y", 0.031, ESTR)
    irs = vanilla_swap(REF, "10Y", 0.033, EURIBOR6M)
    assert CS.pv(ois) == pytest.approx(ois.pv(OIS, OIS), abs=1e-15)
    assert CS.pv(irs) == pytest.approx(irs.pv(OIS, E6), abs=1e-15)
    bs = basis_swap(REF, "5Y", EURIBOR3M, EURIBOR6M, spread=0.001)
    assert CS.pv(bs) == pytest.approx(bs.pv(OIS, E3, E6), abs=1e-15)
    # the wrong projection would give a different answer
    assert abs(irs.pv(OIS, OIS) - CS.pv(irs)) > 1e-3
    with pytest.raises(TypeError):
        CS.pv(object())


def test_bumps_and_edits_are_copies() -> None:
    irs = vanilla_swap(REF, "10Y", 0.033, EURIBOR6M)
    up = CS.bumped(1e-4, "EUR-6M")
    assert up["ESTR"] is OIS and CS["EUR-6M"] is E6
    assert up["EUR-6M"].zero_rate(5.0) == pytest.approx(0.0341, abs=1e-14)
    assert up.pv(irs) > CS.pv(irs)  # payer gains when projected rates rise
    allup = CS.bumped(1e-4)
    assert all(
        allup[n].zero_rate(3.0) == pytest.approx(CS[n].zero_rate(3.0) + 1e-4, abs=1e-14) for n in CS
    )
    with pytest.raises(KeyError):
        CS.bumped(1e-4, ["nope"])
    new = CS.with_curve("EUR-6M", OIS)
    assert new.pv(irs) == pytest.approx(irs.pv(OIS, OIS), abs=1e-15) and CS["EUR-6M"] is E6


def test_from_fit_and_as_helper_curves() -> None:
    hs = []
    for t in ("2Y", "5Y", "10Y"):
        s = ois_swap(REF, t, 0.0, ESTR)
        hs.append(SwapHelper(s, quote=s.par_rate(OIS), discount="ESTR"))
    res = bootstrap_curve(hs, REF, name="ESTR")
    cs = CurveSet.from_fit(res, "ESTR")
    # a CurveSet is a mapping of named curves, so helpers read it directly
    assert max(abs(h.residual(cs)) for h in hs) < 1e-15
    assert cs.pv(ois_swap(REF, "5Y", hs[1].quote, ESTR)) == pytest.approx(0.0, abs=1e-15)
