"""Bermudan swaption contracts."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from spxa.market import FlatCurve
from spxa.rates.bermudan import BermudanSwaption, bermudan_from_times, bermudan_swaption
from spxa.rates.model_cashflows import swap_cashflows
from spxa.rates.pricing import leg_pv
from spxa.rates.swaptions import swaption

REF = dt.date(2024, 3, 15)
DISC = FlatCurve(0.03, reference_date=REF)
PROJ = FlatCurve(0.034, reference_date=REF)
K = 0.035


def _ber(lockout: int = 0) -> BermudanSwaption:
    # 1Y x 5Y payer: the first exercise is the 1Y x 5Y European
    sw = swaption(REF, "1Y", "5Y", K, notional=1e4)
    return bermudan_swaption(sw.swap, DISC, PROJ, lockout=lockout)


def test_coterminal_underlyings() -> None:
    sw = swaption(REF, "1Y", "5Y", K, notional=1e4).swap
    ber = _ber()
    assert ber.n_exercise == 5
    for k in range(5):
        start = sw.fixed[k].start
        fixed = [c for c in sw.fixed if c.start >= start]
        flt = [c for c in sw.floating if c.start >= start]
        pv = leg_pv(flt, DISC, PROJ) - sum(
            c.amount() * DISC.discount(c.payment_date) for c in fixed
        )
        assert ber.underlying_pv(k, DISC) == pytest.approx(pv, rel=1e-12)
        assert ber.exercise_times[k] == pytest.approx(DISC.time(flt[0].fixing), abs=1e-14)
        assert ber.underlyings[k].times[0] > ber.exercise_times[k]
        # close to the matching spot-starting European (schedules differ by a day or so)
        eu = swaption(REF, f"{1 + k}Y", f"{5 - k}Y", K, notional=1e4)
        assert ber.exercise_times[k] == pytest.approx(DISC.time(eu.expiry), abs=4 / 365)
    ref = swap_cashflows(sw, DISC, PROJ)
    np.testing.assert_allclose(ber.underlyings[0].amounts, ref.amounts, rtol=1e-14)
    # lockout drops the first exercise; restrict keeps chosen dates
    assert _ber(1).n_exercise == 4
    assert np.array_equal(ber.restrict([4, 2]).exercise_times, ber.exercise_times[[2, 4]])


def test_from_times() -> None:
    ber = bermudan_from_times([1.0, 2.0, 3.0], 5.0, 0.03, notional=100.0)
    P = DISC.discount
    for k, t in enumerate([1.0, 2.0, 3.0]):
        pay = np.arange(t + 1, 6.0)
        fwd = 100 * (P(t) - P(5.0)) - 100 * 0.03 * float(np.sum(P(pay)))
        assert ber.underlying_pv(k, DISC) == pytest.approx(fwd, rel=1e-13)
    rec = bermudan_from_times([2.0], 5.0, 0.03, payer=False, frequency=2)
    assert rec.underlyings[0].times.size == 7 and rec.underlyings[0].amounts[0] == -1.0
    with pytest.raises(ValueError):
        bermudan_from_times([1.3], 5.0, 0.03)
    with pytest.raises(ValueError):
        BermudanSwaption(np.array([2.0, 1.0]), ber.underlyings[:2])
