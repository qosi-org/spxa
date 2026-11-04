"""Swaps as discount-curve cashflows."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.curves import FlatCurve
from spxa.rates.indices import EURIBOR6M, SOFR
from spxa.rates.model_cashflows import swap_cashflows
from spxa.rates.swaps import ois_swap, vanilla_swap
from tests.unit.rates.test_capfloor import DISC, PROJ, REF


def test_reproduces_the_swap_value() -> None:
    for payer in (True, False):
        sw = vanilla_swap(
            REF, "7Y", 0.031, EURIBOR6M, notional=1e6, payer=payer, forward_start="2Y"
        )
        cf = swap_cashflows(sw, DISC, PROJ)
        assert cf.pv(DISC) == pytest.approx(sw.pv(DISC, PROJ), abs=1e-8)
        assert np.all(np.diff(cf.times) > 0)
    sw = vanilla_swap(REF, "5Y", 0.03, EURIBOR6M, spread=0.001, forward_start="1Y")
    assert swap_cashflows(sw, DISC, PROJ).pv(DISC) == pytest.approx(sw.pv(DISC, PROJ), abs=1e-14)


def test_single_curve_telescopes() -> None:
    # coupon dates as accrual periods and one curve: the float leg is N (P(t0) - P(tn))
    sw = vanilla_swap(REF, "5Y", 0.03, EURIBOR6M, forward_start="1Y", use_index_dates=False)
    cf = swap_cashflows(sw, DISC)
    t0 = float(DISC.time(sw.start))
    assert cf.times[0] == t0 and cf.amounts[0] == pytest.approx(1.0, abs=1e-14)
    fixed = {float(DISC.time(c.payment_date)): c.amount() for c in sw.fixed}
    for t, a in zip(cf.times[1:-1], cf.amounts[1:-1]):
        assert a == pytest.approx(-fixed[t], abs=1e-12)
    assert cf.amounts[-1] == pytest.approx(-1.0 - fixed[cf.times[-1]], abs=1e-12)


def test_ois_and_model_value() -> None:
    sw = ois_swap(REF, "3Y", 0.029, SOFR, forward_start="1Y")
    disc = FlatCurve(0.03, reference_date=REF)
    cf = swap_cashflows(sw, disc)
    assert cf.pv(disc) == pytest.approx(sw.pv(disc, disc), abs=1e-14)

    class Det:
        """Deterministic model: P(t, T) = P(0, T)/P(0, t)."""

        def discount_bond(self, t, T, x):  # type: ignore[no-untyped-def]
            return disc.discount(T) / disc.discount(t) + 0.0 * x

    v = cf.value(Det(), 0.5, np.array([0.0, 0.0]))
    assert v == pytest.approx(np.full(2, cf.pv(disc) / disc.discount(0.5)), rel=1e-13)
