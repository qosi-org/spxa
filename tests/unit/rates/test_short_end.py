"""O/N and T/N deposits at the front of a bootstrapped curve."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from spxa.rates import Deposit
from spxa.rates.bootstrap import bootstrap_curve
from spxa.rates.helpers import DepositHelper

FRI = dt.date(2024, 3, 8)


def test_stub_dates() -> None:
    on, tn = Deposit.overnight(FRI, 0.039), Deposit.tom_next(FRI, 0.0391)
    sn = Deposit.spot_next(FRI, 0.0392)
    assert (on.start, on.end) == (FRI, dt.date(2024, 3, 11))  # over the weekend: 3 days
    assert (tn.start, tn.end) == (dt.date(2024, 3, 11), dt.date(2024, 3, 12))
    assert (sn.start, sn.end) == (dt.date(2024, 3, 12), dt.date(2024, 3, 13))
    assert tn.end == Deposit.from_tenor(FRI, "1W", 0.04).start  # T/N ends at spot
    # Easter 2024 (TARGET): Thursday 28 March trade, O/N to Tuesday 2 April
    assert Deposit.overnight(dt.date(2024, 3, 28), 0.04).end == dt.date(2024, 4, 2)
    assert on.accrual == pytest.approx(3 / 360, abs=1e-16)


def test_front_end_with_and_without_stubs() -> None:
    on, tn = Deposit.overnight(FRI, 0.0390), Deposit.tom_next(FRI, 0.0395)
    terms = [
        Deposit.from_tenor(FRI, t, r) for t, r in (("1W", 0.0398), ("1M", 0.0400), ("3M", 0.0395))
    ]
    hs = [DepositHelper(d) for d in (on, tn, *terms)]
    res = bootstrap_curve(hs, FRI)
    # one-day accruals amplify rounding in P by 1/tau = 360
    assert res.max_residual < 1e-13
    spot = tn.end
    expected = 1 / ((1 + on.accrual * on.rate) * (1 + tn.accrual * tn.rate))
    assert res.curve.discount(spot) == pytest.approx(expected, rel=1e-15, abs=2e-16)
    # without the stubs, [trade, spot] is extrapolated from the 1W node instead
    bare = bootstrap_curve([DepositHelper(d) for d in terms], FRI)
    assert bare.max_residual < 1e-14
    assert abs(bare.curve.discount(spot) - expected) > 1e-7
    # spot-starting instruments are unaffected: only forwards from spot are pinned
    t1, t2 = res.curve.time(terms[0].start), res.curve.time(terms[2].end)
    f_res = np.log(res.curve.discount(t1) / res.curve.discount(t2))
    f_bare = np.log(bare.curve.discount(t1) / bare.curve.discount(t2))
    assert f_res == pytest.approx(f_bare, abs=1e-14)
