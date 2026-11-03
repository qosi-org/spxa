"""Meeting-date (policy step) curves."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from spxa.rates import SOFR
from spxa.rates.bootstrap import bootstrap_curve
from spxa.rates.futures import OvernightFuture
from spxa.rates.helpers import FutureHelper
from spxa.rates.meetings import flat_forward_curve, meeting_date_curve

REF = dt.date(2024, 1, 15)
# first day of each new rate after the 2024 FOMC meetings
STEPS = [
    dt.date(2024, m, d)
    for m, d in [(2, 1), (3, 21), (5, 2), (6, 13), (8, 1), (9, 19), (11, 8), (12, 19)]
]
END = dt.date(2025, 1, 1)
START = dt.date(2024, 1, 16)  # 15 Jan 2024 is a SOFR holiday (MLK day), so forwards start on the 16th
RATES = [0.0531, 0.0531, 0.0518, 0.0506, 0.0481, 0.0456, 0.0431, 0.0418, 0.0406]
TRUE = flat_forward_curve(REF, [*STEPS, END], RATES)


def _fwd(start: dt.date, end: dt.date, method: str = "compounded") -> FutureHelper:
    probe = OvernightFuture(start, end, 95.0, SOFR, method)
    price = probe.implied_price(TRUE)
    return FutureHelper(OvernightFuture(start, end, price, SOFR, method))


def test_flat_forward_curve() -> None:
    t = TRUE.time([*STEPS, END])
    x = -np.log(TRUE.discount(t))
    assert np.diff(np.concatenate([[0.0], x])) / np.diff(
        np.concatenate([[0.0], t])
    ) == pytest.approx(RATES, abs=1e-15)
    eps = 1e-9
    for s, r0, r1 in zip(STEPS, RATES, RATES[1:]):
        ts = float(TRUE.time(s))
        assert TRUE.instantaneous_forward(ts - eps) == pytest.approx(r0, abs=1e-14)
        assert TRUE.instantaneous_forward(ts + eps) == pytest.approx(r1, abs=1e-14)
    with pytest.raises(ValueError):
        flat_forward_curve(REF, [STEPS[1], STEPS[0]], [0.01, 0.02])


def test_meeting_dated_forwards_fit_exactly() -> None:
    bounds = [START, *STEPS, END]
    hs = [_fwd(a, b) for a, b in zip(bounds, bounds[1:])]
    res = meeting_date_curve(hs, REF, STEPS)
    assert res.dates == [*STEPS, END]
    assert res.max_residual < 1e-11  # price units: 1e-11 = 1e-13 in rate
    assert res.rates == pytest.approx(RATES, abs=1e-12)
    days = [REF + dt.timedelta(d) for d in range(1, 352)]
    assert res.curve.discount(days) == pytest.approx(TRUE.discount(days), rel=1e-13)


def test_monthly_futures_least_squares() -> None:
    hs = [_fwd(START, STEPS[0])]  # spot-to-first-meeting stub fixes the first segment
    hs += [
        _fwd(dt.date(2024, mo, 1), dt.date(2024 + mo // 12, mo % 12 + 1, 1), "averaged")
        for mo in range(2, 13)
    ]
    res = meeting_date_curve(hs, REF, STEPS)
    assert len(hs) == 12 and len(res.rates) == 9  # over-determined
    assert res.max_residual < 1e-11
    assert res.rates == pytest.approx(RATES, abs=1e-12)

    # noisy prices: the fit no longer reprices, but solves the normal equations
    rng = np.random.default_rng(3)
    noisy = [FutureHelper(h.future, h.quote + 0.005 * rng.standard_normal()) for h in hs]
    fit = meeting_date_curve(noisy, REF, STEPS)
    assert 1e-4 < fit.max_residual < 0.02
    J = np.empty((12, 9))
    for j in range(9):
        e = np.zeros(9)
        e[j] = 1e-6
        up = flat_forward_curve(REF, fit.dates, fit.rates + e)
        dn = flat_forward_curve(REF, fit.dates, fit.rates - e)
        J[:, j] = [
            (h.implied_quote({"discount": up}) - h.implied_quote({"discount": dn})) / 2e-6
            for h in noisy
        ]
    assert np.max(np.abs(J.T @ fit.residuals)) < 1e-8 * np.max(np.abs(J))
    assert np.max(np.abs(fit.rates - RATES)) < 5e-4


def test_step_is_sharp_where_a_node_bootstrap_smears_it() -> None:
    hs = [_fwd(START, STEPS[0])]
    hs += [
        _fwd(dt.date(2024, mo, 1), dt.date(2024 + mo // 12, mo % 12 + 1, 1), "averaged")
        for mo in range(2, 13)
    ]
    step = meeting_date_curve(hs, REF, STEPS).curve
    boot = bootstrap_curve(hs, REF).curve  # nodes at month starts
    # mid-June: the step of 13 June is resolved by the meeting curve but averaged by month nodes
    t = float(TRUE.time(dt.date(2024, 6, 5)))
    assert step.instantaneous_forward(t) == pytest.approx(RATES[3], abs=1e-11)
    assert abs(boot.instantaneous_forward(t) - RATES[3]) > 3e-4


def test_unidentified_segment_raises() -> None:
    hs = [_fwd(dt.date(2024, mo, 1), dt.date(2024, mo + 1, 1), "averaged") for mo in range(2, 12)]
    with pytest.raises(ValueError, match="not identified"):
        meeting_date_curve(hs, REF, STEPS)  # nothing covers 15 Jan - 1 Feb
    with pytest.raises(ValueError, match="helpers"):
        meeting_date_curve(hs[:2], REF, STEPS)
    with pytest.raises(ValueError):
        meeting_date_curve([], REF, STEPS)
