"""Bootstrapping a SOFR curve through a futures strip with Hull-White convexity."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from spxa.rates import SOFR, imm_date, ois_swap
from spxa.rates.bootstrap import bootstrap_curve
from spxa.rates.helpers import SwapHelper, quarterly_futures
from spxa.rates.interpolation import MonotoneConvex, PiecewiseCurve

REF = dt.date(2024, 2, 1)
TRUE = PiecewiseCurve.from_zero_rates(
    [0.25, 0.5, 1, 2, 3, 5, 10, 20, 30],
    [0.053, 0.052, 0.050, 0.046, 0.043, 0.041, 0.040, 0.041, 0.039],
    MonotoneConvex(),
    reference_date=REF,
)
A, SIG = 0.03, 0.012


def _helpers(a: float, sigma: float) -> list:
    probe = quarterly_futures(REF, [95.0] * 8)
    prices = [h.future.implied_price(TRUE, A, SIG) for h in probe]
    hs: list = quarterly_futures(REF, prices, a=a, sigma=sigma)
    for t in ("3Y", "5Y", "10Y", "20Y", "30Y"):
        s = ois_swap(REF, t, 0.0, SOFR, payment_lag=2)
        hs.append(SwapHelper(s, quote=s.par_rate(TRUE)))
    return hs


def test_strip_layout() -> None:
    hs = quarterly_futures(REF, [95.0, 95.1, 95.2])
    assert [h.future.start for h in hs] == [imm_date(2024, 3), imm_date(2024, 6), imm_date(2024, 9)]
    assert [h.pillar for h in hs] == [imm_date(2024, 6), imm_date(2024, 9), imm_date(2024, 12)]
    assert all(h.future.end == n.future.start for h, n in zip(hs, hs[1:]))


def test_convexity_aware_bootstrap_pins_true_forwards() -> None:
    hs = _helpers(A, SIG)
    res = bootstrap_curve(hs, REF)
    assert res.max_residual < 1e-12
    for h in hs[:8]:
        f = h.future
        assert f.forward_rate(res.curve) == pytest.approx(f.forward_rate(TRUE), abs=1e-13)
    t = np.linspace(2.5, 30, 50)  # swap-only region: close to the market curve
    assert np.max(np.abs(res.curve.zero_rate(t) - TRUE.zero_rate(t))) < 5e-4


def test_ignoring_convexity_overstates_forwards_exactly() -> None:
    res = bootstrap_curve(_helpers(0.0, 0.0), REF)
    assert res.max_residual < 1e-12
    for h in _helpers(A, SIG)[:8]:
        f = h.future
        tau = f.accrual
        g = math.log1p(tau * f.implied_rate(TRUE, A, SIG)) - math.log1p(tau * f.forward_rate(TRUE))
        assert g > 0
        # without convexity the curve forward equals the futures rate: (1 + tau F') = (1 + tau F) e^g
        lhs = 1 + tau * f.forward_rate(res.curve)
        assert lhs == pytest.approx((1 + tau * f.forward_rate(TRUE)) * math.exp(g), rel=1e-13)
    last = _helpers(A, SIG)[7].future
    bias = last.forward_rate(res.curve) - last.forward_rate(TRUE)
    assert 1e-5 < bias < 1e-3  # a few basis points at two years
