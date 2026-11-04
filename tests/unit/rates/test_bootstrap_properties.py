"""Randomised round-trip properties of curve construction (seeded, deterministic)."""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pytest

from spxa.rates import ESTR, Deposit, ois_swap
from spxa.rates.bootstrap import bootstrap_curve
from spxa.rates.helpers import DepositHelper, SwapHelper
from spxa.rates.interpolation import (
    CubicSplineRT,
    LinearZero,
    LogLinearDiscount,
    MonotoneConvex,
    MonotoneCubicRT,
    PiecewiseCurve,
)

REF = dt.date(2024, 1, 15)
SCHEMES = [LogLinearDiscount(), LinearZero(), MonotoneConvex(), MonotoneCubicRT(), CubicSplineRT()]


def _random_curve(rng: np.random.Generator, scheme: object) -> PiecewiseCurve:
    t = np.sort(rng.choice(np.arange(1, 31), size=rng.integers(5, 9), replace=False)).astype(float)
    t = np.concatenate([[0.25], t])
    r = 0.03 + np.cumsum(rng.normal(0, 0.003, t.size))  # a random walk in zero rates
    return PiecewiseCurve.from_zero_rates(t, r, scheme, reference_date=REF)  # type: ignore[arg-type]


def _helpers(curve: object, rng: np.random.Generator) -> list:
    hs: list = []
    for t in ("1M", "3M"):
        d = Deposit.from_tenor(REF, t, 0.0)
        hs.append(DepositHelper(d, quote=d.implied_rate(curve)))  # type: ignore[arg-type]
    years = np.sort(rng.choice(np.arange(1, 31), size=7, replace=False))
    for y in years:
        s = ois_swap(REF, f"{y}Y", 0.0, ESTR)
        hs.append(SwapHelper(s, quote=s.par_rate(curve)))  # type: ignore[arg-type]
    return hs


@pytest.mark.parametrize("seed", range(5))
def test_curve_round_trips(seed: int) -> None:
    rng = np.random.default_rng(seed)
    scheme = SCHEMES[seed % len(SCHEMES)]
    c = _random_curve(rng, scheme)
    t = np.linspace(0.01, 35, 200)
    assert c.zero_rate(t) == pytest.approx(-np.log(c.discount(t)) / t, abs=1e-15)
    same = c.with_discounts(c.node_discounts)
    assert same.discount(t) == pytest.approx(c.discount(t), rel=1e-15)
    # the forward integrates back to -log P: trapezoid on each node interval, since
    # the forward may jump at nodes (and monotone convex has kinks inside intervals)
    knots = np.concatenate([[0.0], c.node_times[c.node_times < 20.0], [20.0]])
    integral = 0.0
    for a, b in zip(knots, knots[1:]):
        g = np.linspace(a, b, 20001)
        inner = g.copy()  # one-sided limits at the ends of the interval
        inner[0], inner[-1] = a + 1e-12, b - 1e-12
        integral += float(np.trapezoid(c.instantaneous_forward(inner), g))
    assert integral == pytest.approx(-np.log(c.discount(20.0)), abs=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_bootstrap_reprices_random_markets(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    market = _random_curve(rng, MonotoneConvex())
    hs = _helpers(market, rng)
    scheme = SCHEMES[seed % len(SCHEMES)]
    res = bootstrap_curve(hs, REF, scheme)  # type: ignore[arg-type]
    assert res.max_residual < 1e-12
    assert np.all(np.diff(res.base.node_times) > 0)


@pytest.mark.parametrize("seed", range(3))
def test_log_linear_market_recovered_on_its_pillars(seed: int) -> None:
    rng = np.random.default_rng(200 + seed)
    hs0 = _helpers(_random_curve(rng, MonotoneConvex()), rng)
    pillars = sorted(h.pillar for h in hs0)
    src = _random_curve(rng, MonotoneConvex())
    true = PiecewiseCurve(pillars, src.discount(pillars), reference_date=REF)
    hs = [replace(h, quote=h.implied_quote({"discount": true})) for h in hs0]
    res = bootstrap_curve(hs, REF, LogLinearDiscount())
    assert res.curve.node_discounts == pytest.approx(true.node_discounts, rel=1e-13)
