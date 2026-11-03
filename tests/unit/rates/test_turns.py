"""Turn-of-year forward spreads and bootstrapping around them."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from spxa.market.curves import FlatCurve
from spxa.rates.bootstrap import CurveSpec, bootstrap_curve, fit_curves
from spxa.rates.interpolation import PiecewiseCurve
from spxa.rates.turns import ForwardSpread, SpreadAdjustedCurve, turn_of_year
from tests.unit.rates.test_bootstrap import MARKET, REF, _estr_helpers

TURN = turn_of_year(2024, 0.02)  # 200bp over the 2024 year end


def test_turn_dates() -> None:
    assert (TURN.start, TURN.end) == (dt.date(2024, 12, 31), dt.date(2025, 1, 2))
    t2 = turn_of_year(2022, 0.01)  # 31 Dec 2022 is a Saturday
    assert (t2.start, t2.end) == (dt.date(2022, 12, 30), dt.date(2023, 1, 2))
    with pytest.raises(ValueError):
        ForwardSpread(dt.date(2025, 1, 2), dt.date(2024, 12, 31), 0.01)


def test_spread_adjusted_curve() -> None:
    base = FlatCurve(0.03, reference_date=REF)
    c = SpreadAdjustedCurve(base, [TURN])
    a, b = base.time(TURN.start), base.time(TURN.end)
    for d, overlap in (
        (dt.date(2024, 6, 1), 0.0),
        (dt.date(2025, 1, 1), 1 / 365),
        (dt.date(2026, 1, 1), b - a),
    ):
        t = base.time(d)
        assert c.discount(d) == pytest.approx(math.exp(-0.03 * t - 0.02 * overlap), rel=1e-15)
    # the overnight forward over the turn carries the spread, others do not
    on_turn = c.forward_rate(dt.date(2024, 12, 31), dt.date(2025, 1, 2), "continuous")
    assert on_turn == pytest.approx(0.05, rel=1e-12)
    assert c.forward_rate(dt.date(2025, 3, 3), dt.date(2025, 3, 4), "continuous") == pytest.approx(
        0.03
    )
    t = np.array([base.time(dt.date(2025, 1, 1)), base.time(dt.date(2025, 2, 1))])
    assert c.instantaneous_forward(t) == pytest.approx([0.05, 0.03], abs=1e-9)  # FD base forward
    assert SpreadAdjustedCurve(base, []).discount(2.0) == base.discount(2.0)
    with pytest.raises(ValueError):
        SpreadAdjustedCurve(FlatCurve(0.03), [TURN])


def test_bootstrap_recovers_base_under_turn() -> None:
    hs0 = _estr_helpers(MARKET)
    pillars = sorted(h.pillar for h in hs0)
    base = PiecewiseCurve(pillars, MARKET.discount(pillars), reference_date=REF)
    true = SpreadAdjustedCurve(base, [TURN])
    hs = _estr_helpers(true)
    res = bootstrap_curve(hs, REF, spreads=[TURN])
    assert res.max_residual < 1e-15
    assert res.base.node_discounts == pytest.approx(base.node_discounts, rel=1e-14)
    assert res.curve.discount(np.linspace(0, 30, 301)) == pytest.approx(
        true.discount(np.linspace(0, 30, 301)), rel=1e-14
    )
    # ignoring the turn still reprices, but smears it over the 1Y bucket
    plain = bootstrap_curve(hs, REF)
    assert plain.max_residual < 1e-15
    on = plain.curve.forward_rate(TURN.start, TURN.end, "continuous")
    assert on < true.forward_rate(TURN.start, TURN.end, "continuous") - 0.015
    # fit_curves accepts the same spreads
    g = fit_curves([CurveSpec("discount", hs, spreads=[TURN])], REF)
    assert g.max_residual < 1e-14
    assert g.curves["discount"].discount(5.0) == pytest.approx(true.discount(5.0), rel=1e-13)
