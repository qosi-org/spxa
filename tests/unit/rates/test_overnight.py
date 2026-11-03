"""Tests for spxa.rates.overnight (ISDA 2021 compounding conventions)."""

from __future__ import annotations

import datetime as dt
from fractions import Fraction as Fr

import numpy as np
import pytest

from spxa.market import FlatCurve, InterpolatedDiscountCurve
from spxa.rates import (
    ESTR,
    SOFR,
    FixingHistory,
    compounded_overnight_rate,
    curve_projection,
    forward_overnight_rate,
    observation_schedule,
    projected_overnight_rate,
)

D = dt.date

FIX = FixingHistory.from_pairs(
    "SOFR",
    [
        ("2024-06-26", 0.0533),
        ("2024-06-27", 0.0532),
        ("2024-06-28", 0.0540),
        ("2024-07-01", 0.0540),
        ("2024-07-02", 0.0535),
        ("2024-07-03", 0.0533),
        ("2024-07-05", 0.0531),
    ],
)
S, E = "2024-07-01", "2024-07-08"


def exact(rates: list[str], days: list[int], basis: int = 360) -> float:
    g = Fr(1)
    for r, n in zip(rates, days):
        g *= 1 + Fr(r) * n / basis
    return float((g - 1) * basis / sum(days))


class TestPlain:
    def test_hand_calculation(self) -> None:
        res = compounded_overnight_rate(SOFR, S, E, FIX)
        assert res.observation_dates == [D(2024, 7, 1), D(2024, 7, 2), D(2024, 7, 3), D(2024, 7, 5)]
        assert np.allclose(res.weights * 360, [1, 1, 2, 3], rtol=0, atol=1e-12)
        assert res.accrual == pytest.approx(7 / 360, abs=1e-16)
        ref = exact(["0.0540", "0.0535", "0.0533", "0.0531"], [1, 1, 2, 3])
        assert res.rate == pytest.approx(ref, rel=1e-14)
        assert res.fully_fixed and res.n_fixed == 4
        assert res.growth_factor == pytest.approx(1 + ref * 7 / 360, rel=1e-15)

    def test_constant_rate(self) -> None:
        r = 0.04
        res = compounded_overnight_rate(ESTR, "2024-01-02", "2024-07-02", projection=lambda d: r)
        g = np.prod(1 + r * res.weights)
        assert res.rate == pytest.approx((g - 1) / res.accrual, rel=1e-14)
        assert res.rate > r  # compounding beats simple interest
        assert res.n_fixed == 0 and not res.fully_fixed

    def test_averaged(self) -> None:
        res = compounded_overnight_rate(SOFR, S, E, FIX, method="averaged")
        ref = (0.0540 + 0.0535 + 2 * 0.0533 + 3 * 0.0531) / 7
        assert res.rate == pytest.approx(ref, rel=1e-14)

    def test_spreads(self) -> None:
        base = compounded_overnight_rate(SOFR, S, E, FIX).rate
        exc = compounded_overnight_rate(SOFR, S, E, FIX, spread=0.001).rate
        inc = compounded_overnight_rate(
            SOFR, S, E, FIX, spread=0.001, spread_compounding="inclusive"
        ).rate
        assert exc == pytest.approx(base + 0.001, abs=1e-16)
        ref = exact(["0.0550", "0.0545", "0.0543", "0.0541"], [1, 1, 2, 3])
        assert inc == pytest.approx(ref, rel=1e-14)
        assert inc > exc


class TestConventions:
    def test_lookback_without_shift(self) -> None:
        res = compounded_overnight_rate(SOFR, S, E, FIX, lookback=2)
        assert res.observation_dates == [
            D(2024, 6, 27),
            D(2024, 6, 28),
            D(2024, 7, 1),
            D(2024, 7, 2),
        ]
        # weights stay those of the calculation period
        assert np.allclose(res.weights * 360, [1, 1, 2, 3], rtol=0, atol=1e-12)
        ref = exact(["0.0532", "0.0540", "0.0540", "0.0535"], [1, 1, 2, 3])
        assert res.rate == pytest.approx(ref, rel=1e-14)

    def test_observation_shift(self) -> None:
        res = compounded_overnight_rate(SOFR, S, E, FIX, lookback=2, observation_shift=True)
        # observation period [27 Jun, 3 Jul): weights from the shifted period
        assert res.observation_dates == [
            D(2024, 6, 27),
            D(2024, 6, 28),
            D(2024, 7, 1),
            D(2024, 7, 2),
        ]
        assert np.allclose(res.weights * 360, [1, 3, 1, 1], rtol=0, atol=1e-12)
        assert res.accrual == pytest.approx(6 / 360, abs=1e-16)
        ref = exact(["0.0532", "0.0540", "0.0540", "0.0535"], [1, 3, 1, 1])
        assert res.rate == pytest.approx(ref, rel=1e-14)

    def test_lockout(self) -> None:
        res = compounded_overnight_rate(SOFR, S, E, FIX, lockout=2)
        # lockout date = 2 business days before 8 Jul = 3 Jul; it covers 3 and 5 Jul
        assert res.observation_dates == [D(2024, 7, 1), D(2024, 7, 2), D(2024, 7, 3), D(2024, 7, 3)]
        ref = exact(["0.0540", "0.0535", "0.0533", "0.0533"], [1, 1, 2, 3])
        assert res.rate == pytest.approx(ref, rel=1e-14)
        assert compounded_overnight_rate(SOFR, S, E, FIX, lockout=1).rate == pytest.approx(
            compounded_overnight_rate(SOFR, S, E, FIX).rate, abs=0
        )

    def test_isda_lockout_example(self) -> None:
        # ISDA memo: Calculation Period 22 Sep - 22 Dec 2020, lockout 5 -> lockout date 15 Dec,
        # whose rate applies 15-21 Dec inclusive
        _, obs, w, _ = observation_schedule(ESTR, "2020-09-22", "2020-12-22", lockout=5)
        tail = obs[-5:]
        assert tail == [D(2020, 12, 15)] * 5
        assert obs[-6] == D(2020, 12, 14)
        assert round(sum(w[-5:]) * 360) == 7  # 15 Dec to 22 Dec

    def test_period_starting_on_holiday(self) -> None:
        acc, obs, w, total = observation_schedule(SOFR, "2024-07-06", "2024-07-09")
        assert obs == [D(2024, 7, 5), D(2024, 7, 8)]
        assert acc == [D(2024, 7, 6), D(2024, 7, 8)]
        assert np.allclose(w * 360, [2, 1], rtol=0, atol=1e-12)
        assert total == pytest.approx(3 / 360, abs=1e-16)


class TestErrors:
    def test_missing_fixing(self) -> None:
        with pytest.raises(KeyError, match="no SOFR fixing for 2024-07-08"):
            compounded_overnight_rate(SOFR, S, "2024-07-10", FIX)
        res = compounded_overnight_rate(SOFR, S, "2024-07-10", FIX, projection=lambda d: 0.05)
        assert res.n_fixed == 4 and len(res.rates) == 6
        assert res.rates[-1] == 0.05

    def test_mapping_source(self) -> None:
        m = {D(2024, 7, 1): 0.05, D(2024, 7, 2): 0.05}
        assert compounded_overnight_rate(SOFR, "2024-07-01", "2024-07-03", m).rate == pytest.approx(
            exact(["0.05", "0.05"], [1, 1]), rel=1e-14
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"method": "geometric"},
            {"spread_compounding": "both"},
            {"lookback": -1},
            {"lockout": 3, "lookback": 2},
            {"end": "2024-07-01"},
        ],
    )
    def test_bad_arguments(self, kwargs: dict) -> None:
        args = {"index": SOFR, "start": S, "end": E, "fixings": FIX}
        args.update(kwargs)
        with pytest.raises(ValueError):
            compounded_overnight_rate(**args)


class TestProjection:
    curve = InterpolatedDiscountCurve(
        [0.25, 0.5, 1.0, 2.0],
        [0.987, 0.975, 0.952, 0.91],
        reference_date="2024-07-01",
        day_count="ACT/365F",
    )

    def test_forward_overnight_rate(self) -> None:

        f = forward_overnight_rate(SOFR, self.curve, "2024-07-03")
        p0 = self.curve.discount(D(2024, 7, 3))
        p1 = self.curve.discount(D(2024, 7, 5))
        assert f == pytest.approx((p0 / p1 - 1) / (2 / 360), rel=1e-14)

    def test_flat_curve_forward(self) -> None:

        flat = FlatCurve(0.05, reference_date="2024-07-01", day_count="ACT/360")
        # continuously compounded 5% on the same day count: (e^{0.05 tau} - 1)/tau
        tau = 3 / 360
        f = forward_overnight_rate(SOFR, flat, "2024-07-05")
        assert f == pytest.approx((np.exp(0.05 * tau) - 1) / tau, rel=1e-13)

    def test_telescoping_matches_daily_product(self) -> None:

        fast = projected_overnight_rate(SOFR, "2024-07-01", "2024-10-01", self.curve)
        slow = compounded_overnight_rate(
            SOFR, "2024-07-01", "2024-10-01", projection=curve_projection(SOFR, self.curve)
        ).rate
        assert fast == pytest.approx(slow, rel=1e-12)
        tau = SOFR.dc.year_fraction(D(2024, 7, 1), D(2024, 10, 1))
        p = self.curve.discount([D(2024, 7, 1), D(2024, 10, 1)])
        assert fast == pytest.approx((p[0] / p[1] - 1) / tau, rel=1e-14)

    def test_mixed_fixed_and_projected(self) -> None:

        fix = FixingHistory.from_pairs(
            "SOFR", [("2024-07-01", 0.054), ("2024-07-02", 0.0535), ("2024-07-03", 0.0533)]
        )
        fast = projected_overnight_rate(SOFR, "2024-07-01", "2024-08-01", self.curve, fix)
        slow = compounded_overnight_rate(
            SOFR, "2024-07-01", "2024-08-01", fix, curve_projection(SOFR, self.curve)
        )
        assert slow.n_fixed == 3
        assert fast == pytest.approx(slow.rate, rel=1e-12)
        # with a spread (exclusive) the shift is additive
        sp = projected_overnight_rate(
            SOFR, "2024-07-01", "2024-08-01", self.curve, fix, spread=0.002
        )
        assert sp == pytest.approx(fast + 0.002, abs=1e-15)

    def test_conventions_fall_back_to_daily_product(self) -> None:

        fix = FixingHistory.from_pairs("SOFR", [("2024-06-27", 0.0532), ("2024-06-28", 0.054)])
        kw = dict(lookback=2, observation_shift=True)
        a = projected_overnight_rate(SOFR, "2024-07-01", "2024-08-01", self.curve, fix, **kw)
        b = compounded_overnight_rate(
            SOFR, "2024-07-01", "2024-08-01", fix, curve_projection(SOFR, self.curve), **kw
        ).rate
        assert a == b
        # gaps in the fixings also use the daily product
        gap = FixingHistory.from_pairs("SOFR", [("2024-07-01", 0.054), ("2024-07-03", 0.0533)])
        c = projected_overnight_rate(SOFR, "2024-07-01", "2024-08-01", self.curve, gap)
        d = compounded_overnight_rate(
            SOFR, "2024-07-01", "2024-08-01", gap, curve_projection(SOFR, self.curve)
        ).rate
        assert c == d
