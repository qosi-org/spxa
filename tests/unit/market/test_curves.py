"""Discount curves, compounding and equity forwards."""

import math

import numpy as np
import pytest

from spxa.market import (
    Dividend,
    EquityForward,
    FlatCurve,
    InterpolatedDiscountCurve,
    discount_from_rate,
    rate_from_discount,
)


class TestCompounding:
    @pytest.mark.parametrize("comp", ["continuous", "simple", "annual", 2, 4, 12, "quarterly"])
    def test_round_trip(self, comp):
        t = np.array([0.1, 0.5, 1.0, 7.3, 30.0])
        for r in (-0.01, 0.0, 0.035, 0.12):
            df = discount_from_rate(r, t, comp)
            assert rate_from_discount(df, t, comp) == pytest.approx(np.full_like(t, r), abs=1e-13)

    def test_values(self):
        assert discount_from_rate(0.05, 2.0) == pytest.approx(math.exp(-0.1))
        assert discount_from_rate(0.05, 0.5, "simple") == pytest.approx(1 / 1.025)
        assert discount_from_rate(0.05, 3.0, "annual") == pytest.approx(1.05**-3)
        assert discount_from_rate(0.06, 2.0, 2) == pytest.approx(1.03**-4)
        # equivalent rates give the same discount factor
        cont = math.log(1.03) * 2
        assert discount_from_rate(cont, 5.0) == pytest.approx(discount_from_rate(0.06, 5.0, 2))

    def test_errors(self):
        for bad in ("weekly", 0, -2):
            with pytest.raises(ValueError):
                discount_from_rate(0.05, 1.0, bad)
        with pytest.raises(ValueError):
            rate_from_discount(0.99, 0.0)


class TestFlatCurve:
    def test_rates(self):
        c = FlatCurve(0.04)
        assert c.discount(2.5) == pytest.approx(math.exp(-0.1))
        assert c.zero_rate(3.0) == pytest.approx(0.04)
        assert c.zero_rate(3.0, "annual") == pytest.approx(math.exp(0.04) - 1)
        assert c.instantaneous_forward(1.7) == pytest.approx(0.04, abs=1e-9)
        assert c.forward_rate(1.0, 1.5) == pytest.approx((math.exp(0.02) - 1) / 0.5)
        assert c.forward_rate(1.0, 3.0, "continuous") == pytest.approx(0.04)
        assert c.discount([0.0, 1.0]) == pytest.approx([1.0, math.exp(-0.04)])
        assert c.discount(np.array([[1.0, 2.0]])).shape == (1, 2)

    def test_dates(self):
        c = FlatCurve(0.05, compounding="annual", reference_date="2024-01-15")
        assert c.time("2025-01-15") == pytest.approx(366 / 365)
        assert c.discount("2025-01-15") == pytest.approx(1.05 ** -(366 / 365))
        assert c.discount(["2024-01-15", 1.0]) == pytest.approx([1.0, 1 / 1.05])
        with pytest.raises(ValueError, match="reference_date"):
            FlatCurve(0.05).discount("2025-01-15")
        c360 = FlatCurve(0.05, reference_date="2024-01-15", day_count="ACT/360")
        assert c360.time("2024-07-15") == pytest.approx(182 / 360)

    def test_shift(self):
        c = FlatCurve(0.03).shifted(0.01)
        assert c.zero_rate(4.0) == pytest.approx(0.04)
        with pytest.raises(ValueError):
            FlatCurve(0.03).forward_rate(2.0, 1.0)


class TestInterpolatedCurve:
    t = [0.5, 1.0, 2.0, 5.0]
    p = [0.985, 0.968, 0.93, 0.82]

    def test_nodes_and_forwards(self):
        c = InterpolatedDiscountCurve(self.t, self.p)
        assert c.discount(self.t) == pytest.approx(self.p, abs=1e-15)
        assert c.discount(0.0) == 1.0
        # piecewise-constant forwards between nodes
        f1 = math.log(0.968 / 0.93) / 1.0
        for x in (1.1, 1.5, 1.9):
            assert c.instantaneous_forward(x) == pytest.approx(f1, rel=1e-8)
        # flat extrapolation of the last forward
        f_last = math.log(0.93 / 0.82) / 3.0
        assert c.discount(7.0) == pytest.approx(0.82 * math.exp(-f_last * 2.0))
        mids = np.linspace(0.01, 6.0, 200)
        assert np.all(np.diff(c.discount(mids)) < 0)

    def test_validation(self):
        with pytest.raises(ValueError):
            InterpolatedDiscountCurve([1.0, 0.5], [0.99, 0.98])
        with pytest.raises(ValueError):
            InterpolatedDiscountCurve([0.5, 1.0], [0.99, -0.1])
        with pytest.raises(ValueError):
            InterpolatedDiscountCurve([0.0, 1.0], [0.99, 0.95])
        with pytest.raises(ValueError):
            InterpolatedDiscountCurve([1.0], [0.9, 0.8])
        c = InterpolatedDiscountCurve([0.0, 1.0], [1.0, 0.95])
        with pytest.raises(ValueError):
            c.discount(-0.5)


class TestEquityForward:
    def test_continuous_yield(self):
        fwd = EquityForward(100.0, 0.05, dividend_yield=0.02)
        assert fwd.forward(2.0) == pytest.approx(100 * math.exp(0.06))
        assert fwd.discount(2.0) == pytest.approx(math.exp(-0.1))
        assert fwd.forward([0.0, 1.0]) == pytest.approx([100.0, 100 * math.exp(0.03)])

    def test_repo(self):
        fwd = EquityForward(100.0, 0.05, dividend_yield=0.01, repo=0.045)
        assert fwd.forward(1.0) == pytest.approx(100 * math.exp(0.035))
        assert fwd.discount(1.0) == pytest.approx(math.exp(-0.05))

    def test_cash_dividends(self):
        divs = [Dividend(0.5, 2.0), Dividend(1.5, 2.5)]
        fwd = EquityForward(100.0, 0.04, dividends=divs)
        expected = (100 - 2.0 * math.exp(-0.02)) * math.exp(0.04)
        assert fwd.forward(1.0) == pytest.approx(expected)
        expected2 = (100 - 2.0 * math.exp(-0.02) - 2.5 * math.exp(-0.06)) * math.exp(0.08)
        assert fwd.forward(2.0) == pytest.approx(expected2)
        assert fwd.forward(0.4) == pytest.approx(100 * math.exp(0.016))
        assert fwd.dividend_pv(2.0) == pytest.approx(2.0 * math.exp(-0.02) + 2.5 * math.exp(-0.06))
        # forward drops by exactly the dividend across the payment date
        jump = fwd.forward(0.5 - 1e-9) - fwd.forward(0.5)
        assert jump == pytest.approx(2.0, abs=1e-6)

    def test_mixed_dividends_ordering(self):
        """A cash dividend is scaled only by proportional dividends paid after it."""
        r = 0.03
        a = EquityForward(
            100.0, r, dividends=[Dividend(0.25, 1.0), Dividend(0.75, 0.1, "proportional")]
        )
        b = EquityForward(
            100.0, r, dividends=[Dividend(0.25, 0.1, "proportional"), Dividend(0.75, 1.0)]
        )
        fa = (100 * 0.9 - 1.0 * math.exp(-r * 0.25) * 0.9) * math.exp(r)
        fb = (100 * 0.9 - 1.0 * math.exp(-r * 0.75)) * math.exp(r)
        assert a.forward(1.0) == pytest.approx(fa)
        assert b.forward(1.0) == pytest.approx(fb)

    def test_dates_and_validation(self):
        curve = FlatCurve(0.05, reference_date="2024-01-02")
        fwd = EquityForward(50.0, curve, dividends=[Dividend("2024-07-01", 1.0)])
        t_div = curve.time("2024-07-01")
        assert fwd.forward("2025-01-02") == pytest.approx(
            (50 - math.exp(-0.05 * t_div)) * math.exp(0.05 * 366 / 365)
        )
        with pytest.raises(ValueError):
            EquityForward(-1.0, 0.05)
        with pytest.raises(ValueError):
            Dividend(1.0, 1.2, "proportional")
        with pytest.raises(ValueError):
            Dividend(1.0, -1.0)
        with pytest.raises(ValueError):
            Dividend(1.0, 0.1, "stock")
        with pytest.raises(ValueError, match="not positive"):
            EquityForward(1.0, 0.0, dividends=[Dividend(0.5, 2.0)]).forward(1.0)
