"""IMM dates and futures convexity."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from spxa.market import FlatCurve, InterpolatedDiscountCurve
from spxa.rates import (
    EURIBOR3M,
    FixingHistory,
    RateFuture,
    hull_white_futures_convexity,
    hull_white_rate_convexity,
    imm_code,
    imm_date,
    imm_from_code,
    is_imm_date,
    next_imm_date,
)

D = dt.date


class TestIMM:
    def test_third_wednesdays(self) -> None:
        assert [imm_date(2024, m) for m in (3, 6, 9, 12)] == [
            D(2024, 3, 20),
            D(2024, 6, 19),
            D(2024, 9, 18),
            D(2024, 12, 18),
        ]
        assert imm_date(
            2025,
            1,
        ) == D(2025, 1, 15)
        assert is_imm_date("2024-06-19") and not is_imm_date("2024-06-18")
        assert not is_imm_date("2025-01-15") and is_imm_date("2025-01-15", quarterly=False)

    def test_next(self) -> None:
        assert next_imm_date("2024-03-20") == D(2024, 6, 19)
        assert next_imm_date("2024-03-20", include_self=True) == D(2024, 3, 20)
        assert next_imm_date("2024-12-19") == D(2025, 3, 19)
        assert next_imm_date("2024-12-19", quarterly=False) == D(2025, 1, 15)

    def test_codes(self) -> None:
        assert imm_code("2024-03-20") == "H24"
        assert imm_code("2025-12-17") == "Z25"
        assert imm_from_code("H24", "2024-01-15") == D(2024, 3, 20)
        assert imm_from_code("H4", "2024-03-21") == D(
            2034, 3, 15
        )  # one-digit codes roll by decades
        assert imm_from_code("z25", "2024-01-15") == D(2025, 12, 17)
        with pytest.raises(ValueError):
            imm_code("2024-03-21")
        with pytest.raises(ValueError):
            imm_from_code("A24", "2024-01-01")


class TestConvexity:
    @pytest.mark.parametrize("a", [0.0, 1e-6, 0.03, 0.5])
    def test_against_gaussian_quadrature(self, a: float) -> None:
        # E^Q[1/P(t1, t2)] from the Hull-White bond formula and the law of r(t1)
        sig, t1, t2, f0 = 0.012, 1.5, 1.75, 0.03
        B = (lambda t: t) if a == 0 else (lambda t: -math.expm1(-a * t) / a)
        v1 = t1 if a == 0 else -math.expm1(-2 * a * t1) / (2 * a)
        mean, var = f0 + sig**2 * B(t1) ** 2 / 2, sig**2 * v1
        x, wq = np.polynomial.hermite_e.hermegauss(60)
        r = mean + math.sqrt(var) * x
        b12 = B(t2 - t1)
        p = math.exp(-f0 * (t2 - t1)) * np.exp(b12 * f0 - sig**2 / 2 * v1 * b12**2 - b12 * r)
        e_inv = float(np.sum(wq / p)) / math.sqrt(2 * math.pi)
        e_log = float(np.sum(wq * -np.log(p))) / math.sqrt(2 * math.pi)
        g = hull_white_futures_convexity(t1, t2, a, sig)
        assert g == pytest.approx(math.log(e_inv) - f0 * (t2 - t1), rel=1e-10)
        h = hull_white_rate_convexity(t1, t2, a, sig)
        assert h == pytest.approx(e_log / (t2 - t1) - f0, rel=1e-10)

    def test_ho_lee_limits(self) -> None:
        sig, t1, t2 = 0.01, 2.0, 2.25
        tau = t2 - t1
        assert hull_white_futures_convexity(t1, t2, 0.0, sig) == pytest.approx(
            sig**2 * tau * t1 * (t1 + 2 * tau) / 2, rel=1e-15
        )
        assert hull_white_rate_convexity(t1, t2, 0.0, sig) == pytest.approx(
            sig**2 * t1 * t2 / 2, rel=1e-14
        )
        assert hull_white_futures_convexity(t1, t2, 1e-9, sig) == pytest.approx(
            hull_white_futures_convexity(t1, t2, 0.0, sig), rel=1e-8
        )
        # a contract fixing today has no convexity
        assert hull_white_futures_convexity(0.0, 0.25, 0.1, sig) == 0.0
        with pytest.raises(ValueError):
            hull_white_futures_convexity(1.0, 1.0, 0.1, sig)
        with pytest.raises(ValueError):
            hull_white_futures_convexity(1.0, 2.0, -0.1, sig)


class TestRateFuture:
    curve = FlatCurve(0.035, reference_date="2024-01-15", day_count="ACT/365F")

    def test_contract(self) -> None:
        fut = RateFuture.from_imm(EURIBOR3M, imm_from_code("U24", "2024-01-15"), 96.25)
        assert (fut.start, fut.end) == (D(2024, 9, 18), D(2024, 12, 18))
        assert fut.rate == pytest.approx(0.0375, abs=1e-15)
        assert fut.accrual == pytest.approx(91 / 360, abs=1e-16)
        F = fut.forward_rate(self.curve)
        assert F == pytest.approx(math.expm1(0.035 * 91 / 365) / (91 / 360), rel=1e-13)
        assert fut.implied_rate(self.curve) == pytest.approx(F, abs=1e-16)  # no volatility

    def test_hull_white_adjustment(self) -> None:
        fut = RateFuture.from_imm(EURIBOR3M, imm_date(2026, 3), 96.0)
        a, sig = 0.05, 0.01
        t1, t2 = self.curve.time(fut.start), self.curve.time(fut.end)
        g = hull_white_futures_convexity(t1, t2, a, sig)
        F, tau = fut.forward_rate(self.curve), fut.accrual
        R = fut.implied_rate(self.curve, a, sig)
        assert 1 + tau * R == pytest.approx((1 + tau * F) * math.exp(g), rel=1e-15)
        # about 2.6bp: close to Hull's sigma^2 t1 t2 / 2 for slow mean reversion
        assert R - F == pytest.approx(sig**2 * t1 * t2 / 2, rel=0.1)
        assert fut.implied_price(self.curve, a, sig) == pytest.approx(100 * (1 - R), rel=1e-15)
        # quote -> forward inverts the adjustment
        q = RateFuture(fut.start, fut.end, 100 * (1 - R), EURIBOR3M)
        assert q.convexity_adjusted_forward(a, sig, t1, t2) == pytest.approx(F, rel=1e-12)
        with pytest.raises(ValueError, match="t1 and t2"):
            q.convexity_adjusted_forward(a, sig)

    def test_fixed_convexity_and_fixings(self) -> None:
        fut = RateFuture(D(2024, 3, 20), D(2024, 6, 20), 96.1, EURIBOR3M, convexity=2e-4)
        assert fut.convexity_adjusted_forward() == pytest.approx(0.039 - 2e-4, abs=1e-15)
        fix = FixingHistory.from_pairs("EURIBOR3M", [("2024-03-18", 0.0391)])
        assert fut.forward_rate(self.curve, fix) == 0.0391
        curve = InterpolatedDiscountCurve([1.0], [0.96], reference_date="2024-03-20")
        expired = RateFuture(D(2024, 3, 20), D(2024, 6, 20), 96.1, EURIBOR3M)
        assert expired.convexity_adjustment(curve, 0.1, 0.01) == 0.0
