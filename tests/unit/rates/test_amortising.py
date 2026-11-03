"""Amortising and accreting swaps."""

from __future__ import annotations

import datetime as dt

import pytest

from spxa.market import InterpolatedDiscountCurve
from spxa.rates import (
    EURIBOR6M,
    SOFR,
    linear_notional,
    ois_swap,
    step_notional,
    vanilla_swap,
)

D = dt.date
REF = "2024-01-15"
DISC = InterpolatedDiscountCurve(
    [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    [0.987, 0.974, 0.951, 0.91, 0.80, 0.65, 0.30],
    reference_date=REF,
)
PROJ = InterpolatedDiscountCurve(
    [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    [0.985, 0.970, 0.945, 0.90, 0.785, 0.63, 0.28],
    reference_date=REF,
)


class TestNotionalFunctions:
    def test_step(self) -> None:
        f = step_notional(["2024-01-17", "2025-01-17", "2026-01-19"], [3.0, 2.0, 1.0])
        assert f(D(2024, 1, 17)) == 3.0 and f(D(2025, 1, 16)) == 3.0
        assert f(D(2025, 1, 17)) == 2.0 and f(D(2030, 1, 1)) == 1.0
        with pytest.raises(ValueError, match="before"):
            f(D(2024, 1, 16))
        with pytest.raises(ValueError):
            step_notional(["2025-01-01", "2024-01-01"], [1.0, 2.0])
        with pytest.raises(ValueError):
            step_notional(["2025-01-01"], [1.0, 2.0])

    def test_linear(self) -> None:
        f = linear_notional("2024-01-01", "2024-12-31", 100.0)  # 365 days
        assert f(D(2024, 1, 1)) == 100.0 and f(D(2024, 12, 31)) == 0.0
        assert f(D(2024, 7, 1)) == pytest.approx(100 * (1 - 182 / 365), abs=1e-12)
        assert f(D(2023, 1, 1)) == 100.0 and f(D(2026, 1, 1)) == 0.0  # clamped
        g = linear_notional("2024-01-01", "2025-01-01", 1.0, 2.0)  # accreting
        assert g(D(2024, 7, 2)) == pytest.approx(1.5, abs=1e-15)
        with pytest.raises(ValueError):
            linear_notional("2024-01-01", "2024-01-01", 1.0)


class TestDecomposition:
    def test_ois_steps_equal_sum_of_bullets(self) -> None:
        # notional 5, 4, 3, 2, 1 on a 5Y annual OIS = 1Y + 2Y + 3Y + 4Y + 5Y bullets
        K = 0.041
        amort = ois_swap(REF, "5Y", K, notional=[5.0, 4.0, 3.0, 2.0, 1.0])
        bullets = [ois_swap(REF, f"{j}Y", K) for j in range(1, 6)]
        for b in bullets:  # the bullets share the amortising swap's coupon dates
            assert [c.end for c in b.fixed] == [c.end for c in amort.fixed[: len(b.fixed)]]  # type: ignore[attr-defined]
        pv = amort.pv(DISC, PROJ)
        assert pv == pytest.approx(sum(b.pv(DISC, PROJ) for b in bullets), abs=1e-14)
        ann = [b.annuity(DISC) for b in bullets]
        assert amort.annuity(DISC) == pytest.approx(sum(ann), rel=1e-14)
        # the par rate is the annuity-weighted average of the bullet par rates
        avg = sum(a * b.par_rate(DISC, PROJ) for a, b in zip(ann, bullets)) / sum(ann)
        assert amort.par_rate(DISC, PROJ) == pytest.approx(avg, rel=1e-13)
        assert amort.dv01(DISC, PROJ) == pytest.approx(
            sum(b.dv01(DISC, PROJ) for b in bullets), rel=1e-9
        )

    def test_vanilla_step_function_equals_sum_of_bullets(self) -> None:
        # annual fixed vs semiannual EURIBOR 6M: the notional is a function of the date
        K = 0.035
        start = D(2024, 1, 17)
        anniversaries = [start] + [D(2024 + j, 1, 17) for j in range(1, 4)]
        f = step_notional(anniversaries, [4.0, 3.0, 2.0, 1.0])
        amort = vanilla_swap(REF, "4Y", K, notional=f)
        assert [c.notional for c in amort.floating] == [4, 4, 3, 3, 2, 2, 1, 1]  # type: ignore[attr-defined]
        bullets = [vanilla_swap(REF, f"{j}Y", K) for j in range(1, 5)]
        assert amort.pv(DISC, PROJ) == pytest.approx(
            sum(b.pv(DISC, PROJ) for b in bullets), abs=1e-14
        )
        with pytest.raises(ValueError, match="different schedules"):
            vanilla_swap(REF, "4Y", K, notional=[4.0, 3.0, 2.0, 1.0])


class TestHandCalculation:
    def test_single_curve_amortising_par_rate(self) -> None:
        # single curve: S = sum N_i (P(s_i)/P(e_i) - 1) P(p_i) / sum N_i tau_i P(p_i)
        f = linear_notional("2024-01-18", "2029-01-18", 10.0, 0.0)
        sw = ois_swap(REF, "5Y", 0.0, notional=f)
        num = den = 0.0
        for c in sw.fixed:
            n = c.notional  # type: ignore[attr-defined]
            assert n == pytest.approx(f(c.start), abs=0)  # type: ignore[attr-defined]
            p_pay = float(DISC.discount(c.payment_date))
            grow = float(DISC.discount(c.start)) / float(DISC.discount(c.end))  # type: ignore[attr-defined]
            num += n * (grow - 1.0) * p_pay
            den += n * c.accrual * p_pay  # type: ignore[attr-defined]
        assert sw.par_rate(DISC, DISC) == pytest.approx(num / den, abs=1e-15)
        assert sw.fixed[0].notional == 10.0  # type: ignore[attr-defined]

    def test_accreting_dv01_exceeds_bullet(self) -> None:
        bullet = ois_swap(REF, "10Y", 0.04, index=SOFR)
        accreting = ois_swap(REF, "10Y", 0.04, notional=linear_notional(REF, "2034-01-18", 1, 2))
        amort = ois_swap(REF, "10Y", 0.04, notional=linear_notional(REF, "2034-01-18", 1, 0))
        d = [s.dv01(DISC, DISC) for s in (amort, bullet, accreting)]
        assert 0 < d[0] < d[1] < d[2]

    def test_euribor_amortising_notional_per_leg(self) -> None:
        f = linear_notional("2024-01-17", "2027-01-18", 3.0, 0.0)
        sw = vanilla_swap(REF, "3Y", 0.03, index=EURIBOR6M, notional=f)
        assert [c.notional for c in sw.fixed] == [f(c.start) for c in sw.fixed]  # type: ignore[attr-defined]
        assert [c.notional for c in sw.floating] == [f(c.start) for c in sw.floating]  # type: ignore[attr-defined]
