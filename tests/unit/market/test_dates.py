"""Date utilities and tenors."""

import datetime as dt

import numpy as np
import pytest

from spxa.market import (
    Tenor,
    add_months,
    add_tenor,
    days_between,
    days_in_month,
    days_in_year,
    end_of_month,
    is_end_of_month,
    is_leap_year,
    to_date,
)

D = dt.date


class TestToDate:
    def test_inputs(self):
        assert to_date(D(2024, 2, 29)) == D(2024, 2, 29)
        assert to_date(dt.datetime(2024, 2, 29, 17, 30)) == D(2024, 2, 29)
        assert to_date("2024-02-29") == D(2024, 2, 29)
        assert to_date(" 2024-02-29T10:00:00 ") == D(2024, 2, 29)
        assert to_date(np.datetime64("2024-02-29")) == D(2024, 2, 29)
        assert to_date(np.datetime64("1969-12-31T23:00")) == D(1969, 12, 31)

    def test_object_with_date_method(self):
        class Stamp:
            def date(self):
                return D(2030, 1, 2)

        assert to_date(Stamp()) == D(2030, 1, 2)

    @pytest.mark.parametrize("bad", ["2024-13-01", "tomorrow", 3.5, None])
    def test_rejects(self, bad):
        with pytest.raises(TypeError):
            to_date(bad)


def test_calendar_helpers():
    assert [is_leap_year(y) for y in (1900, 2000, 2023, 2024)] == [False, True, False, True]
    assert days_in_year(2024) == 366 and days_in_year(2100) == 365
    assert days_in_month(2024, 2) == 29 and days_in_month(2023, 2) == 28
    assert days_in_month(2023, 4) == 30 and days_in_month(2023, 12) == 31
    assert end_of_month("2023-02-10") == D(2023, 2, 28)
    assert is_end_of_month("2024-02-29") and not is_end_of_month("2023-02-27")
    assert days_between("2024-01-01", "2025-01-01") == 366
    assert days_between("2025-01-01", "2024-01-01") == -366


class TestAddMonths:
    @pytest.mark.parametrize(
        "start, n, expected",
        [
            ("2024-01-31", 1, D(2024, 2, 29)),
            ("2023-01-31", 1, D(2023, 2, 28)),
            ("2024-01-31", 3, D(2024, 4, 30)),
            ("2024-03-31", -1, D(2024, 2, 29)),
            ("2024-01-15", 25, D(2026, 2, 15)),
            ("2024-01-15", -13, D(2022, 12, 15)),
            ("2024-02-29", 12, D(2025, 2, 28)),
            ("2024-02-29", 48, D(2028, 2, 29)),
            ("2024-05-10", 0, D(2024, 5, 10)),
        ],
    )
    def test_clamping(self, start, n, expected):
        assert add_months(start, n) == expected

    def test_end_of_month_rule(self):
        assert add_months("2023-02-28", 1) == D(2023, 3, 28)
        assert add_months("2023-02-28", 1, end_of_month_rule=True) == D(2023, 3, 31)
        assert add_months("2024-04-30", 6, end_of_month_rule=True) == D(2024, 10, 31)
        # not a month end: the rule has no effect
        assert add_months("2024-04-29", 6, end_of_month_rule=True) == D(2024, 10, 29)

    def test_round_trip_mid_month(self):
        d = D(2020, 1, 1)
        while d < D(2030, 1, 1):
            for n in (-24, -7, 1, 5, 120):
                if d.day <= 28:
                    assert add_months(add_months(d, n), -n) == d
            d += dt.timedelta(days=11)


class TestTenor:
    @pytest.mark.parametrize(
        "spec, n, unit", [("3M", 3, "M"), ("10y", 10, "Y"), (" 2W ", 2, "W"), ("1D", 1, "D")]
    )
    def test_parse(self, spec, n, unit):
        t = Tenor(spec)
        assert (t.n, t.unit) == (n, unit)
        assert str(t) == f"{n}{unit}"

    def test_composite_and_equality(self):
        assert Tenor("1Y6M").months == 18 and Tenor("1Y6M").unit == "M"
        assert Tenor("1W3D").days == 10
        assert Tenor("12M") == Tenor("1Y") == "1Y"
        assert Tenor("7D") == Tenor("1W")
        assert Tenor("1M") != Tenor("30D")
        assert len({Tenor("12M"), Tenor("1Y"), Tenor("6M")}) == 2
        assert -Tenor("3M") == Tenor("-3M")
        assert 4 * Tenor("3M") == Tenor("1Y")
        assert Tenor(5, "y").months == 60

    @pytest.mark.parametrize("bad", ["", "M3", "3Q", "1Y2D", "3.5M"])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            Tenor(bad)

    def test_unit_accessors(self):
        assert Tenor("2Y").approx_years() == 2.0
        assert Tenor("73D").approx_years() == pytest.approx(0.2)
        with pytest.raises(ValueError):
            _ = Tenor("3W").months
        with pytest.raises(ValueError):
            _ = Tenor("3M").days


def test_add_tenor():
    assert add_tenor("2024-01-31", "1M") == D(2024, 2, 29)
    assert add_tenor("2024-02-29", "1Y") == D(2025, 2, 28)
    assert add_tenor("2023-02-28", "6M", end_of_month_rule=True) == D(2023, 8, 31)
    assert add_tenor("2024-12-30", "1W") == D(2025, 1, 6)
    assert add_tenor("2024-03-01", "-1D") == D(2024, 2, 29)
