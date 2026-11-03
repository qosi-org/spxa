"""Day-count conventions (ISDA 2006 section 4.16, ICMA Rule 251)."""

import datetime as dt

import pytest

from spxa.market import DayCount, available_day_counts, day_count, year_fraction

D = dt.date


class TestActActISDAMemo:
    """The worked examples of the ISDA memo 'EMU and market conventions' (1998)."""

    def test_regular_period(self):
        assert year_fraction("2003-11-01", "2004-05-01", "ACT/ACT ISDA") == pytest.approx(
            61 / 365 + 121 / 366, abs=1e-15
        )
        assert year_fraction("2003-11-01", "2004-05-01", "ACT/ACT ISDA") == pytest.approx(
            0.497724380567, abs=1e-12
        )
        icma = year_fraction(
            "2003-11-01",
            "2004-05-01",
            "ACT/ACT ICMA",
            ref_start="2003-11-01",
            ref_end="2004-05-01",
            frequency=2,
        )
        assert icma == pytest.approx(0.5, abs=1e-15)

    def test_short_first_period(self):
        kw = dict(ref_start="1998-07-01", ref_end="1999-07-01", frequency=1)
        assert year_fraction("1999-02-01", "1999-07-01", "ACT/ACT ISDA") == pytest.approx(
            0.410958904110, abs=1e-12
        )
        assert year_fraction("1999-02-01", "1999-07-01", "ACT/ACT ICMA", **kw) == pytest.approx(
            0.410958904110, abs=1e-12
        )

    def test_long_first_period(self):
        kw = dict(ref_start="2003-01-15", ref_end="2003-07-15", frequency=2)
        assert year_fraction("2002-08-15", "2003-07-15", "ACT/ACT ISDA") == pytest.approx(
            0.915068493151, abs=1e-12
        )
        # 153 days in the notional period 15 Jul 2002 - 15 Jan 2003 (184 days) + a full period
        expected = 153 / (2 * 184) + 0.5
        got = year_fraction("2002-08-15", "2003-07-15", "ACT/ACT ICMA", **kw)
        assert got == pytest.approx(expected, abs=1e-15)
        assert got == pytest.approx(0.915760869565, abs=1e-12)

    def test_short_final_period(self):
        kw = dict(ref_start="1999-07-30", ref_end="2000-01-30", frequency=2)
        assert year_fraction("1999-07-30", "2000-01-30", "ACT/ACT ISDA") == pytest.approx(
            0.503892506924, abs=1e-12
        )
        assert year_fraction("1999-07-30", "2000-01-30", "ACT/ACT ICMA", **kw) == pytest.approx(
            0.5, abs=1e-15
        )

    def test_long_final_period(self):
        kw = dict(ref_start="1999-11-30", ref_end="2000-02-29", frequency=4, end_of_month=True)
        # 32 days in 1999, 120 days in the leap year 2000
        assert year_fraction("1999-11-30", "2000-04-30", "ACT/ACT ISDA") == pytest.approx(
            32 / 365 + 120 / 366, abs=1e-15
        )
        # a regular quarter, then 61 days of the notional period 29 Feb - 31 May 2000 (92 days)
        got = year_fraction("1999-11-30", "2000-04-30", "ACT/ACT ICMA", **kw)
        assert got == pytest.approx(0.25 + 61 / (4 * 92), abs=1e-15)
        assert got == pytest.approx(0.415760869565, abs=1e-12)


def test_act_act_isda_multi_year_and_additivity():
    # whole calendar years count exactly 1 each
    assert year_fraction("2001-01-01", "2005-01-01", "ACT/ACT ISDA") == pytest.approx(4.0)
    a, m, b = D(2003, 3, 17), D(2006, 9, 2), D(2012, 2, 29)
    yf = DayCount("ACT/ACT")
    assert yf.year_fraction(a, b) == pytest.approx(yf.year_fraction(a, m) + yf.year_fraction(m, b))


@pytest.mark.parametrize(
    "start, end, days",
    [
        ("2024-01-01", "2024-07-01", 182),
        ("2023-12-29", "2024-03-01", 63),
        ("2024-02-29", "2024-02-29", 0),
    ],
)
def test_actual_conventions(start, end, days):
    assert day_count(start, end, "ACT/360") == days
    assert year_fraction(start, end, "ACT/360") == pytest.approx(days / 360)
    assert year_fraction(start, end, "ACT/365F") == pytest.approx(days / 365)
    assert year_fraction(start, end, "ACT/364") == pytest.approx(days / 364)


# start, end, (30/360 bond basis, 30/360 US, 30E/360, 30E/360 ISDA) day counts
THIRTY = [
    ("2007-01-15", "2007-01-30", (15, 15, 15, 15)),
    ("2007-01-15", "2007-02-15", (30, 30, 30, 30)),
    ("2007-02-28", "2007-03-31", (33, 30, 32, 30)),
    ("2008-02-29", "2009-02-28", (359, 360, 359, 360)),
    ("2007-08-31", "2008-02-29", (179, 179, 179, 180)),
    ("2007-01-31", "2007-03-31", (60, 60, 60, 60)),
    ("2007-03-15", "2007-03-31", (16, 16, 15, 15)),
    ("2007-03-30", "2007-03-31", (0, 0, 0, 0)),
    ("2007-02-28", "2007-04-30", (62, 60, 62, 60)),
]


@pytest.mark.parametrize("start, end, expected", THIRTY)
def test_thirty_360_family(start, end, expected):
    names = ("30/360", "30/360 US", "30E/360", "30E/360 ISDA")
    got = tuple(day_count(start, end, n) for n in names)
    assert got == expected
    for n, days in zip(names, expected):
        assert year_fraction(start, end, n) == pytest.approx(days / 360)


def test_30e_360_isda_february_termination():
    dc = DayCount("30E/360 ISDA")
    assert dc.day_count("2008-02-29", "2009-02-28") == 360
    assert dc.day_count("2008-02-29", "2009-02-28", termination="2009-02-28") == 358
    # termination rule only affects February
    assert dc.day_count("2008-03-31", "2008-04-30", termination="2008-04-30") == 30


def test_30_360_us_without_eom():
    assert day_count("2007-02-28", "2007-03-31", "30/360 US", eom=False) == 33


def test_bus_252():
    class FakeCalendar:
        def business_days_between(self, start, end):
            return 21

    assert year_fraction("2024-01-01", "2024-02-01", "BUS/252", calendar=FakeCalendar()) == (
        pytest.approx(21 / 252)
    )
    with pytest.raises(ValueError, match="calendar"):
        year_fraction("2024-01-01", "2024-02-01", "BUS/252")
    with pytest.raises(ValueError):
        day_count("2024-01-01", "2024-02-01", "BUS/252")


class TestDayCountObject:
    def test_aliases(self):
        assert DayCount("Actual/365 Fixed") == DayCount("act/365f") == "ACT/365F"
        assert DayCount("act_act_isma").name == "ACT/ACT ICMA"
        assert DayCount("30U/360").name == "30/360 US"
        assert DayCount(DayCount("A/360")).name == "ACT/360"
        assert DayCount("ACT/360") != "ACT/365F"
        assert DayCount("ACT/360") != "nonsense"
        assert len({DayCount("ACT/365"), DayCount("ACT/365F")}) == 1
        assert str(DayCount("30e/360")) == "30E/360"
        assert set(available_day_counts()) >= {"ACT/360", "ACT/ACT ICMA", "30E/360 ISDA"}

    def test_unknown(self):
        with pytest.raises(ValueError, match="unknown day-count"):
            DayCount("ACT/999")

    def test_reversed_dates(self):
        for name in ("ACT/360", "ACT/ACT ISDA", "30/360"):
            dc = DayCount(name)
            assert dc.year_fraction("2025-06-30", "2024-01-15") == pytest.approx(
                -dc.year_fraction("2024-01-15", "2025-06-30")
            )

    def test_icma_validation(self):
        with pytest.raises(ValueError, match="frequency"):
            year_fraction("2024-01-01", "2024-07-01", "ACT/ACT ICMA")
        with pytest.raises(ValueError, match="frequency"):
            year_fraction("2024-01-01", "2024-07-01", "ACT/ACT ICMA", frequency=5)
        assert year_fraction("2024-01-01", "2024-01-01", "ACT/ACT ICMA", frequency=2) == 0.0
        # without a reference period the accrual period is its own reference
        assert year_fraction("2024-01-01", "2024-07-01", "ACT/ACT ICMA", frequency=2) == 0.5
