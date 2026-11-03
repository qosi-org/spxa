"""Business-day calendars against published holiday lists."""

import datetime as dt

import pytest

from spxa.market import Calendar, JointCalendar, available_calendars, easter_sunday, get_calendar

D = dt.date


@pytest.mark.parametrize(
    "year, expected",
    [
        (2019, D(2019, 4, 21)),
        (2024, D(2024, 3, 31)),
        (2025, D(2025, 4, 20)),
        (2038, D(2038, 4, 25)),  # latest possible date
        (2285, D(2285, 3, 22)),  # earliest possible date
        (1961, D(1961, 4, 2)),
        (2000, D(2000, 4, 23)),
    ],
)
def test_easter(year, expected):
    assert easter_sunday(year) == expected


def test_easter_range_and_weekday():
    for y in range(1583, 2400):
        e = easter_sunday(y)
        assert e.weekday() == 6
        assert D(y, 3, 22) <= e <= D(y, 4, 25)


def _weekday_holidays(name, year):
    return Calendar(name).holidays(D(year, 1, 1), D(year, 12, 31))


def test_target():
    assert _weekday_holidays("TARGET", 2024) == [
        D(2024, 1, 1),
        D(2024, 3, 29),
        D(2024, 4, 1),
        D(2024, 5, 1),
        D(2024, 12, 25),
        D(2024, 12, 26),
    ]
    assert D(1999, 12, 31) in _weekday_holidays("TARGET", 1999)
    assert D(2001, 12, 31) in _weekday_holidays("TARGET", 2001)
    assert D(2002, 12, 31) not in _weekday_holidays("TARGET", 2002)
    # before 2000 only New Year and Christmas
    assert Calendar("TARGET").is_business_day("1999-05-03") is True


def test_uk_2022():
    """2022 England & Wales bank holidays (gov.uk), including the Platinum Jubilee and funeral."""
    assert _weekday_holidays("UK", 2022) == [
        D(2022, 1, 3),
        D(2022, 4, 15),
        D(2022, 4, 18),
        D(2022, 5, 2),
        D(2022, 6, 2),
        D(2022, 6, 3),
        D(2022, 8, 29),
        D(2022, 9, 19),
        D(2022, 12, 26),
        D(2022, 12, 27),
    ]


@pytest.mark.parametrize(
    "year, expected",
    [
        (
            2020,
            [
                D(2020, 1, 1),
                D(2020, 4, 10),
                D(2020, 4, 13),
                D(2020, 5, 8),
                D(2020, 5, 25),
                D(2020, 8, 31),
                D(2020, 12, 25),
                D(2020, 12, 28),
            ],
        ),
        (
            2021,
            [
                D(2021, 1, 1),
                D(2021, 4, 2),
                D(2021, 4, 5),
                D(2021, 5, 3),
                D(2021, 5, 31),
                D(2021, 8, 30),
                D(2021, 12, 27),
                D(2021, 12, 28),
            ],
        ),
        (
            2023,
            [
                D(2023, 1, 2),
                D(2023, 4, 7),
                D(2023, 4, 10),
                D(2023, 5, 1),
                D(2023, 5, 8),
                D(2023, 5, 29),
                D(2023, 8, 28),
                D(2023, 12, 25),
                D(2023, 12, 26),
            ],
        ),
    ],
)
def test_uk_substitutes_and_specials(year, expected):
    assert _weekday_holidays("UK", year) == expected


def test_us_fed_2023():
    """Federal Reserve 2023: Saturday 11 November (Veterans Day) is not observed."""
    assert _weekday_holidays("US-FED", 2023) == [
        D(2023, 1, 2),
        D(2023, 1, 16),
        D(2023, 2, 20),
        D(2023, 5, 29),
        D(2023, 6, 19),
        D(2023, 7, 4),
        D(2023, 9, 4),
        D(2023, 10, 9),
        D(2023, 11, 23),
        D(2023, 12, 25),
    ]


def test_us_sifma_2024():
    assert _weekday_holidays("SOFR", 2024) == [
        D(2024, 1, 1),
        D(2024, 1, 15),
        D(2024, 2, 19),
        D(2024, 3, 29),
        D(2024, 5, 27),
        D(2024, 6, 19),
        D(2024, 7, 4),
        D(2024, 9, 2),
        D(2024, 10, 14),
        D(2024, 11, 11),
        D(2024, 11, 28),
        D(2024, 12, 25),
    ]


def test_us_saturday_rules():
    sifma, fed = Calendar("US-SIFMA"), Calendar("US-FED")
    # Veterans Day 2023 on a Saturday: SIFMA closes Friday 10th, the Fed does not
    assert not sifma.is_business_day("2023-11-10") and fed.is_business_day("2023-11-10")
    # New Year's Day 2022 on a Saturday: neither closes on Friday 31 Dec 2021
    assert sifma.is_business_day("2021-12-31") and fed.is_business_day("2021-12-31")
    # Juneteenth only from 2022; 2022-06-19 was a Sunday -> Monday 20th
    assert fed.is_business_day("2021-06-18") and not fed.is_business_day("2022-06-20")
    # Independence Day 2026 on a Saturday -> SIFMA Friday 3 July
    assert not sifma.is_business_day("2026-07-03") and fed.is_business_day("2026-07-03")


class TestCalendarOperations:
    cal = Calendar("TARGET")

    def test_rolling_and_counting(self):
        # Easter 2024: Good Friday 29 Mar, Easter Monday 1 Apr
        assert self.cal.next_business_day("2024-03-29") == D(2024, 4, 2)
        assert self.cal.previous_business_day("2024-04-01") == D(2024, 3, 28)
        assert self.cal.next_business_day("2024-04-02", include_self=False) == D(2024, 4, 3)
        assert self.cal.add_business_days("2024-03-28", 1) == D(2024, 4, 2)
        assert self.cal.add_business_days("2024-04-02", -1) == D(2024, 3, 28)
        assert self.cal.add_business_days("2024-03-30", 0) == D(2024, 4, 2)
        assert self.cal.business_days_between("2024-03-25", "2024-04-08") == 8
        assert self.cal.business_days_between("2024-04-08", "2024-03-25") == -8
        assert self.cal.business_days_between("2024-01-01", "2025-01-01") == 256

    def test_add_then_count_inverse(self):
        start = D(2024, 1, 2)
        for n in range(1, 60, 7):
            end = self.cal.add_business_days(start, n)
            assert self.cal.business_days_between(start, end) == n

    def test_overrides(self):
        cal = self.cal.with_holidays("2024-07-15").with_business_days("2024-12-26")
        assert not cal.is_business_day("2024-07-15") and cal.is_business_day("2024-12-26")
        assert self.cal.is_business_day("2024-07-15")  # original unchanged
        sat = Calendar("WEEKENDS", extra_business_days=["2024-06-01"])
        assert sat.is_business_day("2024-06-01") and not sat.is_business_day("2024-06-02")
        assert self.cal.holidays("2024-03-28", "2024-04-02", include_weekends=True) == [
            D(2024, 3, 29),
            D(2024, 3, 30),
            D(2024, 3, 31),
            D(2024, 4, 1),
        ]

    def test_names(self):
        assert Calendar("target2").name == "TARGET" and Calendar("gblo").name == "UK"
        assert Calendar("us_sifma").name == "US-SIFMA"
        assert set(available_calendars()) == {"WEEKENDS", "TARGET", "US-FED", "US-SIFMA", "UK"}
        with pytest.raises(ValueError, match="unknown calendar"):
            Calendar("MARS")


def test_joint_calendars():
    any_ = get_calendar("TARGET+UK")
    all_ = JointCalendar(["TARGET", "UK"], rule="holiday_if_all")
    assert isinstance(any_, JointCalendar)
    # 1 May 2024 is a TARGET holiday, 6 May 2024 a UK one, Good Friday both
    for d, any_bd, all_bd in [
        ("2024-05-01", False, True),
        ("2024-05-06", False, True),
        ("2024-03-29", False, False),
        ("2024-05-02", True, True),
    ]:
        assert any_.is_business_day(d) is any_bd
        assert all_.is_business_day(d) is all_bd
    assert get_calendar(["SOFR", Calendar("TARGET")]).is_business_day("2024-11-11") is False
    joint = any_.with_business_days("2024-05-01")
    assert joint.is_business_day("2024-05-01") and isinstance(joint, JointCalendar)
    assert get_calendar(Calendar("UK")).name == "UK"
    with pytest.raises(ValueError):
        JointCalendar([], rule="holiday_if_any")
    with pytest.raises(ValueError):
        JointCalendar(["UK"], rule="sometimes")
