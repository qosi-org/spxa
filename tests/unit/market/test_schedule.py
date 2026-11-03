"""Business-day conventions and schedule generation."""

import datetime as dt

import pytest

from spxa.market import Calendar, Schedule, adjust, available_business_day_conventions
from spxa.market import year_fraction

D = dt.date


class TestAdjust:
    @pytest.mark.parametrize(
        "d, conv, expected",
        [
            ("2024-08-31", "F", D(2024, 9, 2)),
            ("2024-08-31", "MF", D(2024, 8, 30)),
            ("2024-08-31", "P", D(2024, 8, 30)),
            ("2024-06-01", "P", D(2024, 5, 31)),
            ("2024-06-01", "MP", D(2024, 6, 3)),
            ("2024-06-01", "U", D(2024, 6, 1)),
            ("2024-06-04", "MF", D(2024, 6, 4)),
            ("2024-03-29", "Following", D(2024, 4, 2)),  # TARGET Good Friday + Easter Monday
            ("2024-03-29", "modified following", D(2024, 3, 28)),
        ],
    )
    def test_conventions(self, d, conv, expected):
        assert adjust(d, Calendar("TARGET"), conv) == expected

    def test_unknown(self):
        assert available_business_day_conventions() == ["F", "MF", "P", "MP", "U"]
        with pytest.raises(ValueError, match="business-day convention"):
            adjust("2024-01-01", "TARGET", "NEAREST")


def test_regular_swap_schedule():
    s = Schedule("2024-03-15", "2029-03-15", "6M", "TARGET", "MF")
    assert len(s) == 11 and all(s.is_regular)
    assert s.unadjusted_dates[1] == D(2024, 9, 15)
    assert s.dates[1] == D(2024, 9, 16)  # Sunday -> Monday
    assert s.dates[2] == D(2025, 3, 17)  # Saturday -> Monday
    assert s.dates[0] == D(2024, 3, 15) and s.dates[-1] == D(2029, 3, 15)
    assert s.frequency == 2 and list(s)[3] == s[3]


class TestStubs:
    a, b = "2024-01-10", "2025-07-15"

    def test_short_front(self):
        s = Schedule(self.a, self.b, "6M")
        assert s.unadjusted_dates == [
            D(2024, 1, 10),
            D(2024, 1, 15),
            D(2024, 7, 15),
            D(2025, 1, 15),
            D(2025, 7, 15),
        ]
        assert s.is_regular == [False, True, True, True]

    def test_long_front(self):
        s = Schedule(self.a, self.b, "6M", stub="long_front")
        assert s.unadjusted_dates == [
            D(2024, 1, 10),
            D(2024, 7, 15),
            D(2025, 1, 15),
            D(2025, 7, 15),
        ]
        assert s.is_regular == [False, True, True]

    def test_short_back(self):
        s = Schedule(self.a, self.b, "6M", stub="short_back")
        assert s.unadjusted_dates == [
            D(2024, 1, 10),
            D(2024, 7, 10),
            D(2025, 1, 10),
            D(2025, 7, 10),
            D(2025, 7, 15),
        ]
        assert s.is_regular == [True, True, True, False]

    def test_long_back(self):
        s = Schedule(self.a, self.b, "6M", stub="long_back")
        assert s.unadjusted_dates == [
            D(2024, 1, 10),
            D(2024, 7, 10),
            D(2025, 1, 10),
            D(2025, 7, 15),
        ]
        assert s.is_regular == [True, True, False]

    def test_period_shorter_than_tenor(self):
        s = Schedule("2024-01-10", "2024-03-01", "6M")
        assert s.unadjusted_dates == [D(2024, 1, 10), D(2024, 3, 1)]
        assert s.is_regular == [False]
        assert Schedule("2024-01-10", "2024-03-01", "6M", stub="long_front").is_regular == [False]


def test_end_of_month_rule():
    eom = Schedule("2024-02-29", "2025-02-28", "3M", convention="U", end_of_month=True)
    assert eom.unadjusted_dates == [
        D(2024, 2, 29),
        D(2024, 5, 31),
        D(2024, 8, 31),
        D(2024, 11, 30),
        D(2025, 2, 28),
    ]
    assert all(eom.is_regular)
    plain = Schedule("2024-02-29", "2025-02-28", "3M", convention="U")
    assert plain.unadjusted_dates[1:4] == [D(2024, 5, 28), D(2024, 8, 28), D(2024, 11, 28)]
    assert plain.is_regular[0] is False
    # anchor not at a month end: rule has no effect
    mid = Schedule("2024-01-30", "2024-07-30", "1M", convention="U", end_of_month=True)
    assert mid.unadjusted_dates[1] == D(2024, 2, 29) and mid.unadjusted_dates[2] == D(2024, 3, 30)


def test_backward_anchoring_does_not_drift():
    s = Schedule("2023-03-31", "2024-03-31", "1M", convention="U")
    # every date is anchored at 31 Mar 2024: clamped dates do not propagate
    assert s.unadjusted_dates[1] == D(2023, 4, 30) and s.unadjusted_dates[2] == D(2023, 5, 31)
    assert s.unadjusted_dates[-2] == D(2024, 2, 29)
    assert all(s.is_regular)


def test_explicit_stub_dates():
    s = Schedule(
        "2024-01-10",
        "2026-02-20",
        "6M",
        first_date="2024-03-15",
        next_to_last_date="2025-09-15",
        convention="U",
    )
    assert s.unadjusted_dates == [
        D(2024, 1, 10),
        D(2024, 3, 15),
        D(2024, 9, 15),
        D(2025, 3, 15),
        D(2025, 9, 15),
        D(2026, 2, 20),
    ]
    assert s.is_regular == [False, True, True, True, False]
    only_first = Schedule("2024-01-10", "2025-03-15", "6M", first_date="2024-03-15")
    assert only_first.unadjusted_dates[-1] == D(2025, 3, 15) and only_first.is_regular[-1]
    only_last = Schedule("2024-03-15", "2025-05-01", "6M", next_to_last_date="2025-03-15")
    assert only_last.unadjusted_dates[:2] == [D(2024, 3, 15), D(2024, 9, 15)]
    assert only_last.is_regular == [True, True, False]
    with pytest.raises(ValueError, match="whole number"):
        Schedule(
            "2024-01-10",
            "2026-02-20",
            "6M",
            first_date="2024-03-15",
            next_to_last_date="2025-08-15",
        )
    with pytest.raises(ValueError, match="inside"):
        Schedule("2024-01-10", "2026-02-20", "6M", first_date="2027-01-01")


def test_icma_bond_with_long_first_coupon():
    """ISDA memo long first coupon: 15 Aug 2002 -> 15 Jul 2003 accrues 0.91576087."""
    s = Schedule("2002-08-15", "2005-07-15", "6M", first_date="2003-07-15", convention="U")
    assert s.is_regular[0] is False
    p0 = s.periods()[0]
    assert (p0.ref_start, p0.ref_end) == (D(2003, 1, 15), D(2003, 7, 15))
    yfs = s.year_fractions("ACT/ACT ICMA")
    assert yfs[0] == pytest.approx(0.915760869565, abs=1e-12)
    assert yfs[1:] == pytest.approx([0.5] * 4, abs=1e-15)


def test_icma_short_back_stub_reference():
    s = Schedule("2024-01-15", "2025-03-15", "6M", stub="short_back", convention="U")
    last = s.periods()[-1]
    assert (last.ref_start, last.ref_end) == (D(2025, 1, 15), D(2025, 7, 15))
    assert s.year_fractions("ACT/ACT ICMA")[-1] == pytest.approx(59 / (2 * 181))


def test_day_tenor_and_single_period():
    s = Schedule("2024-01-01", "2024-03-25", "28D", convention="U")
    assert s.unadjusted_dates == [D(2024, 1, 1), D(2024, 1, 29), D(2024, 2, 26), D(2024, 3, 25)]
    assert s.frequency is None
    single = Schedule("2024-01-01", "2024-12-31", None)
    assert len(single) == 2 and single.is_regular == [True]
    assert single.year_fractions("ACT/360") == [pytest.approx(365 / 360)]


def test_adjusted_dates_and_termination_convention():
    s = Schedule(
        "2024-05-31",
        "2024-08-31",
        "1M",
        "TARGET",
        "MF",
        termination_convention="F",
        end_of_month=True,
    )
    assert s.unadjusted_dates == [D(2024, 5, 31), D(2024, 6, 30), D(2024, 7, 31), D(2024, 8, 31)]
    assert s.dates == [D(2024, 5, 31), D(2024, 6, 28), D(2024, 7, 31), D(2024, 9, 2)]
    yf = s.year_fractions("ACT/360")
    assert yf == pytest.approx([28 / 360, 33 / 360, 33 / 360])


def test_validation():
    with pytest.raises(ValueError):
        Schedule("2025-01-01", "2024-01-01", "6M")
    with pytest.raises(ValueError):
        Schedule("2024-01-01", "2025-01-01", "6M", stub="middle")
    with pytest.raises(ValueError):
        Schedule("2024-01-01", "2025-01-01", "-6M")


def test_bus_252_with_calendar():
    assert year_fraction("2024-01-01", "2025-01-01", "BUS/252", calendar=Calendar("TARGET")) == (
        pytest.approx(256 / 252)
    )
