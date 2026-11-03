"""Tests for spxa.rates.indices."""

from __future__ import annotations

import datetime as dt

import pytest

from spxa.rates import (
    ESTR,
    EURIBOR3M,
    EURIBOR6M,
    SOFR,
    SONIA,
    FixingHistory,
    IborIndex,
    OvernightIndex,
    available_indices,
    get_index,
)

D = dt.date


class TestOvernight:
    def test_sofr_fixings_skip_independence_day(self) -> None:
        dates = SOFR.fixing_dates("2024-07-01", "2024-07-08")
        assert dates == [D(2024, 7, 1), D(2024, 7, 2), D(2024, 7, 3), D(2024, 7, 5)]
        assert SOFR.accrual("2024-07-03") == pytest.approx(2 / 360, abs=1e-16)
        assert SOFR.accrual("2024-07-05") == pytest.approx(3 / 360, abs=1e-16)
        assert not SOFR.is_fixing_date("2024-07-04")

    def test_accruals_tile_the_period(self) -> None:
        for idx in (SOFR, ESTR, SONIA):
            dates = idx.fixing_dates("2024-01-02", "2024-12-31")
            ends = [idx.next_fixing_date(d) for d in dates]
            assert all(e == n for e, n in zip(ends[:-1], dates[1:]))
            total = sum(idx.accrual(d) for d in dates)
            assert total == pytest.approx(idx.dc.year_fraction(dates[0], ends[-1]), abs=1e-14)

    def test_sonia_and_estr_holidays(self) -> None:
        # UK spring bank holiday Monday 27 May 2024: Friday's fixing accrues 4 days, ACT/365F
        assert SONIA.accrual("2024-05-24") == pytest.approx(4 / 365, abs=1e-16)
        # TARGET closes 25 and 26 December
        assert ESTR.next_fixing_date("2024-12-24") == D(2024, 12, 27)
        assert ESTR.accrual("2024-12-24") == pytest.approx(3 / 360, abs=1e-16)

    def test_period_starting_on_holiday(self) -> None:
        assert SOFR.fixing_dates("2024-07-04", "2024-07-06") == [D(2024, 7, 5)]
        assert SOFR.fixing_dates("2024-07-06", "2024-07-08") == []


class TestIbor:
    def test_spot_lag_and_maturity(self) -> None:
        assert EURIBOR3M.value_date("2024-01-15") == D(2024, 1, 17)
        assert EURIBOR3M.maturity_date("2024-01-17") == D(2024, 4, 17)
        assert EURIBOR3M.accrual("2024-01-17") == pytest.approx(91 / 360, abs=1e-16)

    def test_fixing_date_over_easter(self) -> None:
        # Good Friday 29 March and Easter Monday 1 April 2024 are TARGET holidays
        assert EURIBOR3M.fixing_date("2024-04-03") == D(2024, 3, 28)
        assert EURIBOR3M.value_date("2024-03-28") == D(2024, 4, 3)
        assert EURIBOR3M.fixing_date("2024-01-02") == D(2023, 12, 28)

    def test_end_of_month_and_modified_following(self) -> None:
        # last business day of February -> last business day of May
        assert EURIBOR3M.maturity_date("2024-02-29") == D(2024, 5, 31)
        # 30 Aug 2024 is the last business day (31st is Saturday) -> 29 Nov (30th is Saturday)
        assert EURIBOR3M.maturity_date("2024-08-30") == D(2024, 11, 29)
        assert EURIBOR6M.maturity_date("2024-01-31") == D(2024, 7, 31)
        # ordinary date rolled following: 15 June 2024 is a Saturday
        assert EURIBOR3M.maturity_date("2024-03-15") == D(2024, 6, 17)
        no_eom = IborIndex("X3M", "EUR", "3M", end_of_month=False)
        assert no_eom.maturity_date("2024-02-29") == D(2024, 5, 29)

    def test_zero_spot_lag_and_validation(self) -> None:
        idx = IborIndex("T0", "EUR", "1M", fixing_days=0)
        assert idx.fixing_date("2024-03-30") == D(2024, 4, 2)
        assert idx.value_date("2024-04-02") == D(2024, 4, 2)
        with pytest.raises(ValueError):
            IborIndex("bad", "EUR", "3Q")
        with pytest.raises(ValueError):
            IborIndex("bad", "EUR", "3M", fixing_days=-1)


class TestRegistry:
    def test_lookup(self) -> None:
        assert get_index("sofr") is SOFR
        assert get_index("€STR") is ESTR
        assert get_index("Euribor 3M") is EURIBOR3M
        assert get_index("EURIBOR1Y").tenor == "12M"
        assert "SONIA" in available_indices()
        with pytest.raises(ValueError, match="unknown index"):
            get_index("LIBOR3M")
        assert isinstance(SOFR, OvernightIndex) and isinstance(EURIBOR6M, IborIndex)


class TestFixingHistory:
    def test_basic(self) -> None:
        h = FixingHistory.from_pairs("SOFR", [("2024-07-02", 0.054), (D(2024, 7, 1), 0.0533)])
        assert len(h) == 2 and list(h) == [D(2024, 7, 1), D(2024, 7, 2)]
        assert h["2024-07-01"] == 0.0533 and h.get("2024-07-03") is None
        assert "2024-07-02" in h and "2024-07-03" not in h and 3.5 not in h
        assert h.last_date == D(2024, 7, 2)
        with pytest.raises(KeyError, match="no SOFR fixing"):
            h["2024-07-05"]
        with pytest.raises(ValueError):
            h.add("2024-07-05", float("nan"))
        h2 = FixingHistory("ESTR", {"2024-01-02": 0.039})  # type: ignore[dict-item]
        assert h2.as_mapping() == {D(2024, 1, 2): 0.039}
        assert FixingHistory("X").last_date is None
