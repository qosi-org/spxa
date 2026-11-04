"""Reference CPI, lags and index ratios against published US Treasury values."""

from __future__ import annotations

import datetime as dt

import pytest

from spxa.rates.inflation import CPIFixings, InflationIndex, month_start

# CPI-U, not seasonally adjusted (BLS)
CPI_U = CPIFixings.from_pairs(
    "CPI-U",
    [
        ("2008-11-01", 212.425),
        ("2008-12-01", 210.228),
        ("2009-01-01", 211.143),
        ("2022-10-01", 298.012),
        ("2022-11-01", 297.711),
    ],
)
TIPS = InflationIndex("CPI-U", CPI_U, lag=3, interpolated=True, digits=5)


def test_reference_cpi_matches_treasury_values() -> None:
    # the first of a month carries the CPI of three months earlier
    assert TIPS.reference_cpi("2023-01-01") == 298.012
    assert TIPS.reference_cpi("2009-02-01") == 212.425
    assert TIPS.reference_cpi("2009-03-01") == 210.228
    assert TIPS.reference_cpi("2009-04-01") == 211.143
    # 2023-01-15: 298.012 * 17/31 + 297.711 * 14/31 = 297.87606 (TreasuryDirect)
    assert TIPS.reference_cpi("2023-01-15") == 297.87606
    assert TIPS.reference_cpi(dt.date(2023, 1, 15)) == pytest.approx(
        298.012 * 17 / 31 + 297.711 * 14 / 31, abs=5e-6
    )
    # February 2009 has 28 days: the 15th sits halfway
    assert TIPS.reference_cpi("2009-02-15") == round(212.425 + 0.5 * (210.228 - 212.425), 5)


def test_lags_interpolation_and_ratios() -> None:
    m0, m1, w = TIPS.reference_months("2024-02-15")
    assert (m0, m1) == (
        dt.date(2023, 11, 1),
        dt.date(2023, 12, 1),
    ) and w == 14 / 29  # leap February
    flat = InflationIndex("CPI-U", CPI_U, lag=3, interpolated=False)
    assert flat.reference_cpi("2023-01-31") == 298.012
    assert flat.reference_months("2023-01-31") == (dt.date(2022, 10, 1), dt.date(2022, 10, 1), 0.0)
    assert TIPS.index_ratio("2023-01-15", "2009-02-15") == round(297.87606 / 211.3265, 5)
    assert TIPS.is_fixed("2023-01-15") and not TIPS.is_fixed("2023-02-15")
    with pytest.raises(KeyError, match="2022-12"):
        TIPS.reference_cpi("2023-02-15")
    with pytest.raises(ValueError):
        InflationIndex("x", CPI_U, lag=-1)


def test_fixings_container() -> None:
    series = CPIFixings.from_series("HICPxT", "2023-11-17", [120.0, 121.0, 121.5])
    assert series["2023-11-30"] == 120.0 and series["2024-01-01"] == 121.5
    assert series.last_month == dt.date(2024, 1, 1)
    assert "2023-12-05" in series and "2024-02-01" not in series
    assert month_start("2024-02-29") == dt.date(2024, 2, 1)
    with pytest.raises(ValueError):
        series.add("2024-02-01", -1.0)
