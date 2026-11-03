"""Every numerical statement in docs/theory/market.md, checked against the code."""

from __future__ import annotations

import datetime as dt
import math
import re
from pathlib import Path

import numpy as np
import pytest

import spxa.market as m

DOC = Path(__file__).resolve().parents[2] / "docs" / "theory" / "market.md"


def test_table_names_exist() -> None:
    text = DOC.read_text(encoding="utf-8")
    table = text.split("---", 1)[0]
    names = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", table))
    names -= {"datetime", "date", "spxa", "rates", "vol"}
    missing = sorted(n for n in names if not hasattr(m, n))
    assert not missing, missing


def test_add_months_examples() -> None:
    assert m.add_months("2023-01-31", 1) == dt.date(2023, 2, 28)
    assert m.add_months("2024-01-31", 1) == dt.date(2024, 2, 29)
    assert m.add_months("2023-02-28", 1, end_of_month_rule=True) == dt.date(2023, 3, 31)
    assert m.add_months("2023-02-28", 1) == dt.date(2023, 3, 28)
    for t in ("3M", "2Y", "1W", "10D"):
        m.Tenor(t)


def test_isda_examples() -> None:
    reg = ("2003-11-01", "2004-05-01")
    isda = m.year_fraction(*reg, "ACT/ACT ISDA")
    assert isda == pytest.approx(61 / 365 + 121 / 366, abs=1e-15)
    assert round(isda, 12) == 0.497724380567
    assert m.year_fraction(
        *reg, "ACT/ACT ICMA", ref_start=reg[0], ref_end=reg[1], frequency=2
    ) == pytest.approx(0.5, abs=1e-15)
    long_first = ("2002-08-15", "2003-07-15")
    assert round(m.year_fraction(*long_first, "ACT/ACT ISDA"), 12) == 0.915068493151


def test_thirty_360_month_end() -> None:
    a, b = dt.date(2007, 2, 28), dt.date(2007, 3, 31)
    days = {
        c: round(m.year_fraction(a, b, c) * 360)
        for c in ("30/360", "30/360 US", "30E/360", "30E/360 ISDA")
    }
    assert days == {"30/360": 33, "30/360 US": 30, "30E/360": 32, "30E/360 ISDA": 30}


def test_target_2024() -> None:
    cal = m.get_calendar("TARGET")
    expected = {
        dt.date(2024, 1, 1),
        dt.date(2024, 3, 29),
        dt.date(2024, 4, 1),
        dt.date(2024, 5, 1),
        dt.date(2024, 12, 25),
        dt.date(2024, 12, 26),
    }
    assert set(cal.holidays("2024-01-01", "2024-12-31")) == expected
    for d in expected:
        assert not cal.is_business_day(d)
    assert cal.business_days_between("2024-01-01", "2025-01-01") == 256


def test_option_formula_claims() -> None:
    F, K, T, s = 100.0, 110.0, 0.7, 0.25
    g = m.black_greeks(F, K, T, s)
    sd = s * math.sqrt(T)
    d1 = math.log(F / K) / sd + 0.5 * sd
    d2 = d1 - sd
    assert g["volga"] == pytest.approx(g["vega"] * d1 * d2 / s, rel=1e-12)
    P, sn = 0.95, 7.0
    assert m.bachelier_price(F, F, T, sn, P) == pytest.approx(
        P * sn * math.sqrt(T / (2 * math.pi)), rel=1e-15
    )


def test_implied_vol_claims() -> None:
    # ATM closed forms
    F, T, P = 100.0, 2.0, 0.9
    for sig in (1e-3, 0.2, 3.0):
        v = m.black_price(F, F, T, sig, P)
        assert m.implied_black_volatility(v, F, F, T, P) == pytest.approx(sig, rel=1e-13)
    vn = m.bachelier_price(F, F, T, 12.0, P)
    assert m.implied_normal_volatility(vn, F, F, T, P) * math.sqrt(T) == pytest.approx(
        math.sqrt(2 * math.pi) * vn / P, rel=1e-14
    )
    # normalised put-call symmetry b(x) - b(-x) = e^{x/2} - e^{-x/2}
    x = np.linspace(-3, 3, 13)
    lhs = m.normalised_black_call(x, 0.4) - m.normalised_black_call(-x, 0.4)
    assert np.max(np.abs(lhs - (np.exp(x / 2) - np.exp(-x / 2)))) < 1e-14
    # out-of-bounds prices
    assert math.isnan(m.implied_black_volatility(150.0, F, 90.0, 1.0))


def test_parity_claim() -> None:
    F, D, T = 4512.75, 0.9634, 0.75
    k = np.arange(3800.0, 5201.0, 50.0)
    vol = 0.2
    c = m.black_price(F, k, T, vol, D, "call")
    p = m.black_price(F, k, T, vol, D, "put")
    fit = m.implied_forward(m.OptionChain.from_arrays(T, k, c, c, p, p))
    assert abs(fit.forward / F - 1) < 1e-13
    assert abs(fit.discount / D - 1) < 1e-12
