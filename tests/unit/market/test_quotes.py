"""Tests for spxa.market.quotes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.market import (
    OptionChain,
    OptionQuote,
    OptionSurfaceQuotes,
    bachelier_price,
    black_price,
)

F, DF, T = 100.0, 0.98, 0.5
STRIKES = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])


def smile(k: np.ndarray) -> np.ndarray:
    x = np.log(k / F)
    return 0.2 - 0.1 * x + 0.3 * x * x


def make_chain(half_spread: float = 0.05) -> OptionChain:
    vols = smile(STRIKES)
    c = black_price(F, STRIKES, T, vols, DF, "call")
    p = black_price(F, STRIKES, T, vols, DF, "put")
    return OptionChain.from_arrays(
        T,
        STRIKES,
        call_bid=c - half_spread,
        call_ask=c + half_spread,
        put_bid=np.maximum(p - half_spread, 0.0),
        put_ask=p + half_spread,
        forward=F,
        discount=DF,
    )


class TestOptionQuote:
    def test_mid_spread(self) -> None:
        q = OptionQuote(100.0, "C", 1.0, 1.2)
        assert q.option == "call" and q.is_call
        assert q.mid == pytest.approx(1.1, abs=1e-15)
        assert q.spread == pytest.approx(0.2, abs=1e-15)
        assert q.relative_spread == pytest.approx(0.2 / 1.1, rel=1e-14)

    def test_one_sided_falls_back_to_last(self) -> None:
        q = OptionQuote(100.0, "put", bid=None, ask=2.0, last=1.9)
        assert not q.two_sided
        assert q.mid == 1.9
        assert q.spread is None and q.relative_spread is None
        assert OptionQuote(100.0, "p", ask=2.0).mid is None

    def test_zero_bid_is_quoted(self) -> None:
        q = OptionQuote(200.0, "call", 0.0, 0.05)
        assert q.two_sided and q.mid == pytest.approx(0.025, abs=1e-15)
        assert OptionQuote(200.0, "call", 0.0, 0.0).relative_spread == math.inf

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"option": "straddle"},
            {"bid": 2.0, "ask": 1.0},
            {"bid": -0.1},
            {"ask": float("nan")},
            {"strike": float("inf")},
        ],
    )
    def test_validation(self, kwargs: dict) -> None:
        base = {"strike": 100.0, "option": "call", "bid": 1.0, "ask": 1.1}
        base.update(kwargs)
        with pytest.raises(ValueError):
            OptionQuote(**base)


class TestOptionChain:
    def test_from_arrays_and_sorting(self) -> None:
        ch = make_chain()
        assert len(ch) == 14
        assert np.array_equal(ch.strikes, STRIKES)
        assert len(ch.calls()) == len(ch.puts()) == 7
        ks = [q.strike for q in ch]
        assert ks == sorted(ks)

    def test_nan_marks_missing(self) -> None:
        ch = OptionChain.from_arrays(
            1.0, [90.0, 100.0], call_bid=[np.nan, 1.0], call_ask=[np.nan, 1.2]
        )
        assert len(ch) == 1 and ch.quotes[0].strike == 100.0
        ch2 = OptionChain.from_arrays(1.0, [90.0, 100.0], call_ask=[3.0, np.nan])
        assert len(ch2) == 1 and ch2.quotes[0].bid is None

    def test_expiry_from_dates(self) -> None:
        ch = OptionChain("2025-03-21", [], valuation_date="2024-03-21")
        assert ch.T == pytest.approx(365 / 365, abs=1e-15)
        ch = OptionChain("2024-09-20", [], valuation_date="2024-03-20", day_count="ACT/360")
        assert ch.T == pytest.approx(184 / 360, abs=1e-15)
        with pytest.raises(ValueError, match="valuation_date"):
            OptionChain("2025-03-21", [])
        with pytest.raises(ValueError, match="after"):
            OptionChain(0.0, [])

    def test_duplicates_rejected(self) -> None:
        q = OptionQuote(100.0, "call", 1.0, 1.1)
        with pytest.raises(ValueError, match="duplicate"):
            OptionChain(1.0, [q, q])

    def test_mids_and_pairs(self) -> None:
        ch = make_chain()
        k, m = ch.mids("call")
        exact = black_price(F, STRIKES, T, smile(STRIKES), DF, "call")
        assert np.array_equal(k, STRIKES)
        assert np.max(np.abs(m - exact)) < 1e-13
        kk, c, p = ch.pairs()
        assert np.array_equal(kk, STRIKES)
        # put-call parity of the mids (symmetric spreads; puts not clipped here)
        assert np.max(np.abs(c - p - DF * (F - kk))) < 1e-12

    def test_pairs_only_common_strikes(self) -> None:
        qs = [
            OptionQuote(90.0, "call", 11.0, 11.2),
            OptionQuote(100.0, "call", 4.0, 4.2),
            OptionQuote(100.0, "put", 4.0, 4.2),
            OptionQuote(110.0, "put", 11.0, 11.2),
        ]
        k, c, p = OptionChain(1.0, qs).pairs()
        assert k.tolist() == [100.0] and c[0] == p[0] == pytest.approx(4.1, abs=1e-15)

    def test_filter(self) -> None:
        ch = make_chain()
        assert len(ch.filter(strike_range=(90.0, 110.0))) == 10
        # the deep OTM options have tiny prices -> large relative spreads
        tight = ch.filter(max_relative_spread=0.05)
        assert all(q.relative_spread <= 0.05 for q in tight)
        assert 0 < len(tight) < len(ch)
        assert all(q.bid >= 1.0 for q in ch.filter(min_bid=1.0))
        assert len(ch.filter(min_open_interest=1.0)) == 0
        one = OptionChain(
            1.0, [OptionQuote(100.0, "call", ask=1.0), OptionQuote(100.0, "put", 1.0, 1.1)]
        )
        assert len(one.filter(two_sided=True)) == 1
        # original untouched, T preserved
        assert len(ch) == 14 and tight.T == ch.T

    def test_otm(self) -> None:
        o = make_chain().otm()
        assert all((q.is_call and q.strike >= F) or (not q.is_call and q.strike < F) for q in o)
        assert len(o) == 7
        assert np.array_equal(o.strikes, STRIKES)
        with pytest.raises(ValueError, match="forward"):
            OptionChain(1.0, []).otm()

    def test_implied_vols_recover_smile(self) -> None:
        k, v, is_call = make_chain(half_spread=0.0).implied_vols()
        assert np.array_equal(k, STRIKES)
        assert np.max(np.abs(v - smile(k))) < 1e-12
        assert np.array_equal(is_call, k >= F)

    def test_implied_vols_all_quotes_agree(self) -> None:
        k, v, c = make_chain(half_spread=0.0).implied_vols(otm_only=False)
        assert k.size == 14
        assert np.max(np.abs(v - smile(k))) < 1e-11

    def test_bid_ask_vols_bracket_mid(self) -> None:
        ch = make_chain(0.02)
        _, vb, _ = ch.implied_vols(which="bid")
        _, vm, _ = ch.implied_vols(which="mid")
        _, va, _ = ch.implied_vols(which="ask")
        ok = np.isfinite(vb)
        assert np.all(vb[ok] < vm[ok]) and np.all(vm < va)

    def test_normal_vols(self) -> None:
        sn = 20.0
        c = bachelier_price(F, STRIKES, T, sn, DF, "call")
        p = bachelier_price(F, STRIKES, T, sn, DF, "put")
        ch = OptionChain.from_arrays(T, STRIKES, c, c, p, p, forward=F, discount=DF)
        _, v, _ = ch.implied_vols(model="normal")
        assert np.max(np.abs(v - sn)) < 1e-11

    def test_arbitrageable_quote_gives_nan(self) -> None:
        # an in-the-money call quoted below its discounted intrinsic value (20)
        ch = OptionChain(1.0, [OptionQuote(80.0, "call", 10.0, 10.0)], forward=F, discount=1.0)
        _, v, _ = ch.implied_vols(otm_only=False)
        assert math.isnan(v[0])

    def test_argument_errors(self) -> None:
        ch = make_chain()
        with pytest.raises(ValueError, match="model"):
            ch.implied_vols(model="sabr")
        with pytest.raises(ValueError, match="which"):
            ch.implied_vols(which="last")
        k, v, c = OptionChain(1.0, [], forward=F).implied_vols()
        assert k.size == v.size == c.size == 0

    def test_to_market_slice(self) -> None:
        sl = make_chain(0.0).to_market_slice()
        assert sl.maturity == T and sl.forward == F and sl.discount == DF
        assert np.max(np.abs(np.asarray(sl.quotes) - smile(STRIKES))) < 1e-12


class TestOptionSurfaceQuotes:
    def test_sorted_and_indexing(self) -> None:
        a = OptionChain(1.0, [OptionQuote(100.0, "call", 1, 2)])
        b = OptionChain(0.25, [OptionQuote(100.0, "call", 1, 2)])
        s = OptionSurfaceQuotes([a, b])
        assert s.expiries.tolist() == [0.25, 1.0]
        assert s[0] is b and len(s) == 2 and list(s) == [b, a]
        with pytest.raises(ValueError, match="duplicate"):
            OptionSurfaceQuotes([a, a])

    def test_from_records_and_filter(self) -> None:
        recs = [
            {"expiry": "2024-12-20", "strike": 100, "option": "call", "bid": 5.0, "ask": 5.2},
            {"expiry": "2024-12-20", "strike": 100, "option": "put", "bid": 4.0, "ask": 4.2},
            {"expiry": "2024-09-20", "strike": 100, "option": "C", "bid": 3.0, "ask": 3.1},
            {"expiry": "2024-09-20", "strike": 150, "option": "C", "bid": 0.0, "ask": 0.05},
        ]
        s = OptionSurfaceQuotes.from_records(recs, spot=100.0, valuation_date="2024-06-20")
        assert len(s) == 2
        assert s.expiries.tolist() == pytest.approx([92 / 365, 183 / 365], abs=1e-15)
        assert s[0].spot == 100.0 and len(s[0]) == 2
        f = s.filter(min_bid=4.5)
        assert len(f) == 1 and f[0].T == pytest.approx(183 / 365, abs=1e-15)
