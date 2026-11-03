"""Tests for spxa.market.parity."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market import (
    OptionChain,
    OptionQuote,
    OptionSurfaceQuotes,
    black_price,
    implied_forward,
    implied_forwards,
)

F, D, T = 4512.75, 0.9634, 0.75
K = np.arange(3800.0, 5201.0, 50.0)


def chain(
    f: float = F,
    d: float = D,
    k: np.ndarray = K,
    half_spread: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> OptionChain:
    x = np.log(k / f)
    vol = 0.18 - 0.12 * x + 0.2 * x * x
    c = black_price(f, k, T, vol, d, "call")
    p = black_price(f, k, T, vol, d, "put")
    rng = np.random.default_rng(seed)
    c = c + noise * rng.standard_normal(k.size)
    p = p + noise * rng.standard_normal(k.size)
    hs = half_spread
    return OptionChain.from_arrays(T, k, c - hs, c + hs, np.maximum(p - hs, 0), p + hs)


class TestExact:
    def test_recovers_forward_and_discount(self) -> None:
        fit = implied_forward(chain())
        assert fit.forward == pytest.approx(F, rel=1e-13)
        assert fit.discount == pytest.approx(D, rel=1e-12)
        assert fit.discount_fitted and fit.n == K.size and fit.rejected.size == 0
        assert fit.rmse < 1e-10

    def test_known_discount(self) -> None:
        fit = implied_forward(chain(), discount=D)
        assert fit.forward == pytest.approx(F, rel=1e-14)
        assert fit.discount == D and not fit.discount_fitted

    def test_single_pair_with_known_discount(self) -> None:
        ch = chain(k=np.array([4500.0]))
        assert implied_forward(ch, discount=D).forward == pytest.approx(F, rel=1e-14)
        with pytest.raises(ValueError, match="at least 2"):
            implied_forward(ch)

    def test_near_money_selection(self) -> None:
        fit = implied_forward(chain(), n_strikes=4)
        assert fit.n == 4
        assert set(fit.strikes) == {4450.0, 4500.0, 4550.0, 4600.0}  # |F - K| ranks
        assert fit.forward == pytest.approx(F, rel=1e-12)


class TestNoisy:
    def test_weights_by_spread(self) -> None:
        # wide, noisy quotes far from the money; tight, clean ones near it
        k = K
        ch = chain(noise=0.0)
        rng = np.random.default_rng(3)
        quotes = []
        for q in ch:
            dist = abs(q.strike - F) / 100.0
            hs = 0.05 + 0.5 * dist
            m = q.mid + hs * 0.8 * rng.standard_normal()
            quotes.append(OptionQuote(q.strike, q.option, max(m - hs, 0.0), m + hs))
        noisy = OptionChain(T, quotes)
        w = implied_forward(noisy, discount=D, outlier_threshold=None)
        e = implied_forward(noisy, discount=D, weights="equal", outlier_threshold=None)
        assert abs(w.forward - F) < abs(e.forward - F)
        assert abs(w.forward - F) < 0.1
        assert k.size == w.n

    def test_regression_error_shrinks_with_noise(self) -> None:
        errs = [abs(implied_forward(chain(noise=s, seed=1)).forward - F) for s in (1.0, 0.01)]
        assert errs[1] < errs[0] / 10 and errs[1] < 0.05

    def test_outlier_rejected(self) -> None:
        ch = chain(noise=0.01, seed=2)
        i = next(n for n, q in enumerate(ch.quotes) if q.strike == 4000.0 and q.is_call)
        q = ch.quotes[i]
        assert q.bid is not None and q.ask is not None
        ch.quotes[i] = OptionQuote(q.strike, "call", q.bid + 25.0, q.ask + 25.0)
        robust = implied_forward(ch)
        naive = implied_forward(ch, outlier_threshold=None)
        assert 4000.0 in robust.rejected
        assert abs(robust.forward - F) < 0.05 < abs(naive.forward - F)
        assert abs(robust.discount - D) < 1e-4

    def test_exact_data_rejects_nothing(self) -> None:
        assert implied_forward(chain()).rejected.size == 0


class TestErrors:
    def test_no_pairs(self) -> None:
        ch = OptionChain(T, [OptionQuote(4500.0, "call", 100.0, 101.0)])
        with pytest.raises(ValueError, match="pairs"):
            implied_forward(ch, discount=D)

    def test_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            implied_forward(chain(), weights="vega")
        with pytest.raises(ValueError, match="n_strikes"):
            implied_forward(chain(), n_strikes=0)

    def test_nonpositive_discount(self) -> None:
        qs = [
            OptionQuote(90.0, "call", 5.0, 5.0),
            OptionQuote(90.0, "put", 5.0, 5.0),
            OptionQuote(110.0, "call", 6.0, 6.0),
            OptionQuote(110.0, "put", 5.0, 5.0),
        ]
        with pytest.raises(ValueError, match="non-positive discount"):
            implied_forward(OptionChain(1.0, qs))

    def test_surface(self) -> None:
        a = chain(4500.0, 0.99)
        b = OptionChain(2 * T, list(chain(4600.0, 0.97).quotes))
        fits = implied_forwards(OptionSurfaceQuotes([a, b]))
        assert [f.forward for f in fits] == pytest.approx([4500.0, 4600.0], rel=1e-12)
        fits = implied_forwards(OptionSurfaceQuotes([a, b]), discounts=[0.99, None])
        assert not fits[0].discount_fitted and fits[1].discount_fitted
        with pytest.raises(ValueError, match="match"):
            implied_forwards(OptionSurfaceQuotes([a, b]), discounts=[0.99])
