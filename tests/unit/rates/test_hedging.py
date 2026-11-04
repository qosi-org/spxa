"""Hedging curve risk with par instruments."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates import ESTR, ois_swap
from spxa.rates.bootstrap import bootstrap_curve
from spxa.rates.interpolation import LogLinearDiscount
from spxa.rates.risk import hedge_notionals, par_hedge, quote_deltas
from tests.unit.rates.test_bootstrap import MARKET, REF, _estr_helpers

HS = _estr_helpers(MARKET)
LL = bootstrap_curve(HS, REF, LogLinearDiscount())
BOOK = ois_swap(REF, "6Y", 0.031, ESTR, notional=1e8)


def _pv(swap):  # type: ignore[no-untyped-def]
    return lambda c: swap.pv(c["discount"], c["discount"])


def test_par_swaps_hedge_exactly() -> None:
    # unit par swaps on every pillar: risk A_i e_i, so n_i = -delta_i / A_i
    pars = [h.swap.with_fixed_rate(h.quote) for h in HS[3:]]
    res = par_hedge(_pv(BOOK), LL, [_pv(s) for s in pars])
    book = quote_deltas(_pv(BOOK), LL).deltas
    annuities = np.array([s.annuity(LL.curve) for s in pars])
    # finite-difference noise in the hedges' off-diagonal risk leaks in at ~1e-9 of the book
    scale = np.max(np.abs(res.notionals))
    assert res.notionals == pytest.approx(-book[3:] / annuities, rel=1e-6, abs=1e-8 * scale)
    # with no deposit hedges, the deposit-quote risk is what remains
    assert res.residual_norm == pytest.approx(np.linalg.norm(book[:3]), rel=1e-6, abs=1e-6)


def test_hedged_book_is_flat_under_rebootstrap() -> None:
    pars = [h.swap.with_fixed_rate(h.quote) for h in HS[3:]]
    res = par_hedge(_pv(BOOK), LL, [_pv(s) for s in pars])
    n = res.notionals

    def total(curves) -> float:  # type: ignore[no-untyped-def]
        return _pv(BOOK)(curves) + sum(k * _pv(s)(curves) for k, s in zip(n, pars))

    left = quote_deltas(total, LL).deltas
    assert np.max(np.abs(left[3:])) < 1e-6 * np.max(np.abs(quote_deltas(_pv(BOOK), LL).deltas))


def test_least_squares_residual_is_orthogonal() -> None:
    rng = np.random.default_rng(0)
    d = rng.normal(size=8)
    H = rng.normal(size=(8, 3))
    w = rng.uniform(0.5, 2.0, size=8)
    res = hedge_notionals(d, H, w)
    assert H.T @ (w * res.residual) == pytest.approx(np.zeros(3), abs=1e-12)
    exact = hedge_notionals(d, rng.normal(size=(8, 8)))
    assert exact.residual_norm < 1e-12
    single = hedge_notionals([1.0, 2.0], [1.0, 1.0])
    assert single.notionals == pytest.approx([-1.5])


def test_validation() -> None:
    with pytest.raises(ValueError):
        hedge_notionals([1.0, 2.0], np.ones((3, 1)))
    with pytest.raises(ValueError):
        hedge_notionals([1.0, 2.0], np.ones((2, 1)), weights=[1.0, 0.0])
