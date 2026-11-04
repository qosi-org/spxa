"""CMS coupons and options by static replication."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from spxa.market.black import black_price
from spxa.rates.cms import CMSReplication
from spxa.rates.sabr import SABR

S0, T, SIG, A0, P = 0.03, 5.0, 0.2, 4.3, 0.85


def _black(sig: float):  # type: ignore[no-untyped-def]
    return lambda K, o: float(black_price(S0, max(K, 1e-12), T, sig, 1.0, o))


@pytest.fixture(autouse=True)
def _quiet():  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_linear_tsr_is_exact_under_black() -> None:
    r = CMSReplication(S0, T, A0, P, _black(SIG), n=10, mapping="linear")
    # E^A[S (P/A0 + a (S - S0))] = P/A0 S0 + a S0^2 (exp(sigma^2 T) - 1)
    exact = A0 * (P / A0 * S0 + r.slope * S0**2 * (np.exp(SIG**2 * T) - 1))
    assert r.coupon() == pytest.approx(exact, rel=1e-8)
    for K in (0.02, 0.03, 0.045):
        assert r.caplet(K) - r.floorlet(K) == pytest.approx(r.coupon() - K * P, abs=1e-11)
    assert r.caplet(1e-9) == pytest.approx(r.coupon(), rel=1e-7)


def test_adjustment_vanishes_without_vol_and_grows_with_it() -> None:
    adj = [
        CMSReplication(S0, T, A0, P, _black(s), n=10).convexity_adjustment()
        for s in (1e-3, 0.1, 0.2, 0.3)
    ]
    assert abs(adj[0]) < 1e-7
    assert 0 < adj[1] < adj[2] < adj[3]
    lin = CMSReplication(S0, T, A0, P, _black(SIG), n=10, mapping="linear").convexity_adjustment()
    assert adj[2] == pytest.approx(lin, rel=0.1)  # the mappings agree to first order


def test_fat_tails_raise_the_adjustment() -> None:
    sabr = SABR(0.035, 0.5, -0.2, 0.5)
    atm = sabr.lognormal_vol(S0, S0, T)
    smile = CMSReplication(
        S0, T, A0, P, lambda K, o: float(sabr.price(S0, max(K, 1e-8), T, 1.0, o)), n=10
    )
    flat = CMSReplication(S0, T, A0, P, _black(float(atm)), n=10)
    assert smile.convexity_adjustment() > flat.convexity_adjustment()
    with pytest.raises(ValueError):
        CMSReplication(S0, T, A0, P, _black(SIG), n=10, mapping="cubic")
