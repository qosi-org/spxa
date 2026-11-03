"""Quotes-to-surface pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.market.black import black_price
from spxa.market.quotes import OptionChain, OptionSurfaceQuotes
from spxa.vol import SSVI, PowerLawPhi
from spxa.vol.build import ForwardTermStructure, build_vol_surface, chain_total_variance

S, R, Q = 100.0, 0.03, 0.015
TRUE = SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t)
EXP = [0.1, 0.25, 0.5, 1.0, 2.0]


def _chain(T: float, half_spread: float = 0.0, known: bool = False) -> OptionChain:
    F, D = S * math.exp((R - Q) * T), math.exp(-R * T)
    K = np.arange(60.0, 145.0, 5.0)
    k = np.log(K / F)
    keep = np.abs(k) < 2.5 * math.sqrt(0.04 * T + 0.003 * T * T) + 0.05
    K, k = K[keep], k[keep]
    vol = TRUE.implied_vol(k, T)
    c = black_price(F, K, T, vol, D, "call")
    p = black_price(F, K, T, vol, D, "put")
    hs_c = half_spread * c + np.minimum(1e-4, c / 2)
    hs_p = half_spread * p + np.minimum(1e-4, p / 2)
    kw = {"forward": F, "discount": D} if known else {}
    return OptionChain.from_arrays(T, K, c - hs_c, c + hs_c, p - hs_p, p + hs_p, **kw)


QUOTES = OptionSurfaceQuotes([_chain(T) for T in EXP], spot=S)


def test_forward_term_structure() -> None:
    T = np.array([0.5, 1.0, 2.0])
    F = 100 * np.exp(0.01 * T)
    D = np.exp(-0.03 * T)
    fts = ForwardTermStructure(T, F, D, spot=100.0)
    assert fts.forward(T) == pytest.approx(F, rel=1e-15)
    assert fts.forward(0.25) == pytest.approx(100 * math.exp(0.0025), rel=1e-14)
    assert fts.forward(3.0) == pytest.approx(100 * math.exp(0.03), rel=1e-14)
    assert fts.discount(np.array([0.0, 0.25, 3.0])) == pytest.approx(
        np.exp(-0.03 * np.array([0, 0.25, 3]))
    )
    flat = ForwardTermStructure(T, F, D)
    assert flat.forward(0.1) == pytest.approx(F[0], rel=1e-15)  # flat before the first expiry
    with pytest.raises(ValueError):
        ForwardTermStructure([1.0, 0.5], [1, 1], [1, 1])
    with pytest.raises(ValueError):
        fts.forward(-1.0)


def test_parity_forwards_and_ssvi_recovery() -> None:
    b = build_vol_surface(QUOTES, method="ssvi")
    T = np.array(EXP)
    assert b.expiries == pytest.approx(T)
    assert b.forwards == pytest.approx(S * np.exp((R - Q) * T), rel=1e-10)
    assert b.discounts == pytest.approx(np.exp(-R * T), rel=1e-10)
    assert all(p is not None for p in b.parity)
    s = b.fit.surface
    assert (s.rho, s.phi.eta, s.phi.gamma) == pytest.approx((-0.7, 1.1, 0.4), abs=1e-5)
    assert np.max(b.vol_rmse) < 1e-6


@pytest.mark.parametrize("method", ["svi", "essvi"])
def test_svi_and_essvi_fit_and_reprice(method: str) -> None:
    b = build_vol_surface(QUOTES, method=method)
    assert np.max(b.vol_rmse) < (5e-4 if method == "svi" else 2e-4)
    surf = b.surface
    for T in (0.25, 1.0):
        F, D = S * math.exp((R - Q) * T), math.exp(-R * T)
        assert surf.forward(T) == pytest.approx(F, rel=1e-10)
        K = np.array([85.0, 100.0, 115.0])
        ref = black_price(F, K, T, TRUE.implied_vol(np.log(K / F), T), D)
        assert surf.price(K, T) == pytest.approx(ref, rel=2e-3)
    assert surf.atm_vol(0.75) == pytest.approx(TRUE.implied_vol(0.0, 0.75), abs=1e-3)


def test_weights() -> None:
    ch = _chain(1.0, half_spread=0.01)
    F, D = S * math.exp(R - Q), math.exp(-R)
    k, w, u = chain_total_variance(ch, F, D)
    assert np.all(u == 1.0)
    _, _, v = chain_total_variance(ch, F, D, "vega")
    d1 = -k / np.sqrt(w) + np.sqrt(w) / 2
    ref = np.exp(-d1 * d1) / (2 * math.pi * 4 * w)  # (dc/dw)^2
    assert v == pytest.approx(ref / np.mean(ref), rel=1e-13)
    _, _, sp = chain_total_variance(ch, F, D, "spread")
    assert np.mean(sp) == pytest.approx(1.0)
    noisy = OptionSurfaceQuotes([_chain(T, 0.01) for T in EXP], spot=S)
    for wname in ("vega", "spread"):
        b = build_vol_surface(noisy, method="svi", weights=wname)
        assert np.max(b.vol_rmse) < 1e-3
    with pytest.raises(ValueError):
        build_vol_surface(QUOTES, method="essvi", weights="vega")
    with pytest.raises(ValueError):
        chain_total_variance(ch, F, D, "bogus")
    one_sided = OptionChain.from_arrays(
        1.0, [90.0, 100.0], call_bid=[12.0, 7.0], put_bid=[3.0, 6.0]
    )
    assert chain_total_variance(one_sided, F, D, "vega")[0].size == 0
    locked = OptionChain.from_arrays(
        1.0, [90.0, 110.0], [12.0, 3.0], [12.0, 3.0], [3.0, 12.0], [3.0, 12.0]
    )
    with pytest.raises(ValueError):
        chain_total_variance(locked, F, D, "spread")


def test_known_forwards_skip_parity_and_min_quotes() -> None:
    q = OptionSurfaceQuotes([_chain(T, known=True) for T in EXP])
    b = build_vol_surface(q, method="ssvi")
    assert all(p is None for p in b.parity)
    assert b.forwards[2] == S * math.exp((R - Q) * 0.5)
    short = build_vol_surface(QUOTES, method="ssvi", min_quotes=12)
    assert short.expiries.size < len(EXP)
    with pytest.raises(ValueError):
        build_vol_surface(QUOTES, method="ssvi", min_quotes=100)
    with pytest.raises(ValueError):
        build_vol_surface(QUOTES, method="sabr")
