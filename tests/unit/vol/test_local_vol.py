"""Dupire local volatility from implied-volatility surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.curves import EquityForward
from spxa.vol import SSVI, PowerLawPhi, durrleman_g
from spxa.vol.local_vol import LocalVolSurface, dupire_local_variance
from spxa.vol.surface import VolSurface

FWD = EquityForward(100.0, 0.03, dividend_yield=0.01)


class TermOnly:
    """No smile: w(k, T) = theta(T)."""

    def total_variance(self, k, T):  # type: ignore[no-untyped-def]
        return np.full(np.shape(k), 0.04 * T + 0.01 * T * T) + 0.0 * np.asarray(k)


def test_flat_surface() -> None:
    lv = LocalVolSurface(VolSurface.flat(0.23, FWD))
    K = np.array([50.0, 80.0, 100.0, 130.0, 200.0])
    for T in (0.1, 1.0, 5.0):
        assert lv.local_vol(K, T) == pytest.approx(np.full(5, 0.23), abs=1e-10)


def test_term_structure_gives_forward_variance() -> None:
    lv = LocalVolSurface(VolSurface(TermOnly(), FWD))
    k = np.linspace(-1.0, 1.0, 9)
    for T in (0.25, 1.0, 3.0):
        assert lv.local_variance_k(k, T) == pytest.approx(np.full(9, 0.04 + 0.02 * T), abs=1e-12)


SS = SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t)


def test_denominator_is_durrleman_g() -> None:
    lv = LocalVolSurface(VolSurface(SS, FWD))
    k = np.linspace(-1.5, 1.0, 26)
    T = 0.8
    sl = SS.slice(T)
    terms = lv.terms(k, T)
    assert terms["w_y"] == pytest.approx(sl.dw(k), abs=1e-10)
    assert terms["w_yy"] == pytest.approx(sl.d2w(k), abs=1e-8)
    assert terms["g"] == pytest.approx(durrleman_g(k, sl(k), sl.dw(k), sl.d2w(k)), abs=1e-8)


def test_ssvi_against_analytic_derivatives() -> None:
    # w(k, T) = W(k, theta(T)); dW/dtheta by differencing theta only, theta' = 0.04 + 0.006 T
    lv = LocalVolSurface(VolSurface(SS, FWD))
    k = np.linspace(-1.2, 0.8, 21)
    for T in (0.3, 1.0, 2.5):
        th = 0.04 * T + 0.003 * T * T
        e = 1e-7 * th
        dW = (SS.total_variance_theta(k, th + e) - SS.total_variance_theta(k, th - e)) / (2 * e)
        w_T = dW * (0.04 + 0.006 * T)
        sl = SS.slice(T)
        exact = dupire_local_variance(k, sl(k), sl.dw(k), sl.d2w(k), w_T)
        assert lv.local_variance_k(k, T) == pytest.approx(exact, rel=1e-7)


def test_negative_skew_puts_more_local_vol_on_the_downside() -> None:
    lv = LocalVolSurface(VolSurface(SS, FWD))
    T = 1.0
    F = float(FWD.forward(T))
    sig = lv.local_vol(np.array([0.7, 1.0, 1.3]) * F, T)
    assert sig[0] > sig[1] > sig[2]


class LinearSmile:
    """sigma(k) = 0.2 - 0.1 k at every expiry."""

    def total_variance(self, k, T):  # type: ignore[no-untyped-def]
        return (0.2 - 0.1 * np.asarray(k, float)) ** 2 * T


def test_short_expiry_local_skew_is_twice_implied() -> None:
    # with a time-independent smile, the T -> 0 local skew at the money is 2 x implied
    lv = LocalVolSurface(VolSurface(LinearSmile(), FWD))
    h = 1e-4
    for T, tol in ((1e-2, 2e-2), (1e-4, 2e-4)):
        skew = (lv.local_vol_k(h, T) - lv.local_vol_k(-h, T)) / (2 * h)
        assert skew == pytest.approx(-0.2, rel=tol)
    assert lv.local_vol_k(0.0, 1e-4) == pytest.approx(0.2, rel=1e-4)


def test_arbitrage_gives_nan() -> None:
    class Calendar:
        def total_variance(self, k, T):  # type: ignore[no-untyped-def]
            return np.full(np.shape(k), 0.04 * T - 0.02 * T * T) + 0.0 * np.asarray(k)

    lv = LocalVolSurface(VolSurface(Calendar(), FWD))
    assert np.isnan(lv.local_vol_k(0.0, 1.5))  # theta decreasing past T = 1
    assert lv.local_vol_k(0.0, 0.5) == pytest.approx(np.sqrt(0.04 - 0.02), abs=1e-12)
    assert np.isnan(dupire_local_variance(0.0, 0.04, 0.0, -3.0, 0.04))  # g < 0
    with pytest.raises(ValueError):
        lv.terms(0.0, 0.0)


def test_callable_for_pde_solvers() -> None:
    lv = LocalVolSurface(VolSurface(SS, FWD))
    S = np.array([80.0, 100.0, 120.0])
    assert lv(S, 0.5) == pytest.approx(lv.local_vol(S, 0.5))
    assert np.all(np.isfinite(lv(S, 0.0)))


def test_surface_convenience() -> None:
    surf = VolSurface(SS, FWD)
    K = np.array([80.0, 100.0, 125.0])
    assert surf.local_vol(K, 0.7) == pytest.approx(
        LocalVolSurface(surf).local_vol(K, 0.7), rel=1e-15
    )
    lv = surf.local_vol_surface(dk=2e-3)
    assert isinstance(lv, LocalVolSurface) and lv.dk == 2e-3
    assert VolSurface.flat(0.3, FWD).local_vol(100.0, 2.0) == pytest.approx(0.3, abs=1e-10)
