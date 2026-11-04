"""Local-volatility diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.curves import EquityForward
from spxa.vol import SSVI, LocalVolSurface, PowerLawPhi, VolSurface
from spxa.vol.lv_diagnostics import local_vol_diagnostics

FWD = EquityForward(100.0, 0.03, dividend_yield=0.01)
K = np.array([80.0, 100.0, 120.0])


def test_clean_surface_reprices_and_refines() -> None:
    lv = LocalVolSurface(
        VolSurface(SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t), FWD)
    )
    coarse = local_vol_diagnostics(lv, 100.0, 0.03, 0.01, K, [0.5, 1.0], n_space=301, n_time=150)
    fine = local_vol_diagnostics(lv, 100.0, 0.03, 0.01, K, [0.5, 1.0], n_space=601, n_time=300)
    assert fine.undefined_fraction == 0.0 and fine.first_undefined_time() is None
    assert fine.max_error < 1e-4
    assert fine.max_error < 0.5 * coarse.max_error
    T, k, e = fine.worst()
    assert T in (0.5, 1.0) and k in K and abs(e) == fine.max_error


def test_calendar_arbitrage_is_located() -> None:
    # theta(t) = 0.04 t - 0.012 t^2 peaks at t = 5/3: calendar arbitrage afterwards
    ss = SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t - 0.012 * t * t)
    lv = LocalVolSurface(VolSurface(ss, FWD))
    d = local_vol_diagnostics(lv, 100.0, 0.03, 0.01, K, [1.0, 2.0], n_t=40)
    assert d.first_undefined_time() == pytest.approx(1.7, abs=0.051)
    assert 0 < d.undefined_fraction < 0.5
    assert np.all(np.isnan(d.model_vols))
    with pytest.raises(ValueError):
        local_vol_diagnostics(lv, 100.0, 0.03, 0.01, K, [0.0])
