"""SLV forward equation."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.vol.heston import Heston
from spxa.vol.slv import LeverageSurface, slv_grid
from spxa.vol.slv_fp import slv_forward

M = Heston(0.04, 1.5, 0.04, 0.5, -0.7)
K = np.array([-0.3, -0.1, 0.0, 0.1, 0.2])


def test_mass_forward_and_heston_smile() -> None:
    out = slv_forward(M, slv_grid(M, 1.0, 121, 61), 1.0, 100, record=[0.5])
    for t in (0.5, 1.0):
        d = out[t]
        assert d.mass == pytest.approx(1.0, abs=1e-12)
        assert d.forward() == pytest.approx(1.0, abs=1e-6)
        assert d.p.min() > -1e-6 * d.p.max()
        err = d.implied_vol(K) - M.implied_vol(np.exp(K), t)
        assert np.abs(err).max() < 5e-4
    with pytest.raises(KeyError):
        out[0.3]


def test_refinement_reduces_smile_error() -> None:
    ref = M.implied_vol(np.exp(K), 1.0)
    errs = [
        np.abs(slv_forward(M, slv_grid(M, 1.0, ny, nv), 1.0, nt).final.implied_vol(K) - ref).max()
        for ny, nv, nt in ((81, 41, 50), (161, 81, 200))
    ]
    assert errs[1] < 1.5e-4 and errs[1] < 0.6 * errs[0]


def test_constant_leverage_rescales_the_variance() -> None:
    # L = c turns v into c^2 v: Heston with (c^2 v0, kappa, c^2 theta, c xi, rho)
    c = 1.2
    scaled = Heston(c * c * M.v0, M.kappa, c * c * M.theta, c * M.xi, M.rho)
    g = slv_grid(scaled, 1.0, 121, 61)
    d = slv_forward(
        M, slv_grid(M, 1.0, 121, 61, vol=c * 0.2), 1.0, 100, leverage=LeverageSurface.constant(c)
    ).final
    assert g.y.hi == pytest.approx(d.grid.y.hi)
    np.testing.assert_allclose(d.implied_vol(K), scaled.implied_vol(np.exp(K), 1.0), atol=5e-4)
    assert d.conditional_variance()[60] == pytest.approx(
        float(
            np.sum(d.p[60] * d.grid.v.nodes * d.grid.v.cell_widths)
            / np.sum(d.p[60] * d.grid.v.cell_widths)
        )
    )


def _ssvi_local_vol():  # type: ignore[no-untyped-def]
    from spxa.market import EquityForward
    from spxa.vol import SSVI, LocalVolGrid, LocalVolSurface, PowerLawPhi, VolSurface

    ss = SSVI(-0.7, PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t)
    lv = LocalVolSurface(VolSurface(ss, EquityForward(100.0, 0.03, dividend_yield=0.01)))
    return ss, LocalVolGrid.from_surface(
        lv, np.linspace(-1.5, 1.5, 601), np.geomspace(1e-3, 1.0, 121)
    )


def test_calibrated_leverage_reprices_the_local_vol_smile() -> None:
    from spxa.vol.slv_fp import calibrate_leverage

    ss, lv = _ssvi_local_vol()
    K2 = np.array([-0.3, -0.15, 0.0, 0.1, 0.2])
    errs = {}
    for eta, (ny, nv, nt) in ((0.5, (121, 61, 100)), (1.0, (81, 41, 50)), (1.0, (161, 81, 200))):
        m = M.scaled(eta)
        cal = calibrate_leverage(lv, m, 1.0, slv_grid(m, 1.0, ny, nv, vol=0.2), nt, record=[0.5])
        errs[eta, ny] = [np.abs(cal.smile(K2, t) - ss.implied_vol(K2, t)).max() for t in (0.5, 1.0)]
        assert cal.leverage.values.shape == (nt, ny)
    assert max(errs[0.5, 121]) < 5e-4 and errs[0.5, 121][1] < 1.5e-4
    assert max(errs[1.0, 161]) < 2.5e-4 and max(errs[1.0, 161]) < 0.35 * max(errs[1.0, 81])


def test_heston_local_vol_is_a_fixed_point() -> None:
    # feeding the model its own Gyongy local variance E[v | y] gives L = 1
    from spxa.vol.slv_fp import calibrate_leverage

    g = slv_grid(M, 1.0, 81, 41)
    ts = np.linspace(0.0, 1.0, 41)
    own = slv_forward(M, g, 1.0, 40, record=list(ts[1:-1]))
    cond = {0.0: np.full(g.y.n, M.v0)} | {
        t: d.conditional_variance(1e-6) for t, d in own.densities.items()
    }
    keys = np.array(sorted(cond))

    def lv(y: np.ndarray, t: float) -> np.ndarray:
        return np.sqrt(cond[keys[np.searchsorted(keys, t + 1e-12) - 1]])

    cal = calibrate_leverage(lv, M, 1.0, g, 40)
    np.testing.assert_allclose(cal.leverage.values, 1.0, atol=1e-9)
    np.testing.assert_allclose(cal.forward.final.p, own.final.p, atol=1e-9 * own.final.p.max())
