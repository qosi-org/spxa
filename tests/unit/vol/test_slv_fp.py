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
