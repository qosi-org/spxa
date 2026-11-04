"""Lifted Heston: a multi-factor approximation of rough Heston."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gamma

from spxa.pde.reference import heston_cf
from spxa.vol.heston import Heston
from spxa.vol.lifted_heston import LiftedHeston, lifted_weights
from spxa.vol.rough_heston import RoughHeston

PARAMS = (0.04, 0.04, 0.3, 0.3, -0.7, 0.1)
K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])


def test_one_factor_is_heston() -> None:
    m = LiftedHeston(0.04, 0.04, 1.5, 0.3, -0.7, 0.5, weights=[1.0], speeds=[0.0])
    U = np.array([0.5, 2.0, 5 - 0.5j, 10 - 0.5j])
    ref = np.array([heston_cf(1.0, 0.04, 1.5, 0.04, 0.3, -0.7)(u) for u in U])
    errs = [np.abs(m.cf(U, 1.0, n) - ref).max() for n in (100, 200, 400)]
    assert errs[2] < 1e-5 and errs[0] / errs[1] == pytest.approx(4.0, rel=0.05)
    t = np.array([0.1, 1.0, 3.0])
    h = Heston(0.02, 1.5, 0.04, 0.3, -0.7)
    m2 = LiftedHeston(0.02, 0.04, 1.5, 0.3, -0.7, 0.5, weights=[1.0], speeds=[0.0])
    np.testing.assert_allclose(m2.expected_variance(t), h.mean_variance(t), rtol=1e-5)


def test_kernel_approximation_improves_with_factors() -> None:
    t = np.geomspace(0.01, 1.0, 30)
    exact = t ** (0.6 - 1) / gamma(0.6)
    errs = []
    for n in (5, 20, 80):
        c, x = lifted_weights(n, 0.1)
        approx = np.exp(-np.outer(t, x)) @ c
        errs.append(np.abs(approx / exact - 1).max())
    assert errs[0] > errs[1] > errs[2]


def test_smile_converges_to_rough_heston() -> None:
    rough = RoughHeston(*PARAMS).implied_vol(K, 1.0, n_steps=400)
    gaps = [
        np.abs(LiftedHeston(*PARAMS, n=n).implied_vol(K, 1.0, n_steps=400) - rough).max()
        for n in (5, 20, 50)
    ]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 0.003
    # E[V_t] also approaches the Mittag-Leffler mean
    t = np.array([0.1, 1.0, 3.0])
    up = LiftedHeston(0.02, 0.04, 0.3, 0.3, -0.7, 0.1, n=50).expected_variance(t)
    np.testing.assert_allclose(
        up, RoughHeston(0.02, 0.04, 0.3, 0.3, -0.7, 0.1).expected_variance(t), rtol=0.03
    )
    with pytest.raises(ValueError):
        LiftedHeston(*PARAMS, weights=[1.0, 2.0], speeds=[0.0])
