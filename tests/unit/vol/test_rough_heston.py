"""Rough Heston characteristic function."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.pde.reference import heston_cf
from spxa.rough import FractionalRiccatiSolver
from spxa.vol.heston import Heston
from spxa.vol.rough_heston import RoughHeston

U = np.array([0.5, 2.0, 5.0 - 0.5j, 10.0 - 0.5j, 25.0 - 0.5j])
ROUGH = RoughHeston(0.04, 0.04, 0.3, 0.3, -0.7, 0.1)


def test_half_hurst_is_heston() -> None:
    m = RoughHeston(0.04, 0.04, 1.5, 0.3, -0.7, 0.5)
    ref = heston_cf(1.0, 0.04, 1.5, 0.04, 0.3, -0.7)
    errs = [np.abs(m.cf(U, 1.0, n) - np.array([ref(u) for u in U])).max() for n in (100, 200, 400)]
    assert errs[-1] < 2e-6
    # second order: halving the step divides the error by about four
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.05)
    assert errs[1] / errs[2] == pytest.approx(4.0, rel=0.05)
    h = m.heston_limit()
    assert isinstance(h, Heston) and (h.kappa, h.xi) == (1.5, 0.3)


def test_matches_the_scalar_solver_and_converges() -> None:
    scalar = FractionalRiccatiSolver(0.1, 0.3, -0.7, 0.3)
    for u in (0.7, 3.1):
        assert ROUGH.cf(np.array([u]), 1.0, 100)[0] == pytest.approx(
            scalar.rough_heston_char_func(u, 1.0, 0.04, 0.04, 100), abs=1e-14
        )
    ref = ROUGH.cf(U, 1.0, 1600)
    errs = [np.abs(ROUGH.cf(U, 1.0, n) - ref).max() for n in (100, 200, 400)]
    # order 1 + alpha = 1.6 for H = 0.1
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all((orders > 1.4) & (orders < 2.1))
    assert errs[-1] < 1e-4


def test_martingale_and_expected_variance() -> None:
    # u = -i: E[F_T / F_0] = 1 exactly (the Riccati solution vanishes)
    assert ROUGH.cf(np.array([-1j]), 2.0, 50)[0] == pytest.approx(1.0, abs=1e-15)
    assert abs(ROUGH.cf(np.array([0.0]), 2.0, 50)[0] - 1.0) < 1e-15
    # E[V_t] by Mittag-Leffler; reduces to Heston's mean at H = 1/2
    m = RoughHeston(0.02, 0.04, 1.5, 0.3, -0.7, 0.5)
    t = np.array([0.1, 1.0, 5.0])
    np.testing.assert_allclose(
        m.expected_variance(t), Heston(0.02, 1.5, 0.04, 0.3, -0.7).mean_variance(t), rtol=1e-13
    )
    up = RoughHeston(0.02, 0.04, 0.3, 0.3, -0.7, 0.1).expected_variance(
        np.array([0.0, 0.1, 1.0, 5.0])
    )
    assert up[0] == 0.02 and np.all(np.diff(up) > 0) and up[-1] < 0.04
    with pytest.raises(ValueError):
        RoughHeston(0.04, 0.04, 0.3, 0.3, -0.7, 0.7)
    with pytest.raises(ValueError):
        ROUGH.heston_limit()
