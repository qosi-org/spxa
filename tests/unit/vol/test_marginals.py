"""Discrete marginals from call prices and the convex order."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market import black_price
from spxa.vol.marginals import DiscreteMarginal, marginal_from_calls

K = np.linspace(0.2, 3.0, 57)


def test_second_differences_reproduce_calls() -> None:
    c = np.asarray(black_price(1.0, K, 0.5, 0.2), float)
    mu = marginal_from_calls(K, c)
    assert mu.mean == pytest.approx(1.0, abs=1e-15)
    assert mu.weights.sum() == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(mu.call(K), c, atol=1e-14)
    # the law of a piecewise-linear call function: the call is linear between nodes
    mid = 0.5 * (K[1:] + K[:-1])
    np.testing.assert_allclose(mu.call(mid), 0.5 * (c[1:] + c[:-1]), atol=1e-14)
    assert mu.call(0.0) == pytest.approx(1.0, abs=1e-15) and mu.call(5.0) == 0.0


def test_truncated_tail_is_tilted_back_to_unit_mean() -> None:
    # at T = 1 the grid cuts 1e-9 of call value: the tilt fixes the mean, prices move by ~1e-8
    c = np.asarray(black_price(1.0, K, 1.0, 0.2), float)
    nu = marginal_from_calls(K, c)
    assert nu.mean == pytest.approx(1.0, abs=1e-15)
    assert np.abs(nu.call(K) - c).max() < 1e-7


def test_convex_order() -> None:
    c1 = np.asarray(black_price(1.0, K, 0.5, 0.2), float)
    c2 = np.asarray(black_price(1.0, K, 1.0, 0.2), float)
    mu, nu = marginal_from_calls(K, c1), marginal_from_calls(K, c2, floor=c1)
    assert mu.precedes(nu) and not nu.precedes(mu)
    # Dirac at the mean precedes everything; different means are never ordered
    assert DiscreteMarginal(np.array([1.0]), np.array([1.0])).precedes(nu)
    assert not DiscreteMarginal(np.array([1.1]), np.array([1.0])).precedes(nu)
    # a calendar violation (total variance 0.0169 < 0.02) is repaired by the floor
    low = np.asarray(black_price(1.0, K, 1.0, 0.13), float)
    assert marginal_from_calls(K, low).precedes(mu) and not mu.precedes(marginal_from_calls(K, low))
    assert mu.precedes(marginal_from_calls(K, low, floor=c1))


def test_validation() -> None:
    with pytest.raises(ValueError):
        DiscreteMarginal(np.array([1.0, 2.0]), np.array([0.5, 0.6]))
    with pytest.raises(ValueError):
        marginal_from_calls(np.array([0.5, 1.0, 2.0]), np.array([0.5, 0.1, 0.0]))  # uneven grid
    with pytest.raises(ValueError):
        marginal_from_calls(np.linspace(1.0, 2.0, 5), np.zeros(5))  # forward on the edge


def test_surface_marginals_from_ssvi() -> None:
    import spxa.vol as v
    from spxa.market import EquityForward
    from spxa.vol.conventions import normalised_call_from_total_variance
    from spxa.vol.marginals import surface_marginals

    fwd = EquityForward(100.0, 0.03, dividend_yield=0.01)
    surf = v.VolSurface(
        v.SSVI(-0.7, v.PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t), fwd
    )
    k = np.linspace(0.02, 3.02, 101)
    ms = surface_marginals(surf, [0.5, 1.0, 1.5], k)
    assert all(m.mean == pytest.approx(1.0, abs=1e-14) for m in ms)
    assert ms[0].precedes(ms[1]) and ms[1].precedes(ms[2])
    for m, T in zip(ms, (0.5, 1.0, 1.5)):
        c = normalised_call_from_total_variance(np.log(k), surf.total_variance(np.log(k), T))
        assert np.abs(m.call(k) - c).max() < 1e-6  # the truncated tails, re-tilted
    # the normalised calls are the surface's undiscounted prices over the forward
    F, df = fwd.forward(1.0), fwd.discount(1.0)
    assert surf.price(F * k[30], 1.0) / (F * df) == pytest.approx(ms[1].call(k[30]), abs=1e-8)
    # a grid that cuts the fat left tail loses exactly the truncated put value
    narrow = np.linspace(0.2, 2.6, 49)
    lost = (
        normalised_call_from_total_variance(np.log(0.2), surf.total_variance(np.log(0.2), 1.0))
        - 0.8
    )
    assert surface_marginals(surf, [0.5, 1.0], narrow)[1].call(0.2) == pytest.approx(0.8, abs=1e-15)
    assert lost == pytest.approx(4.76e-5, abs=1e-7)
    with pytest.raises(ValueError):
        surface_marginals(surf, [1.0, 0.5], k)
