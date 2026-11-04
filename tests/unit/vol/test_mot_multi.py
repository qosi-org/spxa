"""Three-marginal MOT for path-dependent claims."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market import black_price
from spxa.vol.marginals import marginal_from_calls
from spxa.vol.mot import mot_solve
from spxa.vol.mot_multi import mot_multi

K = np.linspace(0.3, 2.3, 21)


def marginals() -> list:
    out = []
    for T in (0.5, 1.0, 1.5):
        c = np.asarray(black_price(1.0, K, T, 0.2), float)
        out.append(marginal_from_calls(K, c, floor=out[-1].call(K) if out else None))
    return out


MS = marginals()


def test_reduces_to_two_marginals() -> None:
    for sense in ("max", "min"):
        # consecutive dates: identical to the two-marginal problem
        v3, _ = mot_multi(MS, lambda a, b, c: np.abs(b - a), sense)
        assert v3 == pytest.approx(
            mot_solve(MS[0], MS[1], lambda x, y: np.abs(y - x), sense).value, abs=1e-9
        )
        # first and last dates: the intermediate smile can only tighten
        w3, _ = mot_multi(MS, lambda a, b, c: np.abs(c - a), sense)
        w2 = mot_solve(MS[0], MS[2], lambda x, y: np.abs(y - x), sense).value
        assert (w3 <= w2 + 1e-9) if sense == "max" else (w3 >= w2 - 1e-9)


def test_asian_bounds_contain_black_scholes() -> None:
    def asian(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.maximum((a + b + c) / 3 - 1.0, 0.0)

    lo, _ = mot_multi(MS, asian, "min")
    hi, plan = mot_multi(MS, asian, "max")
    X1, X2, X3 = np.meshgrid(*[m.points for m in MS], indexing="ij")
    # marginal and martingale constraints hold to the LP feasibility tolerance (1e-10);
    # tail weights below it (here 6e-12) may come back as exactly zero
    for t, m in enumerate(MS):
        np.testing.assert_allclose(
            plan.sum(axis=tuple(a for a in range(3) if a != t)), m.weights, atol=1e-10
        )
    assert np.abs((plan * (X2 - X1)).sum(axis=(1, 2))).max() < 1e-10
    assert np.abs((plan * (X3 - X2)).sum(axis=2)).max() < 1e-10
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200_000, 3)) * 0.2 * np.sqrt(0.5)
    S = np.exp(np.cumsum(z - 0.01, axis=1))
    bs = np.maximum(S.mean(axis=1) - 1.0, 0.0)
    assert lo + 3 * bs.std() / np.sqrt(bs.size) < bs.mean() < hi - 3 * bs.std() / np.sqrt(bs.size)
    assert (round(lo, 4), round(hi, 4)) == (0.0634, 0.0778)


def test_validation() -> None:
    with pytest.raises(ValueError):
        mot_multi(MS[:1], lambda a: a)
    with pytest.raises(ValueError):
        mot_multi([MS[1], MS[0]], lambda a, b: np.abs(b - a))
    with pytest.raises(ValueError):
        mot_multi(MS, lambda a, b, c: a, "sup")
