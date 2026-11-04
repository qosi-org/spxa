"""Entropic MOT converges to the linear-programming bounds."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.vol.mot import mot_solve
from spxa.vol.mot_sinkhorn import mot_sinkhorn
from tests.unit.vol.test_mot import MU, NU, straddle


def test_converges_to_the_lp_bounds_from_inside() -> None:
    X, Y = np.meshgrid(MU.points, NU.points, indexing="ij")
    for sense in ("max", "min"):
        lp = mot_solve(MU, NU, straddle, sense).value
        gaps = []
        for eps in (1e-2, 3e-3):
            r = mot_sinkhorn(MU, NU, straddle, sense, eps=eps)
            assert r.marginal_error < 1e-9 and r.martingale_error < 1e-12
            np.testing.assert_allclose(r.plan.sum(1), MU.weights, atol=1e-9)
            assert np.abs((r.plan * (Y - X)).sum(1)).max() < 1e-12
            gaps.append(lp - r.value if sense == "max" else r.value - lp)
        # a feasible plan never beats the LP; the gap shrinks faster than eps
        assert 0 < gaps[1] < gaps[0] / 5 and gaps[1] < 3e-4


def test_validation() -> None:
    with pytest.raises(ValueError):
        mot_sinkhorn(MU, NU, straddle, "sup")
    with pytest.raises(ValueError):
        mot_sinkhorn(MU, NU, straddle, eps=0.0)
