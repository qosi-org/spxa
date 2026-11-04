"""Hull-White two-factor parameters and G2++."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.g2pp import G2PP
from spxa.rates.hw2f import HullWhite2F
from tests.unit.rates.test_hull_white import CURVE


def _instantaneous(g: G2PP) -> np.ndarray:
    """Covariance of (dr, du) per unit time implied by r = x + y + phi, u = (a - b) y."""
    J = np.array([[1.0, 1.0], [0.0, g.a - g.b]])
    S = np.array([[g.sigma**2, g.rho * g.sigma * g.eta], [g.rho * g.sigma * g.eta, g.eta**2]])
    return J @ S @ J.T


@pytest.mark.parametrize(
    "p", [HullWhite2F(0.6, 0.012, 0.05, 0.004, -0.3), HullWhite2F(0.05, 0.01, 0.4, 0.006, 0.5)]
)
def test_mapping_preserves_dynamics_and_round_trips(p: HullWhite2F) -> None:
    g = p.to_g2pp(CURVE)
    target = np.array(
        [
            [p.sigma1**2, p.rho_bar * p.sigma1 * p.sigma2],
            [p.rho_bar * p.sigma1 * p.sigma2, p.sigma2**2],
        ]
    )
    assert _instantaneous(g) == pytest.approx(target, rel=1e-12, abs=1e-20)
    back = HullWhite2F.from_g2pp(g)
    assert (back.a_bar, back.sigma1, back.b_bar, back.sigma2, back.rho_bar) == pytest.approx(
        (p.a_bar, p.sigma1, p.b_bar, p.sigma2, p.rho_bar), rel=1e-12
    )


def test_drift_of_r_is_hull_white() -> None:
    # -a x - b y = u - a r + a phi with u = (a - b) y and r = x + y + phi
    g = HullWhite2F(0.6, 0.012, 0.05, 0.004, -0.3).to_g2pp(CURVE)
    x, y, phi = 0.003, -0.002, 0.03
    r, u = x + y + phi, (g.a - g.b) * y
    assert -g.a * x - g.b * y == pytest.approx(u - g.a * r + g.a * phi, abs=1e-17)


def test_validation() -> None:
    with pytest.raises(ValueError):
        HullWhite2F(0.1, 0.01, 0.1, 0.01, 0.0)
    with pytest.raises(ValueError):
        HullWhite2F(0.1, 0.01, 0.2, 0.01, -1.0)
