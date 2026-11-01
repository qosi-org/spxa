"""Exactness tests: ADI schemes on a convection-diffusion problem with a mixed term.

u = exp(alpha x + beta y + gamma tau) solves the constant-coefficient
equation exactly when gamma equals the symbol of the operator, which gives
exact Dirichlet and Neumann data on a stretched grid.
"""

import numpy as np
import pytest

from spxa.pde.adi import ADISolver, Dirichlet2D, DiscreteOperator2D, Grid2D, LinearPDE2D, Neumann2D
from spxa.pde.grid import Grid1D

pytestmark = pytest.mark.exactness

AL, BE = 0.7, -0.4
COEF = dict(axx=0.5, axy=0.3, ayy=0.4, bx=0.2, by=-0.1, c=-0.05)


def gamma(coef):
    return (
        coef["axx"] * AL**2
        + coef["axy"] * AL * BE
        + coef["ayy"] * BE**2
        + coef["bx"] * AL
        + coef["by"] * BE
        + coef["c"]
    )


def solve(n, n_time, scheme, neumann=False, coef=COEF, damping=0):
    g = gamma(coef)

    def ex(X, Y, t):
        return np.exp(AL * X + BE * Y + g * t)

    grid = Grid2D(Grid1D.sinh(0.0, 1.0, n, 0.4, 0.3), Grid1D.sinh(0.0, 1.0, n, 0.6, 0.3))
    edges = {
        "x_lo": Dirichlet2D(lambda y, t: ex(0.0, y, t)),
        "x_hi": Dirichlet2D(lambda y, t: ex(1.0, y, t)),
        "y_lo": Dirichlet2D(lambda x, t: ex(x, 0.0, t)),
        "y_hi": Dirichlet2D(lambda x, t: ex(x, 1.0, t)),
    }
    if neumann:
        edges["x_hi"] = Neumann2D(lambda y, t: AL * ex(1.0, y, t))
        edges["y_lo"] = Neumann2D(lambda x, t: BE * ex(x, 0.0, t))
    op = DiscreteOperator2D(LinearPDE2D(**coef), grid, edges, upwind="none")
    X, Y = grid.mesh()
    res = ADISolver(op, scheme, damping_steps=damping).solve(ex(X, Y, 0.0), 1.0, n_time)
    return res.values, ex(X, Y, 1.0)


@pytest.mark.parametrize("scheme", ["craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer"])
def test_second_order_in_space(scheme):
    # (Douglas is first order in time with a mixed term; its time error
    # would mask the spatial order at this step count.)
    errs = []
    for n in (11, 21, 41):
        u, ex = solve(n, 400, scheme)
        errs.append(np.abs(u - ex).max())
    assert 0.5 * np.log2(errs[0] / errs[2]) > 1.7
    assert errs[-1] < 3e-5


@pytest.mark.parametrize(
    "scheme, order",
    [
        ("douglas", 1.0),
        ("craig_sneyd", 2.0),
        ("modified_craig_sneyd", 2.0),
        ("hundsdorfer_verwer", 2.0),
    ],
)
def test_temporal_order(scheme, order):
    # temporal error measured against the same spatial grid with many steps
    ref, _ = solve(21, 1280, scheme)
    errs = [np.abs(solve(21, nt, scheme)[0] - ref).max() for nt in (5, 10, 20)]
    observed = 0.5 * np.log2(errs[0] / errs[2])
    assert observed == pytest.approx(order, abs=0.3)


def test_douglas_without_mixed_term_is_second_order_for_theta_half():
    coef = dict(COEF, axy=0.0)
    ref, _ = solve(21, 1280, "douglas", coef=coef)
    errs = [np.abs(solve(21, nt, "douglas", coef=coef)[0] - ref).max() for nt in (5, 10, 20)]
    assert 0.5 * np.log2(errs[0] / errs[2]) > 1.8


def test_neumann_edges():
    coef = dict(COEF, axy=0.0)  # the cross derivative is omitted on Neumann edges
    errs = [
        np.abs(np.subtract(*solve(n, 200, "hv", neumann=True, coef=coef))).max()
        for n in (11, 21, 41)
    ]
    assert 0.5 * np.log2(errs[0] / errs[2]) > 1.8


def test_damping_keeps_accuracy():
    u, ex = solve(41, 100, "hv", damping=2)
    assert np.abs(u - ex).max() < 5e-5
