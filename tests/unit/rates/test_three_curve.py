"""Three-curve EUR build: ESTR discounting, EURIBOR 3M, and 6M from 3s6s basis swaps."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates import EURIBOR3M, EURIBOR6M, FRA, basis_swap, vanilla_swap
from spxa.rates.bootstrap import CurveSpec, bootstrap_curve, fit_curves
from spxa.rates.helpers import BasisSwapHelper, FRAHelper, SwapHelper
from spxa.rates.interpolation import MonotoneConvex, PiecewiseCurve
from tests.unit.rates.test_bootstrap import MARKET, REF

E3 = PiecewiseCurve.from_zero_rates(
    [0.25, 1, 2, 5, 10, 20],
    [0.0395, 0.0385, 0.036, 0.0335, 0.0335, 0.033],
    MonotoneConvex(),
    reference_date=REF,
)
# 6M sits above 3M by a basis that decays with maturity
E6 = PiecewiseCurve.from_zero_rates(
    [0.5, 1, 2, 5, 10, 20],
    [0.0415, 0.0402, 0.0374, 0.0346, 0.0344, 0.0337],
    MonotoneConvex(),
    reference_date=REF,
)
TRUE = {"discount": MARKET, "EUR-3M": E3, "EUR-6M": E6}


def _three_month() -> list:
    hs: list = []
    for start in ("0M", "3M", "6M", "9M"):
        f = FRA.from_index(EURIBOR3M, REF, start, 0.0)
        hs.append(FRAHelper(f, quote=f.forward_rate(E3), curve="EUR-3M"))
    for t in ("2Y", "3Y", "5Y", "10Y", "20Y"):
        s = vanilla_swap(REF, t, 0.0, EURIBOR3M)
        hs.append(SwapHelper(s, quote=s.par_rate(MARKET, E3), forward="EUR-3M"))
    return hs


def _six_month() -> list:
    hs: list = []
    for start in ("0M", "6M"):
        f = FRA.from_index(EURIBOR6M, REF, start, 0.0)
        hs.append(FRAHelper(f, quote=f.forward_rate(E6), curve="EUR-6M"))
    for t in ("2Y", "3Y", "5Y", "10Y", "20Y"):
        bs = basis_swap(REF, t, EURIBOR3M, EURIBOR6M)
        hs.append(
            BasisSwapHelper(bs, bs.par_spread(MARKET, E3, E6), first="EUR-3M", second="EUR-6M")
        )
    return hs


@pytest.fixture(scope="module")
def sequential() -> tuple:
    r3 = bootstrap_curve(
        _three_month(), REF, MonotoneConvex(), name="EUR-3M", known={"discount": MARKET}
    )
    r6 = bootstrap_curve(_six_month(), REF, MonotoneConvex(), name="EUR-6M", known=r3.curves)
    return r3, r6


def test_sequential_three_curve_build(sequential: tuple) -> None:
    r3, r6 = sequential
    assert r3.max_residual < 1e-13 and r6.max_residual < 1e-13
    spreads = [h.quote for h in _six_month()[2:]]
    assert all(s > 0 for s in spreads)  # 3M pays a positive spread to swap into 6M
    assert spreads == sorted(spreads, reverse=True)  # and the basis decays with maturity
    # the 6M curve sits above the 3M one, as in the market it came from
    t = np.linspace(1, 20, 39)
    assert np.all(r6.curve.zero_rate(t) > r3.curve.zero_rate(t))


def test_joint_fit_matches_sequential(sequential: tuple) -> None:
    hs3, hs6 = _three_month(), _six_month()
    joint = fit_curves(
        [CurveSpec("EUR-3M", hs3, MonotoneConvex()), CurveSpec("EUR-6M", hs6, MonotoneConvex())],
        REF,
        known={"discount": MARKET},
    )
    assert joint.max_residual < 1e-13
    r6 = sequential[1]
    t = np.linspace(0.1, 20, 100)
    assert joint.curves["EUR-6M"].discount(t) == pytest.approx(r6.curve.discount(t), rel=1e-12)
    # 6M quotes do not move the 3M curve: the Jacobian block (3M rows, 6M columns) vanishes
    n3 = len(hs3)
    assert np.max(np.abs(joint.jacobian[:n3, n3:])) < 1e-10
