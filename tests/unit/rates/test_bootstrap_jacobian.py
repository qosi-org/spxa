"""Node Jacobians of bootstrapped curves."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spxa.rates.bootstrap import CurveSpec, bootstrap_curve, fit_curves
from spxa.rates.interpolation import LogLinearDiscount, MonotoneConvex

from tests.unit.rates.test_bootstrap import MARKET, REF, _estr_helpers


def test_local_scheme_jacobian_is_lower_triangular() -> None:
    res = bootstrap_curve(_estr_helpers(MARKET), REF, LogLinearDiscount())
    J = res.jacobian()
    assert J.shape == (12, 12)
    assert np.max(np.abs(np.triu(J, 1))) < 1e-9
    assert np.all(np.diag(J) != 0)
    assert res.node_values == pytest.approx(-np.log(res.curve.node_discounts), abs=1e-15)


def test_node_sensitivities_match_rebootstrapping() -> None:
    hs = _estr_helpers(MARKET)
    res = bootstrap_curve(hs, REF, MonotoneConvex())
    S = res.node_sensitivities()
    h = 1e-6
    for i in (4, 9):  # 2Y and 15Y OIS
        up = [replace(x, quote=x.quote + (h if k == i else 0.0)) for k, x in enumerate(res.helpers)]
        dn = [replace(x, quote=x.quote - (h if k == i else 0.0)) for k, x in enumerate(res.helpers)]
        xu = bootstrap_curve(up, REF, MonotoneConvex()).node_values
        xd = bootstrap_curve(dn, REF, MonotoneConvex()).node_values
        assert S[:, i] == pytest.approx((xu - xd) / (2 * h), abs=2e-6)
    # non-local: a quote moves nodes on both sides of its pillar
    assert np.max(np.abs(np.triu(res.jacobian(), 1))) > 1e-6


def test_agrees_with_global_fit_jacobian() -> None:
    hs = _estr_helpers(MARKET)
    res = bootstrap_curve(hs, REF, MonotoneConvex())
    glob = fit_curves([CurveSpec("discount", hs, MonotoneConvex())], REF)
    assert res.jacobian() == pytest.approx(glob.jacobian, abs=1e-6)
