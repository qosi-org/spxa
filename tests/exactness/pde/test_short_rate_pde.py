"""Exactness tests: Vasicek and CIR short-rate PDEs."""

import numpy as np
import pytest

from spxa.pde import reference as R
from spxa.pde.short_rate import ShortRatePDE

pytestmark = pytest.mark.exactness

MODELS = [
    ("vasicek", 0.3, 0.05, 0.02),
    ("cir", 0.3, 0.05, 0.1),  # Feller satisfied
    ("cir", 0.2, 0.03, 0.15),  # Feller violated: r = 0 is attainable
]


def closed_forms(model):
    if model == "vasicek":
        return R.vasicek_bond, R.vasicek_bond_option, R._vasicek_ab
    return R.cir_bond, R.cir_bond_option, R._cir_ab


class TestZeroCouponBond:
    @pytest.mark.parametrize("model,k,th,s", MODELS)
    @pytest.mark.parametrize("T", [1.0, 5.0, 10.0])
    @pytest.mark.parametrize("r0", [0.01, 0.06])
    def test_price(self, model, k, th, s, T, r0):
        bond = closed_forms(model)[0]
        pde = ShortRatePDE(k, th, s, model).zero_coupon_bond(T, r0)
        assert pde.price == pytest.approx(float(bond(r0, T, k, th, s)), abs=1e-6)

    @pytest.mark.parametrize("model,k,th,s", MODELS)
    def test_rate_sensitivities(self, model, k, th, s):
        bond, _, ab = closed_forms(model)
        res = ShortRatePDE(k, th, s, model).zero_coupon_bond(5.0, 0.04)
        B = ab(5.0, k, th, s)[1]
        P = float(bond(0.04, 5.0, k, th, s))
        assert res.delta == pytest.approx(-B * P, rel=2e-5)
        assert res.gamma == pytest.approx(B**2 * P, rel=2e-4)

    @pytest.mark.parametrize("model,k,th,s", MODELS)
    def test_whole_curve(self, model, k, th, s):
        bond = closed_forms(model)[0]
        res = ShortRatePDE(k, th, s, model).zero_coupon_bond(5.0, 0.04)
        r = np.linspace(0.0 if model == "cir" else -0.02, 0.15, 31)
        # r = 0 carries the first-order one-sided boundary stencil for CIR
        np.testing.assert_allclose(res.solution.value(r), bond(r, 5.0, k, th, s), atol=3e-6)

    def test_second_order_convergence(self):
        m = ShortRatePDE(0.2, 0.03, 0.15, "cir")
        ref = float(R.cir_bond(0.04, 5.0, 0.2, 0.03, 0.15))
        errs = [abs(m.zero_coupon_bond(5.0, 0.04, n, n // 2).price - ref) for n in (101, 201, 401)]
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(orders > 1.8)

    def test_coupon_bond_is_sum_of_zeros(self):
        k, th, s = 0.3, 0.05, 0.1
        flows = [(0.5, 0.025), (1.0, 0.025), (1.5, 0.025), (2.0, 1.025)]
        pde = ShortRatePDE(k, th, s, "cir").coupon_bond(flows, 0.04).price
        ref = sum(c * float(R.cir_bond(0.04, t, k, th, s)) for t, c in flows)
        assert pde == pytest.approx(ref, abs=5e-7)


class TestBondOptions:
    @pytest.mark.parametrize("model,k,th,s", MODELS)
    @pytest.mark.parametrize("K", [0.7, 0.8, 0.9])
    @pytest.mark.parametrize("option", ["call", "put"])
    def test_closed_form(self, model, k, th, s, K, option):
        bond, opt, _ = closed_forms(model)
        if K > float(bond(0.0, 4.0, k, th, s)) and option == "call":
            K = 0.85  # keep the strike attainable
        pde = ShortRatePDE(k, th, s, model).bond_option(K, 1.0, 5.0, 0.04, option)
        assert pde.price == pytest.approx(opt(0.04, K, 1.0, 5.0, k, th, s, option), abs=5e-6)

    def test_kink_on_node_and_convergence(self):
        m = ShortRatePDE(0.3, 0.05, 0.02, "vasicek")
        ref = R.vasicek_bond_option(0.04, 0.8, 1.0, 5.0, 0.3, 0.05, 0.02)
        res = [
            m.bond_option(0.8, 1.0, 5.0, 0.04, n_space=n, n_time=n // 2) for n in (101, 201, 401)
        ]
        r_star = res[-1].solution.info["r_star"]
        assert np.min(np.abs(res[-1].solution.grid.nodes - r_star)) == 0.0
        errs = np.array([abs(x.price - ref) for x in res])
        assert np.all(np.log2(errs[:-1] / errs[1:]) > 1.8)

    def test_parity_and_delta(self):
        k, th, s = 0.3, 0.05, 0.1
        m = ShortRatePDE(k, th, s, "cir")
        c = m.bond_option(0.8, 1.0, 5.0, 0.04, "call")
        p = m.bond_option(0.8, 1.0, 5.0, 0.04, "put")
        fwd = float(R.cir_bond(0.04, 5.0, k, th, s) - 0.8 * R.cir_bond(0.04, 1.0, k, th, s))
        assert c.price - p.price == pytest.approx(fwd, abs=1e-6)
        h = 1e-4
        fd = (
            R.cir_bond_option(0.04 + h, 0.8, 1.0, 5.0, k, th, s)
            - R.cir_bond_option(0.04 - h, 0.8, 1.0, 5.0, k, th, s)
        ) / (2 * h)
        assert c.delta == pytest.approx(fd, rel=1e-3)


class TestAmericanBondOptions:
    def test_call_on_zero_bond_is_never_exercised_with_positive_rates(self):
        m = ShortRatePDE(0.3, 0.05, 0.1, "cir")
        e = m.bond_option(0.8, 1.0, 5.0, 0.04, "call").price
        a = m.bond_option(0.8, 1.0, 5.0, 0.04, "call", american=True).price
        assert a == pytest.approx(e, abs=1e-10)

    @pytest.mark.parametrize("method", ["penalty", "brennan_schwartz", "ikonen_toivanen"])
    def test_put_premium_and_methods(self, method):
        m = ShortRatePDE(0.3, 0.05, 0.1, "cir")
        e = m.bond_option(0.8, 1.0, 5.0, 0.04, "put").price
        a = m.bond_option(0.8, 1.0, 5.0, 0.04, "put", american=True, method=method)
        assert a.price > e
        ref = m.bond_option(0.8, 1.0, 5.0, 0.04, "put", american=True).price
        assert a.price == pytest.approx(ref, abs=1e-5)
        # never below immediate exercise
        r = np.linspace(0.0, 0.2, 21)
        intrinsic = np.maximum(0.8 - R.cir_bond(r, 5.0, 0.3, 0.05, 0.1), 0.0)
        assert np.all(a.solution.value(r) >= intrinsic - 1e-6)


class TestValidation:
    def test_errors(self):
        with pytest.raises(ValueError):
            ShortRatePDE(0.3, 0.05, 0.1, "hull-white-2f")
        with pytest.raises(ValueError):
            ShortRatePDE(0.3, -0.01, 0.1, "cir")
        m = ShortRatePDE(0.3, 0.05, 0.1, "cir")
        with pytest.raises(ValueError):
            m.bond_option(0.8, 2.0, 1.0, 0.04)
        with pytest.raises(ValueError):
            m.bond_option(1.5, 1.0, 5.0, 0.04)
        with pytest.raises(ValueError):
            m.coupon_bond([(0.0, 1.0)], 0.04)
        assert "cir" in repr(m) and m.feller
