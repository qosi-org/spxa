"""Unit tests for the discrete Lévy jump integral."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from spxa.pde.pide import LevyJumpOperator

LAM, MU, DELTA = 0.7, -0.2, 0.15


def merton_density(y):
    return LAM * norm.pdf(y, MU, DELTA)


def vg_density(y, C=5.0, G=8.0, M=12.0):
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    pos, neg = y > 0, y < 0
    out[pos] = C * np.exp(-M * y[pos]) / y[pos]
    out[neg] = C * np.exp(-G * np.abs(y[neg])) / np.abs(y[neg])
    return out


def mass(f, a, b, power=0):
    return quad(lambda y: y**power * float(f(np.array([y]))[0]), a, b, limit=200)[0]


class TestWeights:
    def test_offsets_and_shape(self):
        op = LevyJumpOperator(merton_density, 0.01, 1.0)
        assert op.offsets[0] == -100 and op.offsets[-1] == 100
        assert op.weights.shape == op.offsets.shape
        assert np.all(op.weights >= 0.0)

    def test_moments_of_hat_functions(self):
        h = 0.02
        op = LevyJumpOperator(merton_density, h, 1.5)
        y_max = op.offsets[-1] * h
        # total mass and first moment of the jumps with |y| >= h/2 are exact
        m0 = mass(merton_density, -y_max, -h / 2) + mass(merton_density, h / 2, y_max)
        m1 = mass(merton_density, -y_max, -h / 2, 1) + mass(merton_density, h / 2, y_max, 1)
        m2 = mass(merton_density, -y_max, -h / 2, 2) + mass(merton_density, h / 2, y_max, 2)
        k = op.offsets * h
        assert op.intensity == pytest.approx(m0, rel=1e-12)
        assert np.dot(op.weights, k) == pytest.approx(m1, rel=1e-11)
        # linear interpolation over-weights y^2 by exactly interpolation_variance
        assert np.dot(op.weights, k**2) - op.interpolation_variance == pytest.approx(m2, rel=1e-11)
        assert op.interpolation_variance > 0.0

    def test_small_jumps_and_tail(self):
        h = 0.02
        op = LevyJumpOperator(merton_density, h, 0.5)
        assert op.small_jump_variance == pytest.approx(
            mass(merton_density, -h / 2, h / 2, 2), rel=1e-10
        )
        y_max = op.offsets[-1] * h
        tail = mass(merton_density, -10, -y_max) + mass(merton_density, y_max, 10)
        assert op.tail_mass == pytest.approx(tail, rel=1e-8)
        assert op.tail_mass > 1e-3  # 0.5 cuts into N(-0.2, 0.15)
        wide = LevyJumpOperator(merton_density, h, 3.0)
        assert wide.tail_mass < 1e-12

    def test_local_variance_definition(self):
        op = LevyJumpOperator(vg_density, 0.01, 2.0)
        assert op.local_variance == pytest.approx(
            op.small_jump_variance - op.interpolation_variance
        )

    def test_compensator_converges_to_continuous(self):
        omega = LAM * (math.exp(MU + 0.5 * DELTA**2) - 1.0)
        errs = []
        for h in (0.02, 0.01, 0.005):
            op = LevyJumpOperator(merton_density, h, 2.0)
            # the small-jump region carries the missing part of the integral
            errs.append(abs(op.compensator - omega))
        assert errs[-1] < 1e-4
        assert errs[0] > errs[1] > errs[2]
        h = 0.005
        op = LevyJumpOperator(merton_density, h, 2.0)
        assert op.intensity + mass(merton_density, -h / 2, h / 2) == pytest.approx(LAM, rel=1e-9)

    def test_infinite_activity_weights_finite(self):
        h = 0.01
        op = LevyJumpOperator(vg_density, h, 2.0)
        assert np.all(np.isfinite(op.weights))
        # nu(y) ~ C/|y|: the intensity of jumps above h/2 grows like 2 C log(1/h)
        assert op.intensity == pytest.approx(
            mass(vg_density, -2.0, -h / 2) + mass(vg_density, h / 2, 2.0), rel=1e-10
        )
        assert op.small_jump_variance == pytest.approx(2 * 5.0 * (h / 2) ** 2 / 2, rel=0.05)

    def test_invalid_spacing(self):
        with pytest.raises(ValueError):
            LevyJumpOperator(merton_density, 0.0, 1.0)


class TestApply:
    def test_matches_direct_sum(self):
        h = 0.05
        op = LevyJumpOperator(merton_density, h, 0.6)
        x = np.linspace(-1.0, 1.0, 41)
        u = np.sin(3 * x) + x**2

        def far(xx, tau):
            return np.sin(3 * xx) + xx**2 + tau

        got = op.apply(u, x, 0.5, far, far)
        m = op.offsets[-1]
        direct = np.zeros_like(x)
        for i, xi in enumerate(x):
            for k, w in zip(op.offsets, op.weights):
                j = i + k
                val = u[j] if 0 <= j < x.size else far(np.array([xi + k * h]), 0.5)[0]
                direct[i] += w * val
        assert m == 12
        np.testing.assert_allclose(got, direct, rtol=1e-12, atol=1e-12)

    def test_far_field_used_outside(self):
        h = 0.1
        op = LevyJumpOperator(merton_density, h, 1.0)
        x = np.linspace(0.0, 1.0, 11)
        zero = op.apply(
            np.zeros_like(x),
            x,
            0.0,
            lambda xx, t: np.ones_like(xx),
            lambda xx, t: np.zeros_like(xx),
        )
        # only jumps leaving through the left end see the unit far field
        assert zero[0] > 0.1
        assert abs(zero[-1]) < 1e-12
        assert zero[0] == pytest.approx(np.sum(op.weights[op.offsets < 0]), rel=1e-12)

    def test_exponential_is_reproduced(self):
        h = 0.01
        op = LevyJumpOperator(merton_density, h, 2.0)
        x = np.linspace(-0.5, 0.5, 101)

        def ex(xx, tau):
            return np.exp(xx)

        got = op.apply(np.exp(x), x, 0.0, ex, ex)
        np.testing.assert_allclose(got, np.exp(x) * (op.intensity + op.compensator), rtol=1e-12)

    def test_custom_weights(self):
        h = 0.05
        op = LevyJumpOperator(merton_density, h, 0.5)
        x = np.linspace(0.0, 1.0, 21)
        w = np.zeros_like(op.weights)
        w[op.offsets[-1]] = 1.0  # identity
        u = np.cos(x)

        def far(xx, tau):
            return np.cos(xx)

        np.testing.assert_allclose(op.apply(u, x, 0.0, far, far, weights=w), u, atol=1e-13)
