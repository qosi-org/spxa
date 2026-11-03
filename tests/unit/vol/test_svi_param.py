"""Natural and jump-wings SVI parametrisations (Gatheral & Jacquier 2014, Section 3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spxa.vol import SVIJumpWings, SVINatural, SVIRaw, check_butterfly

K = np.linspace(-1.5, 1.5, 31)
CASES = [
    SVIRaw(0.02, 0.1, -0.4, 0.05, 0.2),
    SVIRaw(0.01, 0.2, 0.3, -0.3, 0.15),
    SVIRaw(0.01, 0.2, 0.3, 0.2, 0.15),
    SVIRaw(0.04, 0.05, -0.9, 0.0, 0.3),
    SVIRaw(-0.01, 0.3, 0.1, 0.4, 0.5),
]


@pytest.mark.parametrize("p", CASES)
def test_natural_round_trip(p: SVIRaw) -> None:
    n = p.to_natural()
    assert np.allclose(n(K), p(K), rtol=1e-14, atol=1e-16)
    q = n.to_raw()
    assert np.allclose(q.to_array(), p.to_array(), rtol=1e-13, atol=1e-15)


def test_natural_lemma_3_2() -> None:
    n = SVINatural(delta=0.01, mu=0.1, rho=-0.5, omega=0.2, zeta=2.0)
    r = n.to_raw()
    assert r.a == pytest.approx(0.01 + 0.1 * 0.75, abs=1e-16)
    assert r.b == pytest.approx(0.2, abs=1e-16)
    assert r.m == pytest.approx(0.1 + 0.25, abs=1e-16)
    assert r.sigma == pytest.approx(math.sqrt(0.75) / 2, abs=1e-16)


@pytest.mark.parametrize("p", CASES)
@pytest.mark.parametrize("t", [0.1, 1.0, 3.0])
def test_jump_wings_round_trip(p: SVIRaw, t: float) -> None:
    jw = p.to_jump_wings(t)
    q = jw.to_raw()
    assert np.allclose(q.to_array(), p.to_array(), rtol=1e-12, atol=1e-14)
    assert np.allclose(jw.total_variance(K), p(K), rtol=1e-12, atol=1e-15)


def test_jump_wings_meaning() -> None:
    p, t = CASES[0], 0.5
    jw = p.to_jump_wings(t)
    w0 = p(0.0)
    assert jw.v == pytest.approx(w0 / t, rel=1e-15)
    # ATM skew of sqrt(w): d/dk sqrt(w) = w'/(2 sqrt w)
    h = 1e-6
    fd = (math.sqrt(p(h)) - math.sqrt(p(-h))) / (2 * h)
    assert jw.psi == pytest.approx(fd, rel=1e-8)
    assert jw.p * math.sqrt(w0) == pytest.approx(p.left_slope, rel=1e-14)
    assert jw.c * math.sqrt(w0) == pytest.approx(p.right_slope, rel=1e-14)
    assert jw.v_min * t == pytest.approx(p.min_variance, rel=1e-14)


def test_symmetric_smile_m_zero() -> None:
    # chosen so that beta = rho - 2 psi sqrt(w)/b = 0.5 - 0.5 = 0 exactly (m = 0 branch)
    jw = SVIJumpWings(v=0.25, psi=0.25, p=0.5, c=1.5, v_min=0.2, t=1.0)
    q = jw.to_raw()
    assert q.m == 0.0 and q.b == 0.5 and q.rho == 0.5
    back = q.to_jump_wings(1.0)
    for name in ("v", "psi", "p", "c", "v_min"):
        assert getattr(back, name) == pytest.approx(getattr(jw, name), rel=1e-14)


def test_jump_wings_validation() -> None:
    with pytest.raises(ValueError, match="beta"):
        SVIJumpWings(v=0.04, psi=-2.0, p=0.5, c=0.5, v_min=0.03, t=1.0).to_raw()
    with pytest.raises(ValueError, match="positive"):
        SVIJumpWings(v=0.0, psi=0.0, p=0.5, c=0.5, v_min=0.0, t=1.0)
    with pytest.raises(ValueError, match="wing"):
        SVIJumpWings(v=0.04, psi=0.0, p=0.0, c=0.0, v_min=0.03, t=1.0)
    with pytest.raises(ValueError, match="positive ATM"):
        SVIRaw(-0.25, 0.5, 0.0, 0.0, 0.5).to_jump_wings(1.0)  # w(0) = 0
    with pytest.raises(ValueError):
        SVINatural(0.0, 0.0, 0.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        SVINatural(0.0, 0.0, 0.0, 1.0, 0.0)


class TestButterflyFreeProjection:
    vogt = SVIRaw(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)

    def test_vogt_jump_wings_match_paper(self) -> None:
        # Gatheral & Jacquier (2014), Section 5.1
        jw = self.vogt.to_jump_wings(1.0)
        got = (jw.v, jw.psi, jw.p, jw.c, jw.v_min)
        assert got == pytest.approx(
            (0.01742625, -0.1752111, 0.6997381, 1.316798, 0.0116249), rel=3e-6
        )

    def test_projection(self) -> None:
        jw = self.vogt.to_jump_wings(1.0)
        pr = jw.butterfly_free_projection()
        assert (pr.v, pr.psi, pr.p) == (jw.v, jw.psi, jw.p)
        assert pr.c == pytest.approx(0.6997381 - 2 * 0.1752111, rel=1e-6)
        assert pr.v_min == pytest.approx(jw.v * 4 * pr.p * pr.c / (pr.p + pr.c) ** 2, rel=1e-15)
        assert pr.satisfies_ssvi_bounds()
        k = np.linspace(-3, 3, 6001)
        raw = pr.to_raw()
        assert check_butterfly(raw, k)
        assert not check_butterfly(self.vogt, k)
        nat = pr.to_ssvi_natural()
        assert np.allclose(raw(k), nat(k), rtol=1e-12)
        # ATM level and skew are preserved
        assert raw(0.0) == pytest.approx(self.vogt(0.0), rel=1e-13)
        assert raw.dw(0.0) == pytest.approx(self.vogt.dw(0.0), rel=1e-12)

    def test_zero_skew_and_errors(self) -> None:
        flat = SVIJumpWings(0.04, 0.0, 0.5, 0.9, 0.03, 1.0).butterfly_free_projection()
        assert flat.c == 0.5
        nat = flat.to_ssvi_natural()
        assert nat.rho == 0.0 and nat(0.0) == pytest.approx(0.04, rel=1e-15)
        with pytest.raises(ValueError, match="p \\+ 2 psi"):
            SVIJumpWings(0.04, -0.3, 0.5, 0.9, 0.03, 1.0).butterfly_free_projection()
        with pytest.raises(ValueError, match="not an SSVI slice"):
            SVIJumpWings(0.04, -0.1, 0.5, 0.9, 0.03, 1.0).to_ssvi_natural()
        assert not SVIJumpWings(4.0, 0.0, 2.5, 2.5, 3.0, 1.0).satisfies_ssvi_bounds()
