"""Every numerical statement in docs/theory/vol.md, checked against the code."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

import spxa.vol as v
from tests.unit.vol.test_essvi import TABLE1

DOC = Path(__file__).resolve().parents[2] / "docs" / "theory" / "vol.md"
VOGT = v.SVIRaw(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


def test_table_names_exist() -> None:
    table = DOC.read_text(encoding="utf-8").split("---", 1)[0]
    names = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", table))
    names -= {"spxa", "vol"}
    missing = sorted(n for n in names if not hasattr(v, n))
    assert not missing, missing


def test_doc_mentions_the_checked_numbers() -> None:
    text = DOC.read_text(encoding="utf-8")
    for s in (
        "-0.032864",
        "k = 0.879",
        "(0.01742625, -0.1752111, 0.6997381, 1.316798, 0.0116249)",
        "c' = 0.3493158",
        "0.00231",
        "(-0.7, 1.1, 0.4)",
        "(-0.6, 2)",
        "5 \\times 10^{-7}",
        "10^{-17}",
        "(s - s_i)(s_{i+1} - s)(s_i + s + s_{i+1})",
        "\\theta_{i+1}/20",
    ):
        assert s in text, s


class TestArbitrageAndWings:
    def test_durrleman_flat_and_density(self) -> None:
        # flat slices have g = 1 and the lognormal density
        assert v.durrleman_g(0.3, 0.04, 0.0, 0.0) == 1.0
        k = np.linspace(-1, 1, 11)
        sw = 0.2
        dm = -k / sw - sw / 2
        flat = v.SVIRaw(0.04, 0.0, 0.0, 0.0, 0.1)
        ref = np.exp(-0.5 * dm * dm) / (np.sqrt(2 * np.pi) * sw)
        assert np.allclose(v.risk_neutral_density(flat, k), ref, rtol=1e-14, atol=0)

    def test_lee_formula(self) -> None:
        for x in (0.05, 0.5, 3.0):
            beta = 2 - 4 * (math.sqrt(x * x + x) - x)
            assert v.lee_slope_from_moment(x) == pytest.approx(beta, abs=1e-14)
            assert v.lee_moment_from_slope(beta) == pytest.approx((2 - beta) ** 2 / (8 * beta))
        wa = v.wing_asymptotics(v.SVIRaw(0.02, 0.1, -0.4, 0.05, 0.2))
        assert wa.right_decay == pytest.approx(1 + wa.right_moment, rel=1e-15)
        assert wa.left_decay == wa.left_moment
        with pytest.raises(ValueError):
            v.lee_moment_from_slope(2.5)

    def test_rogers_tehranchi(self) -> None:
        assert v.rogers_tehranchi_slope(VOGT) < 4


class TestSVI:
    def test_natural_to_raw(self) -> None:
        d, mu, rho, om, ze = 0.01, 0.1, -0.3, 0.05, 2.0
        raw = v.SVINatural(d, mu, rho, om, ze).to_raw()
        assert (raw.a, raw.b, raw.m, raw.sigma) == pytest.approx(
            (
                d + om / 2 * (1 - rho * rho),
                om * ze / 2,
                mu - rho / ze,
                math.sqrt(1 - rho * rho) / ze,
            ),
            abs=1e-15,
        )
        assert raw.min_variance == pytest.approx(raw.a + raw.b * raw.sigma * math.sqrt(1 - rho**2))
        assert (raw.left_slope, raw.right_slope) == pytest.approx(
            (raw.b * (1 - rho), raw.b * (1 + rho))
        )

    def test_vogt(self) -> None:
        chk = v.check_butterfly(VOGT)
        assert round(chk.min_g, 6) == -0.032864 and round(chk.argmin, 3) == 0.879
        jw = VOGT.to_jump_wings(1.0)
        got = (jw.v, jw.psi, jw.p, jw.c, jw.v_min)
        paper = (0.01742625, -0.1752111, 0.6997381, 1.316798, 0.0116249)
        for g, p in zip(got, paper):
            assert g == pytest.approx(p, rel=5e-7)

    def test_projection(self) -> None:
        jw = VOGT.to_jump_wings(1.0)
        pr = jw.butterfly_free_projection()
        c = jw.p + 2 * jw.psi
        assert pr.c == pytest.approx(c, abs=1e-16) and round(pr.c, 7) == 0.3493158
        assert pr.v_min == pytest.approx(jw.v * 4 * jw.p * c / (jw.p + c) ** 2, rel=1e-14)
        slice_ = pr.to_ssvi_natural()
        assert v.check_butterfly(slice_)
        assert slice_.total_variance(0.0) == pytest.approx(VOGT(0.0), rel=1e-13)
        h = 1e-6
        skew = (slice_.total_variance(h) - slice_.total_variance(-h)) / (2 * h)
        assert skew == pytest.approx(VOGT.dw(0.0), rel=1e-7)

    def test_constrained_fit_and_quasi_explicit(self) -> None:
        k = np.linspace(-1.5, 1.5, 61)
        fit = v.calibrate_svi_butterfly_free(k, VOGT(k))
        assert fit.success and v.check_butterfly(fit.params)
        assert round(fit.rmse, 5) == 0.00231
        exact = v.calibrate_svi(k, VOGT(k)).params
        assert exact.to_array() == pytest.approx(VOGT.to_array(), abs=1e-8)


class TestSurfaces:
    T = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
    TH = np.array([0.004, 0.009, 0.017, 0.033, 0.064])
    KS = [np.linspace(-1.2 * np.sqrt(t), 0.8 * np.sqrt(t), 15) for t in T]

    def _data(self, s: v.SSVI) -> list[np.ndarray]:
        return [np.asarray(s.total_variance(k, t)) for k, t in zip(self.KS, self.T)]

    def test_ssvi_recovery(self) -> None:
        curve = v.ATMVarianceCurve(self.T, self.TH)
        pl = v.calibrate_ssvi(
            self.T, self.KS, self._data(v.SSVI(-0.7, v.PowerLawPhi(1.1, 0.4), curve))
        )
        p = pl.surface
        assert (p.rho, p.phi.eta, p.phi.gamma) == pytest.approx((-0.7, 1.1, 0.4), abs=1e-6)
        hl = v.calibrate_ssvi(
            self.T, self.KS, self._data(v.SSVI(-0.6, v.HestonLikePhi(2.0), curve)), phi="heston"
        )
        assert (hl.surface.rho, hl.surface.phi.lam) == pytest.approx((-0.6, 2.0), abs=1e-6)

    def test_ssvi_conditions(self) -> None:
        th = np.linspace(0.001, 1.0, 200)
        good = v.SSVI(-0.5, v.PowerLawPhi(2 / 1.5, 0.5), lambda t: t)
        assert good.is_arbitrage_free(th)  # eta (1 + |rho|) = 2, gamma = 1/2
        heston = v.SSVI(-0.5, v.HestonLikePhi(1.5 / 4), lambda t: t)
        assert heston.is_arbitrage_free(th)  # lambda = (1 + |rho|) / 4

    def test_essvi_anchor_and_table1(self) -> None:
        rho, psi, ks, ts = -0.6, 0.2, -0.05, 0.04
        th = v.anchored_theta(rho, psi, ks, ts)
        assert th == pytest.approx(((2 * ts - rho * psi * ks) ** 2 - psi**2 * ks**2) / (4 * ts))
        assert v.ESSVISlice(th, psi, rho).total_variance(ks) == pytest.approx(ts, rel=1e-14)
        times = [r[0] for r in TABLE1]
        kss = [np.linspace(-3 * np.sqrt(t), 1.5 * np.sqrt(t), 21) for _, t, _, _ in TABLE1]
        slices = [v.ESSVISlice(t, p, r) for _, t, p, r in TABLE1]
        fit = v.calibrate_essvi(times, kss, [s.total_variance(k) for s, k in zip(slices, kss)])
        err = max(
            max(abs(g.theta - t), abs(g.psi - p), abs(g.rho - r))
            for g, (_, t, p, r) in zip(fit.slices, TABLE1)
        )
        assert err < 5e-9

    def test_svi_surface_reproduces_ssvi(self) -> None:
        surf = v.SSVI(-0.6, v.PowerLawPhi(1.0, 0.4), lambda t: 0.04 * t)
        k = np.linspace(-0.6, 0.4, 21)
        T = [0.25, 0.5, 1.0, 2.0]
        fit = v.calibrate_svi_surface(T, [k] * 4, [surf.total_variance(k, t) for t in T])
        assert fit.success and v.check_calendar(fit.slices, np.linspace(-10, 10, 20001))
        for s, t in zip(fit.slices, T):
            assert np.max(np.abs(s(k) - surf.total_variance(k, t))) < 5e-7

    def test_interpolation_atm_shortfall(self) -> None:
        surf = v.SSVI(-0.65, v.PowerLawPhi(0.9, 0.35), lambda t: 0.035 * t + 0.002 * t * t)
        T = [0.25, 0.5, 1.0, 2.0]
        pis = v.PriceInterpolatedSurface(T, [surf.slice(t) for t in T])
        nodes = np.sqrt(np.concatenate([[0.0], pis.thetas]))
        for t in np.linspace(0.05, 1.95, 20):
            i, a = pis.bucket(t)
            s, lo, hi = math.sqrt(pis.theta(t)), nodes[i], nodes[i + 1]
            lead = s * (s - lo) * (hi - s) * (lo + s + hi) / 12
            short = pis.theta(t) - pis.total_variance(0.0, t)
            assert abs(short - lead) <= hi**2 / 20 * lead + 1e-16
            # the ATM price expansion behind it: (s - s^3/24)/sqrt(2 pi) + O(s^5)
            atm = v.normalised_call_from_total_variance(0.0, s * s)
            assert atm == pytest.approx((s - s**3 / 24) / math.sqrt(2 * math.pi), rel=s**4)


class TestDistributions:
    def test_survival_mass_and_martingale(self) -> None:
        s = v.SVIRaw(0.02, 0.1, -0.4, 0.05, 0.2)
        d = v.ImpliedDistribution(s)
        k = 0.1
        w, dw = s(k), s.dw(k)
        dm = -k / math.sqrt(w) - math.sqrt(w) / 2
        ref = 0.5 * math.erfc(-dm / math.sqrt(2)) - math.exp(-dm * dm / 2) / math.sqrt(
            2 * math.pi
        ) * dw / (2 * math.sqrt(w))
        assert d.survival(k) == pytest.approx(ref, abs=1e-15)
        assert d.cdf(d.k_min) < 1e-17 and math.exp(d.k_max) * d.survival(d.k_max) < 1e-17
        assert d.total_mass() == pytest.approx(1.0, abs=1e-11)
        assert d.forward_ratio() == pytest.approx(1.0, abs=1e-11)
        assert d.call(0.2) == pytest.approx(
            v.normalised_call_from_total_variance(0.2, s(0.2)), abs=1e-11
        )
        bad = v.ImpliedDistribution(VOGT)
        assert np.min(np.diff(bad.cdf(np.linspace(0.5, 1.3, 801)))) < 0
