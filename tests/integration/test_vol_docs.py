"""Every numerical statement in docs/theory/vol.md, checked against the code."""

from __future__ import annotations

import math
from statistics import NormalDist
import re
from pathlib import Path

import numpy as np
import pytest

import spxa.vol as v
from spxa.market.black import black_price
from spxa.market.curves import EquityForward
from tests.unit.vol.test_essvi import TABLE1
from tests.unit.vol.test_fengler import _surface_inputs

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
        "lowers the middle price to 7.5",
        "$0.005$",
        "102.0201 and 98.0199",
        "116.7539",
        "below $10^{-12}$",
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


FWD = EquityForward(100.0, 0.03, dividend_yield=0.01)
SKEWED = v.SSVI(-0.7, v.PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t)


class TestSurfaceConstruction:
    def test_repair_example(self) -> None:
        res = v.repair_call_prices([[90.0, 100.0, 110.0]], [[13.0, 8.0, 2.0]], [100.0], [1.0])
        assert res.prices[0] == pytest.approx([13.0, 7.5, 2.0], abs=1e-9)
        assert res.l1 == pytest.approx(0.005, abs=1e-12)
        assert res.violations_before == {"butterfly": 1} and res.violations_after == {}

    def test_fengler_calendar_rows(self) -> None:
        Ts, strikes, calls, Fs, Ds = _surface_inputs(0.02, 2)
        surf = v.fengler_surface(Ts, strikes, calls, Fs, Ds)
        assert surf.calendar_violation(20001) < 1e-12

    def test_dns_and_delta_strikes(self) -> None:
        assert round(v.atm_dns_strike(100.0, 1.0, 0.2), 4) == 102.0201
        assert round(v.atm_dns_strike(100.0, 1.0, 0.2, premium_adjusted=True), 4) == 98.0199
        k25 = v.strike_from_delta(0.25, 100.0, 1.0, 0.2)
        assert k25 == pytest.approx(
            100.0 * math.exp(0.2 * NormalDist().inv_cdf(0.75) + 0.02), rel=1e-12
        )
        assert round(k25, 4) == 116.7539
        # the premium-adjusted call delta (K/F) N(d2) is the forward delta less the premium
        F, K, T, sig = 100.0, 110.0, 1.0, 0.2
        pa = v.black_delta(F, K, T, sig, premium_adjusted=True)
        assert pa == pytest.approx(
            v.black_delta(F, K, T, sig) - black_price(F, K, T, sig, 1.0, "call") / F, rel=1e-12
        )

    def test_sticky_moneyness_delta_is_larger_under_negative_skew(self) -> None:
        surf = v.VolSurface(SKEWED, FWD)
        K = np.array([80.0, 100.0, 120.0])
        ss = surf.greeks(K, 1.0, dynamics="sticky_strike")["delta"]
        sm = surf.greeks(K, 1.0, dynamics="sticky_moneyness")["delta"]
        assert np.all(sm > ss)

    def test_normalised_interpolation_is_exact_for_constant_phi_sqrt_theta(self) -> None:
        ssvi = v.SSVI(
            -0.5, lambda th: 0.8 / np.sqrt(th), lambda t: 0.04 * t
        )  # phi sqrt(theta) = 0.8
        interp = v.VarianceInterpolatedSurface(
            [0.5, 2.0], [ssvi.slice(0.5), ssvi.slice(2.0)], "normalised"
        )
        k = np.linspace(-1.0, 0.8, 37)
        for t in (0.7, 1.0, 1.6):
            assert interp.total_variance(k, t) == pytest.approx(
                ssvi.total_variance(k, t), abs=1e-14
            )

    def test_forward_vol_raises_on_calendar_arbitrage(self) -> None:
        good = v.VolSurface(SKEWED, FWD)
        assert np.all(
            good.forward_variance_grid([0.25, 0.5, 1.0, 2.0], np.linspace(-1, 1, 21)) >= 0
        )
        # ATM variance increases, but the second slice dips below the first in the left wing
        first, second = v.SVIRaw(0.02, 0.1, -0.5, 0.0, 0.2), v.SVIRaw(0.035, 0.05, -0.5, 0.0, 0.2)
        bad = v.VolSurface.from_slices([0.5, 1.0], [first, second], FWD, "fixed_k")
        assert bad.forward_vol(0.5, 1.0) > 0
        with pytest.raises(ValueError):
            bad.forward_vol(0.5, 1.0, k=-1.0)

    def test_bumps(self) -> None:
        surf = v.VolSurface(SKEWED, FWD)
        s = 0.01
        assert surf.bumped(v.skew_bump(s)).atm_skew(1.0) == pytest.approx(
            surf.atm_skew(1.0) + s, abs=1e-9
        )
        hats = v.expiry_buckets([0.5, 1.0, 2.0], 1.0)
        at = [float(h(np.array([0.0]), 0.75)[0]) for h in hats]
        assert at == pytest.approx([0.5, 0.5, 0.0], abs=1e-15)

    def test_json_round_trip_is_bit_exact(self) -> None:
        surf = v.VolSurface(
            v.SSVI(
                -0.7, v.PowerLawPhi(1.1, 0.4), v.ATMVarianceCurve([0.5, 1, 2], [0.021, 0.043, 0.09])
            ),
            FWD,
        )
        back = v.loads(v.dumps(surf))
        K, T = np.linspace(60, 160, 11), 1.3
        assert np.array_equal(back.implied_vol(K, T), surf.implied_vol(K, T))


class TestLocalVolatility:
    def test_numbers_in_text(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        for s in ("$\\sqrt{0.02}$", "$T = 1.5$", "$bk/\\log(1 + bk/a)$", "tends to $-0.2$"):
            assert s in text, s

    def test_dupire_arbitrage_and_short_expiry(self) -> None:
        fwd = EquityForward(100.0, 0.03, dividend_yield=0.01)

        class Calendar:
            def total_variance(self, k, T):  # type: ignore[no-untyped-def]
                return 0.04 * T - 0.02 * T * T + 0.0 * np.asarray(k, float)

        lv = v.LocalVolSurface(v.VolSurface(Calendar(), fwd))
        assert lv.local_vol_k(0.0, 0.5) == pytest.approx(math.sqrt(0.02), abs=1e-12)
        assert np.isnan(lv.local_vol_k(0.0, 1.5))
        a, b, k = 0.2, -0.15, np.array([-0.4, 0.3])
        assert v.bbf_implied_vol(lambda y: a + b * y, k) == pytest.approx(
            b * k / np.log1p(b * k / a), rel=1e-13
        )

        class Linear:
            def total_variance(self, k, T):  # type: ignore[no-untyped-def]
                return (0.2 - 0.1 * np.asarray(k, float)) ** 2 * T

        lin = v.LocalVolSurface(v.VolSurface(Linear(), fwd))
        h = 1e-4
        skew = (lin.local_vol_k(h, 1e-4) - lin.local_vol_k(-h, 1e-4)) / (2 * h)
        assert skew == pytest.approx(-0.2, rel=2e-4)

    def test_diagnostics_flag_the_calendar_arbitrage(self) -> None:
        fwd = EquityForward(100.0, 0.03, dividend_yield=0.01)
        ss = v.SSVI(-0.7, v.PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t - 0.012 * t * t)
        d = v.local_vol_diagnostics(
            v.LocalVolSurface(v.VolSurface(ss, fwd)), 100.0, 0.03, 0.01, [100.0], [1.0, 2.0], n_t=40
        )
        assert d.first_undefined_time() == pytest.approx(1.7, abs=1e-12)
        assert "first flags undefined points at $t = 1.7$" in DOC.read_text(encoding="utf-8")


class TestStochasticLocalVol:
    def test_numbers_in_text(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        for s in (
            "$8.8948693600540$",
            "within 5 bp on a $121 \\times 61$ grid",
            "within 1.5 bp on $161 \\times 81$",
            "within 5 bp at $\\eta = 0.5$",
            "within 2.5 bp at $\\eta = 1$",
            "$L \\equiv 1$ to $10^{-9}$",
            "within 2.5%",
            "local-vol PDE value $8.18$ within 0.2%",
            "$5.80$, $6.00$ and $6.33$",
            "within 6 bp",
        ):
            assert s in text, s

    def test_heston_reference(self) -> None:
        m = v.Heston(0.04, 1.5, 0.04, 0.3, -0.9)
        got = m.price(100.0, 1.0, F=100 * math.exp(0.025), df=math.exp(-0.025))
        assert round(got, 13) == 8.8948693600540

    def test_mixing_fraction_table(self) -> None:
        from spxa.pde import BlackScholesPDE

        fwd = EquityForward(100.0, 0.03, dividend_yield=0.01)
        surf = v.VolSurface(
            v.SSVI(-0.7, v.PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t), fwd
        )
        lv = v.as_local_vol(surf, 1.0)
        lv_do = float(
            BlackScholesPDE(0.03, lv, 0.01).barrier(100.0, 1.0, 100.0, 85.0, "down-and-out", "call")
        )
        assert round(lv_do, 2) == 8.18
        heston = v.Heston(0.04, 1.5, 0.04, 0.5, -0.7)
        uo = []
        for eta in (0.0, 0.5, 1.0):
            pde = v.calibrate_slv(lv, heston, eta, 1.0, n_time=50).pde(
                0.03, 0.01, n_x=81, n_v=31, n_time=40
            )
            uo.append(round(pde.barrier(100.0, 1.0, "call", 100.0, upper=130.0), 2))
            if eta == 0.0:
                assert pde.barrier(100.0, 1.0, "call", 100.0, lower=85.0) == pytest.approx(
                    lv_do, rel=2e-3
                )
        assert uo == [5.80, 6.00, 6.33]


class TestRoughVolatility:
    PARAMS = (0.04, 0.04, 0.3, 0.3, -0.7, 0.1)

    def test_rough_heston_numbers(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        for s in (
            "$23.23\\%$, $17.38\\%$ and $13.63\\%$",
            "within $0.05$ of $H - 1/2 = -0.4$",
            "$U = 20/\\sqrt{\\bar vT}$",
        ):
            assert s in text, s
        vols = v.RoughHeston(*self.PARAMS).implied_vol(np.array([0.8, 1.0, 1.2]), 1.0, n_steps=400)
        assert [round(100 * x, 2) for x in vols] == [23.23, 17.38, 13.63]
        # skew slope and second-order convergence at H = 1/2: tests/unit/vol/test_rough_heston.py

    def test_lifted_heston_gaps(self) -> None:
        assert "$1.34$, $0.79$, $0.46$ and $0.24$ vol points" in DOC.read_text(encoding="utf-8")
        K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        rough = v.RoughHeston(*self.PARAMS).implied_vol(K, 1.0, n_steps=400)
        gaps = [
            round(
                100
                * np.abs(
                    v.LiftedHeston(*self.PARAMS, n=n).implied_vol(K, 1.0, n_steps=400) - rough
                ).max(),
                2,
            )
            for n in (5, 10, 20, 50)
        ]
        assert gaps == [1.34, 0.79, 0.46, 0.24]

    def test_calibration_claims(self) -> None:
        # recovery to rtol 1e-8 and the H = 1/2 RMSE above 0.01: tests/unit/vol/test_rough_calibration.py
        text = DOC.read_text(encoding="utf-8")
        assert "recovers the parameters to a relative $10^{-8}$" in text
        assert "a third to a half of the Fourier nodes diverge" in text
        assert "leaves an RMSE above one vol point" in text

    def test_unstable_nodes_at_100_and_400_steps(self) -> None:
        import math

        m = v.RoughHeston(0.03, 0.05, 0.3, 0.4, -0.65, 0.08)
        x = np.polynomial.legendre.leggauss(256)[0]
        for T in (0.1, 0.5, 1.5):
            vbar = max(float(m.expected_variance(T)), m.v0, 1e-4)
            u = 0.5 * 20.0 / math.sqrt(vbar * T) * (x + 1.0)
            with np.errstate(all="ignore"):
                bad = [int((~np.isfinite(m.cf(u - 0.5j, T, n))).sum()) for n in (100, 400)]
            assert 256 / 3 <= bad[0] <= 256 / 2 and bad[1] == 0

    def test_rough_bergomi_numbers(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        for s in (
            "only $19\\%$",
            "about $1.3$ at the money and $1.7$ at strike $1.2$",
            "RMSE of $0.3$ vol points",
        ):
            assert s in text, s
        assert "from $(1.6, -0.85)$ to $(1.56, -0.86)$" in text
        K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        p = v.rbergomi_paths(0.1, 1.9, -0.9, 0.04, 1.0, 100, 50_000, seed=2)
        ratio = p.price(K, conditional=False)[1] / p.price(K)[1]
        assert round(ratio[2], 1) == 1.3 and round(ratio[-1], 1) == 1.7
        T, Ks = 0.5, np.exp(np.linspace(-0.25, 0.15, 9))
        target = v.rbergomi_paths(0.08, 1.6, -0.85, 0.035, T, 50, 100_000, seed=9).implied_vol(Ks)
        start = dict(hurst=0.08, eta=1.0, rho=-0.5, xi0=0.035)
        fit = v.calibrate_rbergomi(T, Ks, target, start, free=("eta", "rho"), seed=1)
        assert round(100 * fit.rmse, 1) == 0.3
        assert (round(fit.eta, 2), round(fit.rho, 2)) == (1.56, -0.86)

    def test_vix_numbers(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        for s in (
            "$\\sqrt{E[\\mathrm{VIX}_T^2]} = 21.31\\%$",
            "$18.04\\%$ (lognormal approximation: $17.98\\%$)",
            "from $81.2\\%$ to $81.9\\%$",
            "$135.5\\%$, $97.6\\%$, $66.5\\%$ and $53.6\\%$",
            "futures price is $17.06\\%$ against $\\sqrt{E[\\mathrm{VIX}_T^2]} = 20\\%$",
            "from $94.8\\%$ at strike $0.12$ to $76.0\\%$ at $0.30$",
        ):
            assert s in text, s
        m = v.RBergomiVIX(0.1, 1.9, lambda t: 0.04 + 0.01 * np.asarray(t))
        assert round(100 * math.sqrt(m.expected_vix_squared(0.5)), 2) == 21.31
        assert round(100 * m.futures(0.5, 400_000, seed=1)[0], 2) == 18.04
        assert round(100 * m.futures_lognormal(0.5), 2) == 17.98
        iv = m.implied_vols(0.5, [0.14, 0.36], seed=1)
        assert [round(100 * x, 1) for x in iv] == [81.2, 81.9]
        atm = [
            100 * m.implied_vols(t, m.futures(t, seed=1)[0], seed=1)[0]
            for t in (0.05, 0.25, 1.0, 2.0)
        ]
        assert [round(x, 1) for x in atm] == [135.5, 97.6, 66.5, 53.6]
        h = v.HestonVIX(v.Heston(0.04, 1.5, 0.04, 0.5, -0.7))
        assert round(100 * h.futures(0.5), 2) == 17.06
        assert math.sqrt(h.expected_vix_squared(0.5)) == pytest.approx(0.2, abs=1e-15)
        assert [round(100 * x, 1) for x in h.implied_vols(0.5, [0.12, 0.30])] == [94.8, 76.0]

    def test_joint_calibration_numbers(self) -> None:
        from tests.unit.vol.test_joint_calibration import START, quotes

        text = DOC.read_text(encoding="utf-8")
        for s in (
            "by $17$ vol points RMS",
            "to $3.6$ vol points",
            "$2.0$ vol points on SPX and $2.0$ VIX points",
            "$\\xi = 1.66$",
        ):
            assert s in text, s
        spx, vix = quotes()
        market = [
            v.VIXQuote(q.T, q.futures, q.strikes, q.vols[1] + 0.4 * np.log(q.strikes / q.futures))
            for q in vix
        ]
        spx_only = v.calibrate_heston_joint(spx, market, START, weights=(1.0, 0.0, 0.0))
        assert round(100 * spx_only.rmse("vix")) == 17
        joint = v.calibrate_heston_joint(spx, market, START)
        assert round(100 * joint.rmse("vix"), 1) == 3.6
        assert (
            round(100 * joint.rmse("spx"), 1) == 2.0
            and round(100 * joint.rmse("futures"), 1) == 2.0
        )
        assert round(joint.model.xi, 2) == 1.66


class TestMartingaleTransport:
    def test_bounds_numbers(self) -> None:
        from tests.unit.vol.test_mot import MU, NU, straddle

        text = DOC.read_text(encoding="utf-8")
        for s in (
            "$[0.0507, 0.1389]$",
            "Black–Scholes value $0.1127$",
            "$[0.0466, 0.2721]$",
            "($4.76 \\times 10^{-5}$",
        ):
            assert s in text, s
        lo, hi = v.mot_bounds(MU, NU, straddle)
        assert (round(lo, 4), round(hi, 4)) == (0.0507, 0.1389)
        bs = 2 * (2 * NormalDist().cdf(0.2 * math.sqrt(0.5) / 2) - 1)
        assert round(bs, 4) == 0.1127
        ot = (v.ot_solve(MU, NU, straddle, "min").value, v.ot_solve(MU, NU, straddle, "max").value)
        assert (round(ot[0], 4), round(ot[1], 4)) == (0.0466, 0.2721)
        # the truncated put value: tests/unit/vol/test_marginals.py::test_surface_marginals_from_ssvi

    def test_sinkhorn_gaps(self) -> None:
        from tests.unit.vol.test_mot import MU, NU, straddle

        text = DOC.read_text(encoding="utf-8")
        assert "$3.0 \\times 10^{-3}$ at $\\varepsilon = 10^{-2}$" in text
        assert "$2.0 \\times 10^{-4}$ at $\\varepsilon = 3 \\times 10^{-3}$" in text
        hi = v.mot_solve(MU, NU, straddle).value
        gaps = [hi - v.mot_sinkhorn(MU, NU, straddle, eps=e).value for e in (1e-2, 3e-3)]
        assert [f"{g:.1e}" for g in gaps] == ["3.0e-03", "2.0e-04"]

    def test_backtest_numbers(self) -> None:
        from spxa.vol.marginals import marginal_from_calls

        text = DOC.read_text(encoding="utf-8")
        for s in (
            "$[0.0205, 0.0606]$",
            "on 50,000 paths is $0.0457$",
            "more than $0.0009$, under 4%",
        ):
            assert s in text, s
        heston, grid = v.Heston(0.04, 1.5, 0.04, 0.5, -0.7), np.linspace(0.05, 3.05, 121)
        mu = marginal_from_calls(grid, np.asarray(heston.price(grid, 0.5), float))
        nu = marginal_from_calls(
            grid, np.asarray(heston.price(grid, 1.0), float), floor=mu.call(grid)
        )
        pay = v.forward_start_call(1.0)
        lo, hi = v.mot_solve(mu, nu, pay, "min"), v.mot_solve(mu, nu, pay, "max")
        assert (round(lo.value, 4), round(hi.value, 4)) == (0.0205, 0.0606)
        paths = v.slv_paths(
            heston, lambda y, t: np.ones_like(y), 1.0, 200, 50_000, record=(0.5, 1.0), seed=3
        )
        X, Y = np.exp(paths.at(0.5)[0]), np.exp(paths.at(1.0)[0])
        assert round(float(pay(X, Y).mean()), 4) == 0.0457
        worst = -float(v.superhedge_pnl(hi, pay, X, Y).min())
        assert round(worst, 4) == 0.0009 and worst < 0.04 * (grid[1] - grid[0])

    def test_asian_numbers(self) -> None:
        from tests.unit.vol.test_mot_multi import MS

        assert "$[0.0634, 0.0778]$, around the Black–Scholes value $0.0705$" in DOC.read_text(
            encoding="utf-8"
        )
        asian = lambda a, b, c: np.maximum((a + b + c) / 3 - 1.0, 0.0)  # noqa: E731
        lo, hi = v.mot_multi(MS, asian, "min")[0], v.mot_multi(MS, asian, "max")[0]
        assert (round(lo, 4), round(hi, 4)) == (0.0634, 0.0778)
        rng = np.random.default_rng(0)
        S = np.exp(
            np.cumsum(rng.standard_normal((400_000, 3)) * 0.2 * math.sqrt(0.5) - 0.01, axis=1)
        )
        assert round(float(np.maximum(S.mean(axis=1) - 1.0, 0.0).mean()), 4) == 0.0705


class TestVarianceSwaps:
    SURFACE = v.VolSurface(
        v.SSVI(-0.7, v.PowerLawPhi(1.1, 0.4), lambda t: 0.04 * t + 0.003 * t * t),
        EquityForward(100.0, 0.03, dividend_yield=0.01),
    )

    def test_replication_numbers(self) -> None:
        kvar = v.variance_swap_strike(self.SURFACE, 1.0)
        kgam = v.gamma_swap_strike(self.SURFACE, 1.0)
        atm = float(self.SURFACE.total_variance(np.zeros(1), 1.0)[0])
        assert (round(kvar, 4), round(kgam, 4), round(atm, 3)) == (0.0569, 0.0494, 0.043)
        assert round(v.vix_index(self.SURFACE), 2) == 21.61
        text = DOC.read_text(encoding="utf-8")
        for claim in ("$K_{\\text{var}} = 0.0569$", "$0.0494$", "$21.61$", "$0.2090$", "= 0.2228$"):
            assert claim in text

    def test_volatility_swap_numbers(self) -> None:
        h = v.Heston(0.04, 1.5, 0.06, 0.5, -0.7)
        assert round(v.heston_volatility_swap(h, 1.0), 4) == 0.2090
        assert round(math.sqrt(h.expected_total_variance(1.0)), 4) == 0.2228


class TestModelRisk:
    """§13 numbers are asserted in tests/integration/test_model_risk_slv.py; this pins the text to them."""

    def test_claims_match_the_integration_tests(self) -> None:
        source = (Path(__file__).parent / "test_model_risk_slv.py").read_text(encoding="utf-8")
        for pinned in ("[0.07, 0.21]", "[0.02, 0.08, 0.11]", "[0.01, 0.09]", "== 0.008", "< 15e-4"):
            assert pinned in source
        section = DOC.read_text(encoding="utf-8").split("## 13. Model risk", 1)[1]
        for claim in (
            "7% at the money and 21%",
            "2%, 8% and 11%",
            "by 9%",
            "by 1%",
            "$0.008$",
            "15 bp",
        ):
            assert claim in section

    def test_coherence_claims(self) -> None:
        rng = np.random.default_rng(0)
        mr = v.ModelRisk(
            tuple("abcd"), rng.normal(size=(4, 2)), rng.uniform(0, 1, 4), np.zeros((4, 2))
        )
        for lam in (0.0, 0.5, 1.0, 5.0):
            assert np.all(mr.penalised_width(lam + 0.5) <= mr.penalised_width(lam))
        w = np.array([1.0, -2.0])
        assert mr.combine(w).width() <= abs(w) @ mr.width() + 1e-12


class TestConnections:
    def test_doc_mentions_the_checked_numbers(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        for s in ("12.4 bp at most", "to $10^{-8}$", "reprices to $10^{-12}$"):
            assert s in text, s

    def test_merton_surface(self) -> None:
        from spxa.zoo.levy.merton import MertonJumpDiffusion

        m = MertonJumpDiffusion(sigma=0.15, rate=0.5, jump_mean=-0.1, jump_std=0.15)
        out = v.process_vol_surface(m, 100.0, 0.03, [0.25, 0.5, 1.0, 2.0], dividend_yield=0.01)
        assert round(out.max_vol_error * 1e4, 1) == 12.4
