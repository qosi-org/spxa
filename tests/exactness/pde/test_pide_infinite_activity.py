"""Exactness tests: infinite-activity Lévy models (VG, CGMY, NIG) via the PIDE.

References are Lewis (2001) Fourier prices from each process's
characteristic function. These cases exercise the small-jump diffusion
and the moment-matched local variance, without which the scheme degrades
to first order for infinite-variation measures.
"""

import numpy as np
import pytest

from spxa.pde.pide import LevyPIDE
from spxa.pde.reference import lewis_call_price, levy_cf
from spxa.zoo.levy import CGMY, NIG, VarianceGamma

pytestmark = pytest.mark.exactness

R, Q, S0 = 0.05, 0.02, 100.0

CASES = {
    "VG": (VarianceGamma(sigma=0.12, nu=0.2, theta=-0.14), 1.0, 4e-4),
    "CGMY(Y=0.5)": (CGMY(C=1.0, G=8.8, M=12.0, Y=0.5), 0.5, 2.5e-4),
    "CGMY(Y=1.5)": (CGMY(C=0.1, G=10.0, M=12.0, Y=1.5), 0.5, 1e-4),
    "NIG": (NIG(alpha=15.0, beta=-5.0, delta=0.5, mu=0.0), 1.0, 2.5e-4),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("K", [90.0, 110.0])
def test_calls_against_lewis(name, K):
    proc, T, tol = CASES[name]
    ref = lewis_call_price(levy_cf(proc, T), S0, K, T, R, Q, "call")
    res = LevyPIDE.from_process(proc, R, Q).european(K, T, S0, "call", n_space=801, n_time=200)
    assert res.price == pytest.approx(ref, abs=tol)
    assert res.solution.info["neglected_tail_mass"] < 1e-8


@pytest.mark.parametrize("name", ["NIG", "CGMY(Y=1.5)"])
def test_second_order_for_infinite_variation(name):
    proc, T, _ = CASES[name]
    ref = lewis_call_price(levy_cf(proc, T), S0, 110.0, T, R, Q, "call")
    P = LevyPIDE.from_process(proc, R, Q)
    errs = [
        abs(P.european(110.0, T, S0, n_space=n, n_time=n // 4).price - ref) for n in (201, 401, 801)
    ]
    # judge the overall order (per-level orders wobble on uniform grids)
    assert 0.5 * np.log2(errs[0] / errs[2]) > 1.6
    assert errs[0] > errs[1] > errs[2]


def test_vg_put_and_delta():
    proc, T, _ = CASES["VG"]
    cf = levy_cf(proc, T)
    res = LevyPIDE.from_process(proc, R, Q).european(100.0, T, S0, "put", n_space=1601, n_time=400)
    assert res.price == pytest.approx(lewis_call_price(cf, S0, 100.0, T, R, Q, "put"), abs=1e-4)
    e = 1e-2
    up = lewis_call_price(cf, S0 + e, 100.0, T, R, Q, "put")
    dn = lewis_call_price(cf, S0 - e, 100.0, T, R, Q, "put")
    assert res.delta == pytest.approx((up - dn) / (2 * e), abs=5e-5)


def test_local_variance_can_be_negative_for_pure_jump_models():
    # the interpolation excess dominates the small-jump variance for VG: the
    # moment-matched local diffusion is slightly negative, yet stable
    proc, T, _ = CASES["VG"]
    res = LevyPIDE.from_process(proc, R, Q).european(100.0, T, S0, n_space=401, n_time=100)
    assert res.solution.info["local_diffusion"] < 0.0
    assert np.all(np.isfinite(res.solution.values))
    assert np.min(res.solution.values) > -1e-8
