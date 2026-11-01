"""Exactness tests: least-squares Monte Carlo against the PDE pricers.

The LSM exercise policy is fitted on independent regression paths, so the
estimate is unbiased for a (slightly sub-optimal) policy; with a fixed seed
it must agree with the Bermudan PDE value with the same exercise dates
within three standard errors.
"""

import numpy as np
import pytest

from spxa.pde.black_scholes import BlackScholesPDE
from spxa.pde.lsm import lsm_american

pytestmark = pytest.mark.exactness

N_DATES = 50


@pytest.mark.parametrize("S", [36.0, 40.0, 44.0])
def test_longstaff_schwartz_table(S):
    # Longstaff & Schwartz (2001), Table 1: K = 40, r = 0.06, sigma = 0.2, T = 1
    dates = np.arange(1, N_DATES) / N_DATES
    ber = BlackScholesPDE(0.06, 0.2).bermudan(40.0, 1.0, S, dates, "put", n_space=801, n_time=1000)
    res = lsm_american(40.0, 1.0, S, 0.06, 0.2, n_dates=N_DATES, n_paths=200_000, seed=1)
    assert abs(res.price - ber.price) < 3 * res.std_error
    assert res.std_error < 0.007


@pytest.mark.parametrize("basis, degree", [("monomial", 3), ("laguerre", 5)])
def test_basis_choices(basis, degree):
    dates = np.arange(1, N_DATES) / N_DATES
    ber = BlackScholesPDE(0.06, 0.4).bermudan(
        40.0, 2.0, 40.0, 2 * dates, "put", n_space=801, n_time=1000
    )
    res = lsm_american(
        40.0,
        2.0,
        40.0,
        0.06,
        0.4,
        n_dates=N_DATES,
        n_paths=200_000,
        seed=11,
        basis=basis,
        degree=degree,
    )
    assert abs(res.price - ber.price) < 3 * res.std_error


def test_american_call_with_dividends():
    dates = np.arange(1, N_DATES) / N_DATES
    pde = BlackScholesPDE(0.03, 0.3, q=0.08)
    ber = pde.bermudan(100.0, 1.0, 100.0, dates, "call", n_space=801, n_time=1000)
    eu = pde.european(100.0, 1.0, 100.0, "call").price
    res = lsm_american(
        100.0, 1.0, 100.0, 0.03, 0.3, q=0.08, option="call", n_paths=200_000, seed=21
    )
    assert abs(res.price - ber.price) < 3 * res.std_error
    assert res.price > eu + 3 * res.std_error
