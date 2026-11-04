"""SABR swaption cube."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.rates.cube import SwaptionCube
from spxa.rates.sabr import SABR

EXP = [1.0, 5.0, 10.0]
TEN = [2.0, 10.0]


def _true(T: float, U: float) -> SABR:
    return SABR(0.03 + 0.002 * T, 0.5, -0.2 - 0.01 * U, 0.5 / np.sqrt(T), 0.0)


def _fwd(T: float, U: float) -> float:
    return 0.025 + 0.001 * T + 0.0005 * U


@pytest.fixture(scope="module")
def cube() -> SwaptionCube:
    F = [[_fwd(T, U) for U in TEN] for T in EXP]
    K = [[f * np.exp(np.linspace(-0.5, 0.5, 9)) for f in row] for row in F]
    V = [
        [_true(T, U).lognormal_vol(F[i][j], K[i][j], T) for j, U in enumerate(TEN)]
        for i, T in enumerate(EXP)
    ]
    return SwaptionCube.calibrate(EXP, TEN, F, K, V, beta=0.5)


def test_nodes_recover_their_smiles(cube: SwaptionCube) -> None:
    for T in EXP:
        for U in TEN:
            m, t = cube.sabr(T, U), _true(T, U)
            assert (m.alpha, m.rho, m.nu) == pytest.approx((t.alpha, t.rho, t.nu), rel=1e-7)
            K = np.array([0.02, 0.03, 0.04])
            assert cube.vol(T, U, K, _fwd(T, U)) == pytest.approx(
                t.lognormal_vol(_fwd(T, U), K, T), rel=1e-8
            )


def test_interpolation_and_extrapolation(cube: SwaptionCube) -> None:
    mid = cube.sabr(3.0, 6.0)
    a, b = 0.5, 0.5
    corners = [cube.sabr(1.0, 2.0), cube.sabr(5.0, 2.0), cube.sabr(1.0, 10.0), cube.sabr(5.0, 10.0)]
    w = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b]
    assert mid.rho == pytest.approx(sum(wi * c.rho for wi, c in zip(w, corners)), abs=1e-14)
    assert np.log(mid.alpha) == pytest.approx(
        sum(wi * np.log(c.alpha) for wi, c in zip(w, corners)), abs=1e-14
    )
    assert cube.sabr(0.25, 30.0) == cube.sabr(1.0, 10.0)
    assert cube.vol(3.0, 6.0, 0.03, 0.03, "normal") == pytest.approx(
        mid.normal_vol(0.03, 0.03, 3.0)
    )


def test_validation() -> None:
    m = SABR(0.03, 0.5, 0.0, 0.3)
    with pytest.raises(ValueError):
        SwaptionCube(np.array([1.0]), np.array([2.0, 5.0]), ((m,),))
    with pytest.raises(ValueError):
        SwaptionCube(np.array([1.0]), np.array([2.0, 5.0]), ((m, SABR(0.03, 0.7, 0.0, 0.3)),))
