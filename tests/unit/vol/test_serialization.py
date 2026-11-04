"""JSON round trips of surfaces, models, forwards and curves."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from spxa.market.curves import EquityForward, FlatCurve
from spxa.rates.interpolation import MonotoneConvex, PiecewiseCurve
from spxa.vol import (
    ESSVI,
    SSVI,
    ATMVarianceCurve,
    ESSVISlice,
    HestonLikePhi,
    PowerLawPhi,
    SVIJumpWings,
    SVIRaw,
    VarianceInterpolatedSurface,
)
from spxa.vol.serialization import dumps, from_dict, loads, to_dict
from spxa.vol.surface import VolSurface

REF = dt.date(2024, 1, 15)
K = np.linspace(60, 160, 11)
TS = [0.1, 0.5, 1.0, 3.0]


def _same_surface(a: VolSurface, b: VolSurface) -> None:
    for T in TS:
        assert np.array_equal(b.implied_vol(K, T), a.implied_vol(K, T))  # bit for bit
        assert b.forward(T) == a.forward(T) and b.discount(T) == a.discount(T)


def test_ssvi_surface_on_curves() -> None:
    disc = PiecewiseCurve.from_zero_rates(
        [0.5, 1, 2, 5], [0.04, 0.038, 0.035, 0.033], MonotoneConvex(), reference_date=REF
    )
    fwd = EquityForward(
        4800.0, disc, FlatCurve(0.013, reference_date=REF), FlatCurve(0.039, reference_date=REF)
    )
    ss = SSVI(
        -0.65, PowerLawPhi(1.2, 0.45), ATMVarianceCurve([0.25, 1.0, 2.0], [0.01, 0.035, 0.07])
    )
    surf = VolSurface(ss, fwd)
    s = dumps(surf)
    assert json.loads(s)["type"] == "VolSurface"
    _same_surface(surf, loads(s))


def test_essvi_heston_phi_and_slices() -> None:
    fwd = EquityForward(100.0, 0.03, dividend_yield=0.01)
    es = ESSVI([0.25, 1.0], [ESSVISlice(0.01, 0.08, -0.7), ESSVISlice(0.04, 0.15, -0.6)])
    _same_surface(VolSurface(es, fwd), loads(dumps(VolSurface(es, fwd))))
    hl = SSVI(-0.5, HestonLikePhi(2.0), ATMVarianceCurve([1.0], [0.04]))
    _same_surface(VolSurface(hl, fwd), loads(dumps(VolSurface(hl, fwd))))
    jw = SVIJumpWings(0.04, -0.2, 0.7, 1.3, 0.012, 1.0)
    assert loads(dumps(jw)) == jw
    flat = VolSurface.flat(0.2, fwd)
    _same_surface(flat, loads(dumps(flat)))


def test_interpolated_surfaces() -> None:
    fwd = EquityForward(100.0, 0.02)
    slices = [SVIRaw(0.01 * t, 0.1 * t, -0.4, 0.05, 0.2) for t in (0.5, 1.0, 2.0)]
    for surf in (
        VolSurface.from_slices([0.5, 1.0, 2.0], slices, fwd),
        VolSurface(VarianceInterpolatedSurface([0.5, 1.0, 2.0], slices, "normalised"), fwd),
    ):
        back = loads(dumps(surf))
        for T in (0.7, 1.5):
            assert np.array_equal(back.implied_vol(K, T), surf.implied_vol(K, T))


def test_errors() -> None:
    with pytest.raises(TypeError):
        to_dict(SSVI(-0.5, PowerLawPhi(1.0, 0.4), lambda t: 0.04 * t))  # callable theta
    with pytest.raises(TypeError):
        to_dict(object())
    with pytest.raises(ValueError):
        from_dict({"type": "Nope"})
