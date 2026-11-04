"""Key-rate deltas on zero rates."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.market.curves import FlatCurve
from spxa.rates import ESTR, EURIBOR6M, FixedRateBond, ois_swap, vanilla_swap
from spxa.rates.risk import KeyRateShiftedCurve, key_rate_deltas, key_rate_hats
from tests.unit.rates.test_bootstrap import MARKET, REF

KEYS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]


def test_hats_partition_unity() -> None:
    t = np.linspace(0.0, 40.0, 801)
    h = key_rate_hats(KEYS, t)
    assert h.sum(axis=0) == pytest.approx(np.ones_like(t), abs=1e-15)
    assert np.all(h >= 0)
    assert key_rate_hats(KEYS, 2.0)[:, None].ravel() == pytest.approx([0, 0, 1, 0, 0, 0, 0])
    assert key_rate_hats(KEYS, 3.5)[2:4] == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError):
        key_rate_hats([1.0, 1.0], 0.5)


def test_shifted_curve() -> None:
    shifted = KeyRateShiftedCurve(MARKET, KEYS, [1e-3] * len(KEYS))
    t = np.linspace(0.1, 35.0, 50)
    assert shifted.zero_rate(t) == pytest.approx(MARKET.zero_rate(t) + 1e-3, abs=1e-15)
    one = KeyRateShiftedCurve(MARKET, KEYS, [0, 0, 0, 1e-3, 0, 0, 0])
    assert one.zero_rate(3.5) == pytest.approx(MARKET.zero_rate(3.5) + 0.5e-3, abs=1e-15)
    assert one.zero_rate(12.0) == pytest.approx(MARKET.zero_rate(12.0), abs=1e-15)
    with pytest.raises(ValueError):
        KeyRateShiftedCurve(MARKET, KEYS, [0.0])


def test_zero_coupon_bond_at_a_key() -> None:
    T = 5.0
    risk = key_rate_deltas(lambda c: float(c["discount"].discount(T)), {"discount": MARKET}, KEYS)
    expected = np.zeros(len(KEYS))
    expected[3] = -T * float(MARKET.discount(T))
    # central differences of exp(-bT) carry a relative error (bT)^2/6 = 4.2e-8
    assert risk.deltas == pytest.approx(expected, rel=1e-7, abs=1e-15)


def test_sum_equals_parallel_delta() -> None:
    bond = FixedRateBond("2024-01-15", "2036-01-15", 0.04)
    swap = vanilla_swap(REF, "12Y", 0.032, EURIBOR6M)
    curves = {"discount": MARKET, "EUR-6M": MARKET.shifted(0.002)}
    pvs = [
        lambda c: bond.dirty_price_from_curve(c["discount"], "2024-01-16"),
        lambda c: swap.pv(c["discount"], c["EUR-6M"]),
    ]
    for pv in pvs:
        risk = key_rate_deltas(pv, curves, KEYS)
        h = 1e-6
        par = (
            pv({n: c.shifted(h) for n, c in curves.items()})
            - pv({n: c.shifted(-h) for n, c in curves.items()})
        ) / (2 * h)
        assert risk.total == pytest.approx(par, rel=1e-6)  # bump sizes 1e-4 and 1e-6
        assert risk.per_bp == pytest.approx(1e-4 * risk.deltas)


def test_names_select_curves() -> None:
    swap = ois_swap(REF, "10Y", 0.03, ESTR)
    curves = {"discount": MARKET, "other": FlatCurve(0.05, reference_date=REF)}
    a = key_rate_deltas(
        lambda c: swap.pv(c["discount"], c["discount"]), curves, KEYS, names=["discount"]
    )
    b = key_rate_deltas(lambda c: swap.pv(c["discount"], c["discount"]), curves, KEYS)
    assert a.deltas == pytest.approx(b.deltas, abs=1e-12)
    # dual-curve: discount-only plus projection-only risk is the joint risk, to first order
    vanilla = vanilla_swap(REF, "10Y", 0.033, EURIBOR6M)
    dual = {"discount": MARKET, "EUR-6M": MARKET.shifted(0.003)}

    def pv(c):  # type: ignore[no-untyped-def]
        return vanilla.pv(c["discount"], c["EUR-6M"])

    d = key_rate_deltas(pv, dual, KEYS, names=["discount"]).deltas
    p = key_rate_deltas(pv, dual, KEYS, names=["EUR-6M"]).deltas
    both = key_rate_deltas(pv, dual, KEYS).deltas
    assert d + p == pytest.approx(both, rel=1e-6, abs=1e-9)
    with pytest.raises(KeyError):
        key_rate_deltas(lambda c: 0.0, curves, KEYS, names=["missing"])
