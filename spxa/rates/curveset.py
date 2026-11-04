r"""Named curve sets: one discount curve and a projection curve per index.

A :class:`CurveSet` is a read-only mapping from curve names to curves, so
it can be passed anywhere helpers take ``curves``. It also records the
*role* of each curve:

* ``discount`` names the curve used to discount (for example ``"ESTR"``
  for EUR cash flows collateralised in EUR);
* ``projections`` maps index names to curve names. An index without an
  entry is projected from a curve with the same name as the index, and
  failing that from the discount curve when the index is the
  discounting overnight index.

:meth:`CurveSet.pv` prices swaps and basis swaps by reading the index of
each floating leg from its coupons, so that, for example, an ESTR OIS and
a EURIBOR 6M swap in one set are each projected correctly. The bump
helpers return new sets and leave the original unchanged.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from spxa.market.curves import YieldCurve
from spxa.market.dates import DateLike
from spxa.rates.indices import FixingHistory

__all__ = ["CurveSet"]


def _index_name(index: Any) -> str:
    return index if isinstance(index, str) else str(index.name)


def _leg_index(leg: Sequence[Any]) -> Optional[str]:
    for c in leg:
        idx = getattr(c, "index", None)
        if idx is not None:
            return str(idx.name)
    return None


class CurveSet(Mapping[str, YieldCurve]):
    """Curves by name, with a discount role and index-to-curve projection roles."""

    def __init__(
        self,
        curves: Mapping[str, YieldCurve],
        discount: str,
        projections: Optional[Mapping[str, str]] = None,
        fixings: Optional[Mapping[str, FixingHistory]] = None,
    ) -> None:
        self._curves = dict(curves)
        if discount not in self._curves:
            raise KeyError(f"discount curve {discount!r} is not in the set")
        self.discount = discount
        self.projections = dict(projections or {})
        for idx, name in self.projections.items():
            if name not in self._curves:
                raise KeyError(f"projection curve {name!r} for {idx!r} is not in the set")
        self.fixings = dict(fixings or {})

    # ------------------------------------------------------------ mapping
    def __getitem__(self, name: str) -> YieldCurve:
        return self._curves[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._curves)

    def __len__(self) -> int:
        return len(self._curves)

    # -------------------------------------------------------------- roles
    @property
    def discount_curve(self) -> YieldCurve:
        return self._curves[self.discount]

    def projection_name(self, index: Any) -> str:
        """Name of the curve projecting ``index`` (an index or its name)."""
        name = _index_name(index)
        if name in self.projections:
            return self.projections[name]
        if name in self._curves:
            return name
        raise KeyError(f"no projection curve for index {name!r}")

    def projection(self, index: Any) -> YieldCurve:
        return self._curves[self.projection_name(index)]

    def fixings_for(self, index: Any) -> Optional[FixingHistory]:
        return self.fixings.get(_index_name(index))

    # ------------------------------------------------------------ pricing
    def pv(self, instrument: Any, valuation_date: Optional[DateLike] = None) -> float:
        """PV of a :class:`~spxa.rates.swaps.Swap` or :class:`~spxa.rates.swaps.BasisSwap`."""
        from spxa.rates.swaps import BasisSwap, Swap

        D = self.discount_curve
        if isinstance(instrument, Swap):
            idx = _leg_index(instrument.floating)
            proj = self.projection(idx) if idx else None
            fx = self.fixings_for(idx) if idx else None
            return float(instrument.pv(D, proj, fx, valuation_date))
        if isinstance(instrument, BasisSwap):
            i1, i2 = _leg_index(instrument.first), _leg_index(instrument.second)
            if i1 is None or i2 is None:
                raise ValueError("both basis-swap legs must be floating")
            return float(
                instrument.pv(
                    D,
                    self.projection(i1),
                    self.projection(i2),
                    self.fixings_for(i1),
                    self.fixings_for(i2),
                    valuation_date,
                )
            )
        raise TypeError(f"cannot price {type(instrument).__name__} from a CurveSet")

    # -------------------------------------------------------------- edits
    def with_curve(self, name: str, curve: YieldCurve) -> "CurveSet":
        """Copy with ``name`` added or replaced."""
        return CurveSet(
            {**self._curves, name: curve}, self.discount, self.projections, self.fixings
        )

    def bumped(self, shift: float, names: Union[None, str, Sequence[str]] = None) -> "CurveSet":
        """Copy with a parallel continuously compounded zero shift on ``names`` (default all)."""
        if names is None:
            targets = list(self._curves)
        elif isinstance(names, str):
            targets = [names]
        else:
            targets = list(names)
        for n in targets:
            if n not in self._curves:
                raise KeyError(n)
        curves = {n: (c.shifted(shift) if n in targets else c) for n, c in self._curves.items()}
        return CurveSet(curves, self.discount, self.projections, self.fixings)

    @classmethod
    def from_fit(
        cls,
        result: Any,
        discount: str,
        projections: Optional[Mapping[str, str]] = None,
        fixings: Optional[Mapping[str, FixingHistory]] = None,
    ) -> "CurveSet":
        """Set from any fit result with a ``curves`` mapping (bootstrap, global fit, NSS, ...)."""
        return cls(result.curves, discount, projections, fixings)

    def __repr__(self) -> str:
        return f"CurveSet({sorted(self._curves)}, discount={self.discount!r}, projections={self.projections!r})"
