"""A common interface for analytic short-rate models.

:class:`~spxa.rates.hull_white.HullWhite`,
:class:`~spxa.rates.g2pp.G2PP` and :class:`~spxa.rates.cir_pp.CIRPlusPlus`
all fit a discount curve exactly. They price zero-bond options, caps and
physically settled swaptions in closed or semi-closed form, so
calibration and model comparison need only :class:`ShortRateModel`.
:func:`~spxa.rates.hw_calibration.model_price` dispatches on the instrument
for any of them. :func:`model_implied_vols`
quotes model prices as Black (or Bachelier) volatilities, the usual way to
compare a model's smile and term structure with the market.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from spxa.market.curves import YieldCurve
from spxa.rates.hw_calibration import model_price
from spxa.rates.swaptions import Swaption

__all__ = ["ShortRateModel", "model_implied_vols"]


@runtime_checkable
class ShortRateModel(Protocol):
    curve: YieldCurve

    def zcb_option(self, T: float, S: float, K: Any, option: str = "call") -> Any: ...

    def caplet(self, caplet: Any, projection: Optional[YieldCurve] = None) -> float: ...

    def cap(self, cap: Any, projection: Optional[YieldCurve] = None) -> float: ...

    def swaption(self, swaption: Any, projection: Optional[YieldCurve] = None) -> float: ...


def model_implied_vols(
    model: ShortRateModel,
    instruments: Sequence[Any],
    projection: Optional[YieldCurve] = None,
    quote: str = "black",
    shift: float = 0.0,
) -> np.ndarray:
    """Flat (cap) or swaption implied volatilities of ``model`` prices."""
    if quote not in ("black", "normal"):
        raise ValueError("quote must be 'black' or 'normal'")
    bounds = (1e-7, 5.0) if quote == "black" else (1e-7, 0.2)
    out = []
    for inst in instruments:
        p = model_price(model, inst, projection)
        if isinstance(inst, Swaption):
            out.append(inst.implied_vol(p, model.curve, projection, quote, shift, bounds))
        else:
            out.append(inst.implied_flat_vol(p, model.curve, projection, quote, shift, bounds))
    return np.array(out)
