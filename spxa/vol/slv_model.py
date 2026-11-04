r"""
Calibrated stochastic local volatility models and the mixing fraction.

:func:`calibrate_slv` scales the vol-of-variance of a Heston model by the
*mixing fraction* :math:`\eta \in [0, 1]` and calibrates the leverage to a
local volatility surface, so that every :math:`\eta` reprices the same
vanillas. :math:`\eta = 0` makes the variance deterministic and the model
collapses to local volatility; :math:`\eta = 1` keeps the full stochastic
volatility, with the leverage correcting the residual smile. Path-dependent
prices (barriers, forward-starting options) move with :math:`\eta`, which is
how the mixing fraction is set in practice: from barrier or forward-skew
quotes, the vanillas being matched whatever its value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from spxa.vol.heston import Heston
from spxa.vol.slv import LeverageSurface, SLVGrid, slv_grid
from spxa.vol.slv_fp import LeverageCalibration, calibrate_leverage
from spxa.vol.slv_mc import SLVPaths, slv_paths
from spxa.vol.slv_pde import SLVPDE

__all__ = ["SLVModel", "as_local_vol", "calibrate_slv"]


def as_local_vol(
    local_vol: Any, T: float, n_y: int = 601, n_t: int = 121, y_max: float = 1.5
) -> Any:
    """Tabulate the local volatility of a :class:`VolSurface` on a fine grid (other inputs pass through).

    Times are geometric from ``1e-3 T`` to ``T``: the local volatility
    changes fastest at short expiries. The grid is linear in the local
    variance, so it must be fine in ``y`` (the bias of a 121-point grid
    over the same range is several basis points).
    """
    from spxa.vol.local_vol import LocalVolGrid, LocalVolSurface
    from spxa.vol.surface import VolSurface

    if isinstance(local_vol, VolSurface):
        lv = LocalVolSurface(local_vol)
        ys = np.linspace(-y_max, y_max, n_y)
        return LocalVolGrid.from_surface(lv, ys, np.geomspace(1e-3 * T, T, n_t))
    return local_vol


@dataclass(frozen=True)
class SLVModel:
    """A Heston model scaled by ``eta`` with its calibrated leverage."""

    base: Heston
    eta: float
    model: Heston
    leverage: LeverageSurface
    calibration: LeverageCalibration

    def paths(self, T: float, n_steps: int, n_paths: int, **kwargs: Any) -> SLVPaths:
        """:func:`slv_paths` for this model."""
        return slv_paths(self.model, self.leverage, T, n_steps, n_paths, **kwargs)

    def pde(self, r: float = 0.0, q: float = 0.0, **kwargs: Any) -> SLVPDE:
        """:class:`SLVPDE` for this model."""
        return SLVPDE(self.model, self.leverage, r, q, **kwargs)

    def smile(self, k: Any, t: float) -> Any:
        """Implied vols of the calibrated density at a recorded time."""
        return self.calibration.smile(k, t)


def calibrate_slv(
    local_vol: Any,
    heston: Heston,
    eta: float,
    T: float,
    n_y: int = 121,
    n_v: int = 61,
    n_time: int = 100,
    record: tuple[float, ...] = (),
    grid: Optional[SLVGrid] = None,
    vol: float = 0.2,
) -> SLVModel:
    """Scale ``heston`` by the mixing fraction ``eta`` and calibrate the leverage to ``local_vol``.

    ``local_vol`` may be a :class:`VolSurface`, whose Dupire local
    volatility is then tabulated by :func:`as_local_vol`.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError("the mixing fraction must lie in [0, 1]")
    local_vol = as_local_vol(local_vol, T)
    m = heston.scaled(eta)
    g = grid if grid is not None else slv_grid(m, T, n_y, n_v, vol=vol)
    cal = calibrate_leverage(local_vol, m, T, g, n_time, record=record)
    return SLVModel(heston, float(eta), m, cal.leverage, cal)
