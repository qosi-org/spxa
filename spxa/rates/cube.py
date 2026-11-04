r"""Swaption volatility cube with SABR smiles.

Each (expiry, tenor) node of the ATM matrix carries a calibrated SABR
smile with a common :math:`\beta`. Off the nodes the parameters are
interpolated bilinearly in (expiry, tenor): :math:`\log\alpha`, :math:`\rho`
and :math:`\log\nu`, so :math:`\alpha, \nu > 0` and :math:`|\rho| < 1`
are kept. Beyond the matrix they are held flat. The forward swap rate of
the queried swaption is an input, since it comes from the curves, not
from the cube.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from spxa.rates.sabr import SABR, calibrate_sabr

__all__ = ["SwaptionCube"]


def _weights(grid: np.ndarray, x: float) -> tuple[int, int, float]:
    if grid.size == 1 or x <= grid[0]:
        return 0, 0, 0.0
    if x >= grid[-1]:
        return grid.size - 1, grid.size - 1, 0.0
    j = int(np.searchsorted(grid, x) - 1)
    return j, j + 1, float((x - grid[j]) / (grid[j + 1] - grid[j]))


@dataclass(frozen=True)
class SwaptionCube:
    """SABR parameters on an expiry x tenor matrix (``models[i][j]`` at ``expiries[i]``, ``tenors[j]``)."""

    expiries: np.ndarray
    tenors: np.ndarray
    models: tuple[tuple[SABR, ...], ...]

    def __post_init__(self) -> None:
        if len(self.models) != self.expiries.size or any(
            len(r) != self.tenors.size for r in self.models
        ):
            raise ValueError("models must have shape (len(expiries), len(tenors))")
        if np.any(np.diff(self.expiries) <= 0) or np.any(np.diff(self.tenors) <= 0):
            raise ValueError("expiries and tenors must be increasing")
        betas = {m.beta for r in self.models for m in r}
        shifts = {m.shift for r in self.models for m in r}
        if len(betas) != 1 or len(shifts) != 1:
            raise ValueError("all nodes must share beta and shift")

    def sabr(self, expiry: float, tenor: float) -> SABR:
        """Interpolated SABR parameters."""
        i0, i1, a = _weights(self.expiries, expiry)
        j0, j1, b = _weights(self.tenors, tenor)
        corners = (
            (i0, j0, (1 - a) * (1 - b)),
            (i1, j0, a * (1 - b)),
            (i0, j1, (1 - a) * b),
            (i1, j1, a * b),
        )
        la = sum(w * np.log(self.models[i][j].alpha) for i, j, w in corners)
        rho = sum(w * self.models[i][j].rho for i, j, w in corners)
        ln = sum(w * np.log(max(self.models[i][j].nu, 1e-300)) for i, j, w in corners)
        m0 = self.models[0][0]
        return SABR(float(np.exp(la)), m0.beta, float(rho), float(np.exp(ln)), m0.shift)

    def vol(
        self, expiry: float, tenor: float, K: Any, F: float, vol_type: str = "lognormal"
    ) -> Any:
        m = self.sabr(expiry, tenor)
        return m.normal_vol(F, K, expiry) if vol_type == "normal" else m.lognormal_vol(F, K, expiry)

    @classmethod
    def calibrate(
        cls,
        expiries: Sequence[float],
        tenors: Sequence[float],
        forwards: Any,
        strikes: Any,
        vols: Any,
        beta: float,
        shift: float = 0.0,
        vol_type: str = "lognormal",
    ) -> "SwaptionCube":
        """Calibrate one SABR smile per node.

        ``forwards[i][j]`` is the node's forward swap rate, ``strikes[i][j]``
        and ``vols[i][j]`` its smile quotes.
        """
        rows = []
        for i, T in enumerate(expiries):
            row = []
            for j in range(len(tenors)):
                fit = calibrate_sabr(
                    float(forwards[i][j]),
                    strikes[i][j],
                    vols[i][j],
                    float(T),
                    beta,
                    shift,
                    vol_type,
                )
                row.append(fit.model)
            rows.append(tuple(row))
        return cls(np.asarray(expiries, float), np.asarray(tenors, float), tuple(rows))
