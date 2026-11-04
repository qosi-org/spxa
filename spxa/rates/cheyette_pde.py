r"""Two-dimensional PDE for the Cheyette model.

The value :math:`V(t, x, y)` of a claim solves

.. math::
    V_t + (y - \kappa x)V_x + \tfrac12\sigma_r(t, x)^2 V_{xx}
        + \bigl(\sigma_r(t, x)^2 - 2\kappa y\bigr)V_y - \bigl(f(0, t) + x\bigr)V = 0.

There is no diffusion in :math:`y`, so each backward step is split. First
:math:`y` is advected along its characteristic by semi-Lagrangian linear
interpolation, :math:`V(t, x, y) \leftarrow V(t + h, x, y + \mu_y h)`.
Then each :math:`y`-slice is solved implicitly in :math:`x` (Crank–Nicolson,
after two fully implicit Rannacher steps that damp payoff kinks), with
zero-curvature boundaries. The scheme is first order in time overall.
With :math:`\beta = 0` the slices decouple along the deterministic
:math:`y`, and the PDE reproduces Hull–White. Exercise dates apply
:math:`\max(V, g)` after the step that lands on them.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional

import numpy as np
from scipy.linalg import solve_banded

from spxa.market.curves import YieldCurve
from spxa.rates.cheyette import Cheyette
from spxa.rates.model_cashflows import swap_cashflows

__all__ = ["CheyettePDE"]

Payoff = Callable[[np.ndarray, np.ndarray], np.ndarray]


class CheyettePDE:
    def __init__(
        self, model: Cheyette, n_x: int = 161, n_y: int = 41, n_time: int = 200, n_sd: float = 6.0
    ) -> None:
        self.model = model
        self.n_x, self.n_y, self.n_time, self.n_sd = int(n_x), int(n_y), int(n_time), float(n_sd)

    def _grids(self, T: float) -> tuple[np.ndarray, np.ndarray]:
        m = self.model
        sd = math.sqrt(max(m.ybar(T), 1e-12))
        xmax = self.n_sd * sd
        x = np.linspace(-xmax, xmax, self.n_x)
        smax = float(np.max(m.local_vol(np.linspace(0.0, T, 16)[:, None], x[None, :])))
        ymax = smax**2 * -math.expm1(-2 * m.kappa * T) / (2 * m.kappa) * 1.05
        y = np.linspace(0.0, ymax, self.n_y)
        return x, y

    def solve(
        self,
        payoff: Payoff,
        T: float,
        exercise: Optional[Mapping[float, Payoff]] = None,
    ) -> float:
        """Value at :math:`(0, 0, 0)` of ``payoff(x, y)`` paid at ``T``.

        ``exercise`` maps times in :math:`(0, T)` to exercise values ``g(x, y)``;
        they are added to the time grid.
        """
        m = self.model
        x, y = self._grids(T)
        X, Y = np.meshgrid(x, y, indexing="ij")  # (n_x, n_y)
        V = np.asarray(payoff(X, Y), float)
        ex = {float(t): g for t, g in (exercise or {}).items() if 0.0 < t < T}
        # exercise dates are inserted into the uniform grid so they are hit exactly
        times = np.unique(np.r_[np.linspace(0.0, T, self.n_time + 1), list(ex)])
        times = times[np.r_[True, np.diff(times) > 1e-10 * T]]
        n = times.size - 1
        ex_steps = {int(np.argmin(np.abs(times - t))): g for t, g in ex.items()}
        for k in range(n, 0, -1):
            t0, t1 = times[k - 1], times[k]
            h = t1 - t0
            tm = 0.5 * (t0 + t1)
            s2 = np.asarray(m.local_vol(tm, x), float) ** 2  # (n_x,)
            # 1) advection in y along the characteristic
            target = Y + (s2[:, None] - 2 * m.kappa * Y) * h
            target = np.clip(target, y[0], y[-1])
            V = np.array([np.interp(target[i], y, V[i]) for i in range(self.n_x)])
            # 2) implicit / Crank-Nicolson in x for each y-slice
            theta = 1.0 if k > n - 2 else 0.5
            r = float(m.curve.instantaneous_forward(tm)) + x
            for j in range(self.n_y):
                V[:, j] = _x_step(V[:, j], x, y[j] - m.kappa * x, s2, r, h, theta)
            if k - 1 in ex_steps and k - 1 > 0:
                V = np.maximum(V, np.asarray(ex_steps[k - 1](X, Y), float))
        return float(np.interp(0.0, x, V[:, 0]))

    def zero_bond(self, T: float) -> float:
        return self.solve(lambda X, Y: np.ones_like(X), T)

    def swaption(self, swaption: Any, projection: Optional[YieldCurve] = None) -> float:
        """European physically settled swaption."""
        m = self.model
        crv = m.curve
        T = float(crv.time(swaption.expiry))
        cf = swap_cashflows(swaption.swap, crv, projection)
        live = cf.times > T + 1e-12
        ts, am, const = cf.times[live], cf.amounts[live], float(cf.amounts[~live].sum())

        def g(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return np.maximum(
                const + sum(a * m.discount_bond(T, ti, X, Y) for ti, a in zip(ts, am)), 0.0
            )

        return self.solve(g, T)


def _x_step(
    v: np.ndarray,
    x: np.ndarray,
    mu: np.ndarray,
    s2: np.ndarray,
    r: np.ndarray,
    h: float,
    theta: float,
) -> np.ndarray:
    """One theta-step of v_t + mu v_x + s2/2 v_xx - r v = 0 with zero curvature at both ends.

    The boundary values are eliminated through v_0 = 2v_1 - v_2 and
    v_N = 2v_{N-1} - v_{N-2}, which leaves a tridiagonal system on the interior.
    """
    hx = x[1] - x[0]
    lo = (0.5 * s2 / hx**2 - 0.5 * mu / hx)[1:-1]
    di = (-s2 / hx**2 - r)[1:-1]
    up = (0.5 * s2 / hx**2 + 0.5 * mu / hx)[1:-1]
    # fold the extrapolated boundary values into the first and last interior rows
    di, lo, up = di.copy(), lo.copy(), up.copy()
    di[0] += 2 * lo[0]
    up[0] -= lo[0]
    di[-1] += 2 * up[-1]
    lo[-1] -= up[-1]
    lo[0] = up[-1] = 0.0
    u = v[1:-1]
    Lu = di * u
    Lu[1:] += lo[1:] * u[:-1]
    Lu[:-1] += up[:-1] * u[1:]
    rhs = u + (1 - theta) * h * Lu
    ab = np.zeros((3, u.size))
    ab[0, 1:] = -theta * h * up[:-1]
    ab[1] = 1 - theta * h * di
    ab[2, :-1] = -theta * h * lo[1:]
    w = solve_banded((1, 1), ab, rhs)
    out = np.empty_like(v)
    out[1:-1] = w
    out[0] = 2 * w[0] - w[1]
    out[-1] = 2 * w[-1] - w[-2]
    return out
