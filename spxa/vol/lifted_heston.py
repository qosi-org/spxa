r"""
The lifted Heston model (Abi Jaber, 2019): a Markovian approximation of rough Heston.

The fractional kernel :math:`t^{\alpha-1}/\Gamma(\alpha)` is replaced by a
sum of exponentials :math:`\sum_i c_i e^{-x_i t}`, which turns the Volterra
variance into a combination of :math:`n` ordinary factors,

.. math::
    V_t = g_0(t) + \sum_{i=1}^n c_i U^i_t, \qquad
    dU^i_t = (-x_i U^i_t - \lambda V_t)\,dt + \nu\sqrt{V_t}\,dW_t, \quad U^i_0 = 0,

with :math:`g_0(t) = V_0 + \lambda\theta\sum_i c_i(1 - e^{-x_i t})/x_i`. The
weights and speeds come from a geometric partition with ratio
:math:`r_n = 1 + 10 n^{-0.9}`:

.. math::
    c_i = \frac{(r_n^{1-\alpha} - 1)\,r_n^{(\alpha - 1)(1 + n/2)}}{\Gamma(\alpha)\Gamma(2 - \alpha)}\,r_n^{(1-\alpha)i},
    \qquad
    x_i = \frac{1 - \alpha}{2 - \alpha}\,\frac{r_n^{2-\alpha} - 1}{r_n^{1-\alpha} - 1}\,r_n^{i - 1 - n/2}.

The characteristic function is
:math:`\exp\int_0^T F(u, \sum_j c_j\psi_j(s))\,g_0(T - s)\,ds` with the
Riccati system :math:`\psi_i' = -x_i\psi_i + F(u, \sum_j c_j\psi_j)`. The
fast factors make it stiff, so it is integrated by a second-order
exponential Runge–Kutta (ETD2) scheme that treats :math:`-x_i\psi_i` exactly.
One factor with :math:`c = 1, x = 0` is the classical Heston model.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.special import gamma

from spxa.vol.rough_heston import FourierPricing, _rhs

__all__ = ["LiftedHeston", "lifted_weights"]


def lifted_weights(
    n: int, hurst: float, r: Optional[float] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Weights ``c`` and speeds ``x`` of Abi Jaber's geometric partition (``r`` defaults to ``1 + 10 n^-0.9``)."""
    a = hurst + 0.5
    rn = 1.0 + 10.0 * n**-0.9 if r is None else float(r)
    i = np.arange(1, n + 1)
    c = (
        (rn ** (1 - a) - 1)
        * rn ** ((a - 1) * (1 + n / 2))
        / (gamma(a) * gamma(2 - a))
        * rn ** ((1 - a) * i)
    )
    x = (1 - a) / (2 - a) * (rn ** (2 - a) - 1) / (rn ** (1 - a) - 1) * rn ** (i - 1 - n / 2)
    return c, x


def _phi1(z: np.ndarray, h: float) -> np.ndarray:
    # (1 - e^{-z h}) / z, with the z -> 0 limit h
    zh = z * h
    out = np.full_like(z, h)
    big = np.abs(zh) > 1e-8
    out[big] = -np.expm1(-zh[big]) / z[big]
    return out


def _phi2(z: np.ndarray, h: float) -> np.ndarray:
    # (h - phi1) / (z h), with the z -> 0 limit h / 2
    zh = z * h
    out = np.full_like(z, 0.5 * h)
    big = np.abs(zh) > 1e-5
    out[big] = (h - _phi1(z[big], h)) / zh[big]
    return out


class LiftedHeston(FourierPricing):
    """Lifted Heston with ``n`` factors (or explicit ``weights``/``speeds``)."""

    def __init__(
        self,
        v0: float,
        theta: float,
        lam: float,
        nu: float,
        rho: float,
        hurst: float,
        n: int = 20,
        weights: Optional[Sequence[float]] = None,
        speeds: Optional[Sequence[float]] = None,
    ) -> None:
        if not -1 < rho < 1 or nu <= 0 or min(v0, theta, lam) < 0:
            raise ValueError("need -1 < rho < 1, nu > 0 and v0, theta, lam >= 0")
        self.v0, self.theta, self.lam = float(v0), float(theta), float(lam)
        self.nu, self.rho, self.hurst = float(nu), float(rho), float(hurst)
        if weights is None or speeds is None:
            self.c, self.x = lifted_weights(int(n), self.hurst)
        else:
            self.c, self.x = np.asarray(weights, float), np.asarray(speeds, float)
            if self.c.shape != self.x.shape or np.any(self.x < 0):
                raise ValueError("weights and speeds must match, with speeds >= 0")

    @property
    def n(self) -> int:
        return int(self.c.size)

    def g0(self, t: Any) -> Any:
        r""":math:`g_0(t) = V_0 + \lambda\theta\sum_i c_i(1 - e^{-x_i t})/x_i`, also :math:`E[V_t]`."""
        tt = np.atleast_1d(np.asarray(t, float))
        out = self.v0 + self.lam * self.theta * np.array(
            [self.c @ _phi1(self.x, float(s)) for s in tt]
        )
        return float(out[0]) if np.ndim(t) == 0 else out.reshape(np.shape(t))

    def expected_variance(self, t: Any) -> Any:
        r""":math:`E[V_t]`, from the linear mean dynamics of the factors on 400 steps."""
        tt = np.atleast_1d(np.asarray(t, float))
        T = float(tt.max()) if tt.size else 0.0
        if T == 0.0:
            out = np.full(tt.shape, self.v0)
        else:
            m = 400
            grid = np.linspace(0.0, T, m + 1)
            h = T / m
            ev = np.empty(m + 1)
            ev[0] = self.v0
            # factors E[U^i]: dE = (-x E - lam E[V]) dt, exponential Euler (trapezoid in E[V])
            Ui = np.zeros(self.n)
            e = np.exp(-self.x * h)
            p1 = _phi1(self.x, h)
            for k in range(m):
                # implicit in E[V_{k+1}] (linear): V = g0 + c.(e U + p1 * (-lam)(V_k + V)/2)
                rhs = float(self.g0(grid[k + 1])) + self.c @ (e * Ui - 0.5 * self.lam * p1 * ev[k])
                ev[k + 1] = rhs / (1.0 + 0.5 * self.lam * float(self.c @ p1))
                Ui = e * Ui - 0.5 * self.lam * p1 * (ev[k] + ev[k + 1])
            out = np.interp(tt, grid, ev)
        return float(out[0]) if np.ndim(t) == 0 else out.reshape(np.shape(t))

    def log_cf(self, u: Any, T: float, n_steps: int = 200) -> np.ndarray:
        uu = np.atleast_1d(np.asarray(u, complex))
        h = T / int(n_steps)
        e = np.exp(-self.x * h)
        p1, p2 = _phi1(self.x, h), _phi2(self.x, h)
        psi = np.zeros((uu.size, self.n), complex)
        grid = np.linspace(0.0, T, int(n_steps) + 1)
        Fv = np.empty((uu.size, grid.size), complex)
        Fv[:, 0] = _rhs(uu, psi @ self.c, self.lam, self.nu, self.rho)
        for k in range(int(n_steps)):
            F0 = Fv[:, k][:, None]
            pred = e * psi + p1 * F0
            F1 = _rhs(uu, pred @ self.c, self.lam, self.nu, self.rho)[:, None]
            psi = e * psi + p1 * F0 + p2 * (F1 - F0)
            Fv[:, k + 1] = _rhs(uu, psi @ self.c, self.lam, self.nu, self.rho)
        g = np.asarray(self.g0(T - grid), float)
        return np.trapezoid(Fv * g, grid, axis=1)

    def cf(self, u: Any, T: float, n_steps: int = 200) -> np.ndarray:
        return np.exp(self.log_cf(u, T, n_steps))

    def __repr__(self) -> str:
        return f"LiftedHeston(n={self.n}, hurst={self.hurst}, v0={self.v0}, nu={self.nu}, rho={self.rho})"
