r"""Arbitrage-free SABR: the effective density PDE of Hagan et al. (2014).

Hagan, Kumar, Lesniewski and Woodward (*Arbitrage-free SABR*, Wilmott
2014) integrate out the volatility to obtain a one-dimensional
Fokker–Planck equation for the density :math:`Q(T, F)` of the forward:

.. math::
    \partial_T Q = \partial_{FF}\bigl[M(T, F)\,Q\bigr], \qquad
    M = \tfrac12 D^2(F)\,E(T, F),

with

.. math::
    C(F) &= (F + s)^\beta, \qquad y(F) = \int_f^F \frac{du}{C(u)}, \qquad
    D(F) = \sqrt{\alpha^2 + 2\rho\alpha\nu y + \nu^2 y^2}\;C(F), \\
    E(T, F) &= e^{\rho\nu\alpha\Gamma(F) T}, \qquad \Gamma(F) = \frac{C(F) - C(f)}{F - f}.

The forward is absorbed at :math:`F_{\min} = -s`, where :math:`C` vanishes.
The mass that reaches either end of the grid is stored as point masses
:math:`P_L, P_R`, whose rates are the outgoing fluxes. The PDE reproduces
Hagan's implied-vol expansion to the same order, but its prices come
from a genuine density, so there is no butterfly arbitrage at low
strikes.

Discretisation (conservative, as in the paper): cells of width
:math:`h` centred at :math:`F_j = F_{\min} + (j - \tfrac12)h`, with the
forward placed at a cell centre. Ghost values give
:math:`MQ = 0` at both edges, so

.. math:: \frac{dQ_j}{dT} = \frac{(MQ)_{j+1} - 2(MQ)_j + (MQ)_{j-1}}{h^2}, \qquad
          \frac{dP_L}{dT} = \frac{2(MQ)_1}{h}, \qquad \frac{dP_R}{dT} = \frac{2(MQ)_J}{h}.

Total mass :math:`h\sum Q_j + P_L + P_R` and the forward
:math:`h\sum F_jQ_j + F_{\min}P_L + F_{\max}P_R` are conserved exactly by
every step. Time stepping uses Lawson–Swayne: two implicit Euler steps
of length :math:`b\,\delta`, :math:`b = 1 - \sqrt2/2`, combined as
:math:`(\sqrt2 + 1)\theta^{n+2b} - \sqrt2\,\theta^{n+b}`. The scheme is
second order and L-stable, so the Dirac start does not cause
Crank–Nicolson oscillations. The grid spans :math:`\pm n_{sd}\sqrt T` in
the variable :math:`z` of the paper (mapped to :math:`F` via
:math:`y(z) = \frac{\alpha}{\nu}[\sinh\nu z + \rho(\cosh\nu z - 1)]`), cut
at :math:`F_{\min}`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import solve_banded

from spxa.market.implied import implied_black_volatility, implied_normal_volatility
from spxa.rates.sabr import SABR

__all__ = ["ArbitrageFreeSABR", "SABRDensity"]


@dataclass(frozen=True)
class SABRDensity:
    """Density of the forward at expiry on cells, plus the absorbed masses."""

    F: float
    T: float
    shift: float
    centers: np.ndarray
    h: float
    Q: np.ndarray
    P_L: float
    P_R: float

    @property
    def f_min(self) -> float:
        return float(self.centers[0] - 0.5 * self.h)

    @property
    def f_max(self) -> float:
        return float(self.centers[-1] + 0.5 * self.h)

    def mass(self) -> float:
        return float(self.h * self.Q.sum() + self.P_L + self.P_R)

    def mean(self) -> float:
        return float(
            self.h * np.dot(self.centers, self.Q) + self.f_min * self.P_L + self.f_max * self.P_R
        )

    def call(self, K: Any) -> Any:
        """Undiscounted call value :math:`E[(F_T - K)^+]` (piecewise-constant density, exact per cell)."""
        KK = np.atleast_1d(np.asarray(K, float))
        left = self.centers - 0.5 * self.h
        right = self.centers + 0.5 * self.h
        out = np.empty(KK.shape)
        for i, k in enumerate(KK):
            full = left >= k
            part = (left < k) & (right > k)
            v = self.h * np.sum(self.Q[full] * (self.centers[full] - k))
            v += np.sum(self.Q[part] * 0.5 * (right[part] - k) ** 2)
            v += max(self.f_max - k, 0.0) * self.P_R + max(self.f_min - k, 0.0) * self.P_L
            out[i] = v
        return float(out[0]) if np.ndim(K) == 0 else out

    def put(self, K: Any) -> Any:
        """Put by parity (mass and mean are exact)."""
        return np.asarray(self.call(K)) - (self.mean() - np.asarray(K, float))

    def implied_vol(self, K: Any, vol: str = "lognormal") -> Any:
        """Black (of the shifted rate) or normal implied volatility of out-of-the-money options."""
        KK = np.asarray(K, float)
        opt = np.where(KK >= self.F, "call", "put")
        price = np.where(KK >= self.F, self.call(KK), self.put(KK))
        if vol == "normal":
            return implied_normal_volatility(price, self.F, KK, self.T, option=opt)
        s = self.shift
        return implied_black_volatility(price, self.F + s, KK + s, self.T, option=opt)


class ArbitrageFreeSABR:
    """Solve the Hagan et al. (2014) density PDE for one expiry."""

    def __init__(
        self, model: SABR, F: float, T: float, n: int = 400, n_steps: int = 100, n_sd: float = 4.0
    ) -> None:
        if T <= 0 or n < 10 or n_steps < 1:
            raise ValueError("need T > 0, n >= 10 and n_steps >= 1")
        if F + model.shift <= 0:
            raise ValueError("forward must exceed -shift")
        self.model, self.F, self.T = model, float(F), float(T)
        self.n, self.n_steps, self.n_sd = int(n), int(n_steps), float(n_sd)

    # model functions
    def _C(self, F: np.ndarray) -> np.ndarray:
        return np.maximum(F + self.model.shift, 0.0) ** self.model.beta

    def _y(self, F: np.ndarray) -> np.ndarray:
        b = 1.0 - self.model.beta
        s = self.model.shift
        Fs = np.maximum(F + s, 0.0)
        f = self.F + s
        if b == 0.0:
            return np.log(np.maximum(Fs, 1e-300) / f)
        return (Fs**b - f**b) / b

    def _F_of_y(self, y: float) -> float:
        b = 1.0 - self.model.beta
        s = self.model.shift
        f = self.F + s
        if b == 0.0:
            return f * np.exp(y) - s
        base = f**b + b * y
        return (max(base, 0.0)) ** (1.0 / b) - s

    def _y_of_z(self, z: float) -> float:
        m = self.model
        if m.nu == 0.0:
            return m.alpha * z
        return m.alpha / m.nu * (np.sinh(m.nu * z) + m.rho * (np.cosh(m.nu * z) - 1.0))

    def bounds(self) -> tuple[float, float]:
        zs = self.n_sd * np.sqrt(self.T)
        lo = max(self._F_of_y(self._y_of_z(-zs)), -self.model.shift)
        hi = self._F_of_y(self._y_of_z(zs))
        return lo, hi

    def solve(self) -> SABRDensity:
        m = self.model
        f_min, f_max = self.bounds()
        # put the forward at a cell centre
        h0 = (f_max - f_min) / self.n
        j0 = max(1, int(round((self.F - f_min) / h0 + 0.5)))
        h = (self.F - f_min) / (j0 - 0.5)
        J = int(np.ceil((f_max - f_min) / h))
        F = f_min + (np.arange(1, J + 1) - 0.5) * h
        y = self._y(F)
        D2 = (m.alpha**2 + 2 * m.rho * m.alpha * m.nu * y + m.nu**2 * y**2) * self._C(F) ** 2
        Cf = float(self._C(np.array([self.F]))[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            gamma = np.where(np.abs(F - self.F) > 1e-14, (self._C(F) - Cf) / (F - self.F), 0.0)
        if m.beta > 0:
            # derivative of C at the forward where the difference quotient is 0/0
            at = np.abs(F - self.F) <= 1e-14
            gamma[at] = m.beta * (self.F + m.shift) ** (m.beta - 1)
        x = np.zeros(J + 2)
        x[j0] = 1.0 / h  # state: [P_L, Q_1..Q_J, P_R]
        dt = self.T / self.n_steps
        b = 1.0 - np.sqrt(2.0) / 2.0

        def implicit(xp: np.ndarray, t: float, tau: float) -> np.ndarray:
            M = 0.5 * D2 * np.exp(m.rho * m.nu * m.alpha * gamma * t)
            ab = np.zeros((3, J + 2))
            c = tau / (h * h)
            # Q rows: Q_j - c (M_{j+1}Q_{j+1} - 2 M_j Q_j + M_{j-1} Q_{j-1}) with ghost M_0Q_0 = -M_1Q_1
            diag_q = 1.0 + 2.0 * c * M
            diag_q[0] += c * M[0]
            diag_q[-1] += c * M[-1]
            ab[1, 1:-1] = diag_q
            ab[0, 2:-1] = -c * M[1:]  # super-diagonal: row j uses Q_{j+1}
            ab[2, 1:-2] = -c * M[:-1]  # sub-diagonal: row j+1 uses Q_j
            # P_L and P_R rows
            ab[1, 0] = 1.0
            ab[1, -1] = 1.0
            ab[0, 1] = -2.0 * tau * M[0] / h  # row 0 uses Q_1
            ab[2, -2] = -2.0 * tau * M[-1] / h  # row J+1 uses Q_J
            return solve_banded((1, 1), ab, xp)

        t = 0.0
        for _ in range(self.n_steps):
            x1 = implicit(x, t + b * dt, b * dt)
            x2 = implicit(x1, t + 2 * b * dt, b * dt)
            x = (np.sqrt(2.0) + 1.0) * x2 - np.sqrt(2.0) * x1
            t += dt
        return SABRDensity(self.F, self.T, m.shift, F, h, x[1:-1], float(x[0]), float(x[-1]))
