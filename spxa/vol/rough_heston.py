r"""
The rough Heston model (El Euch & Rosenbaum, 2019).

The variance solves the Volterra equation

.. math::
    V_t = V_0 + \frac{1}{\Gamma(\alpha)}\int_0^t (t - s)^{\alpha - 1}
        \bigl(\lambda(\theta - V_s)\,ds + \nu\sqrt{V_s}\,dW_s\bigr),
    \qquad \alpha = H + \tfrac12,

and the forward has :math:`dF_t/F_t = \sqrt{V_t}\,dB_t` with
:math:`d\langle W, B\rangle = \rho\,dt`. For :math:`H = 1/2` it is Heston
with :math:`\kappa = \lambda` and :math:`\xi = \nu`. For small :math:`H`
the ATM skew explodes like :math:`\tau^{H - 1/2}` at short maturities, as
observed on equity indices.

The characteristic function of :math:`X_T = \log(F_T/F_0)` is
:math:`\exp(\theta\lambda I^1 h(T) + V_0 I^{1 - \alpha}h(T))` where
:math:`h = I^\alpha F(u, h)` solves the fractional Riccati equation

.. math:: F(u, h) = -\tfrac12(u^2 + iu) + (i\rho\nu u - \lambda)h + \tfrac12\nu^2 h^2,

and :math:`I^{1-\alpha}h = I^1 F(u, h)`. :func:`fractional_riccati` solves it
for a whole vector of :math:`u` at once with the fractional Adams
predictor–corrector of Diethelm, Ford & Freed (the scheme of
:class:`spxa.rough.FractionalRiccatiSolver`, vectorised), whose error is
:math:`O(\Delta t^{\min(2, 1 + \alpha)})` for smooth solutions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import gamma

__all__ = ["RoughHeston", "fractional_riccati"]


def _rhs(u: np.ndarray, h: np.ndarray, lam: float, nu: float, rho: float) -> np.ndarray:
    return -0.5 * (u * u + 1j * u) + (1j * rho * nu * u - lam) * h + 0.5 * nu * nu * h * h


def fractional_riccati(
    u: Any, T: float, alpha: float, lam: float, nu: float, rho: float, n_steps: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    r"""Solve :math:`h = I^\alpha F(u, h)` on ``n_steps`` equal steps for every ``u``.

    Returns ``(t, h)`` with ``h`` of shape ``(len(u), n_steps + 1)``.
    """
    uu = np.atleast_1d(np.asarray(u, complex))[:, None]
    n = int(n_steps)
    dt = T / n
    a = float(alpha)
    h = np.zeros((uu.shape[0], n + 1), complex)
    f = np.zeros_like(h)
    f[:, 0] = _rhs(uu[:, 0], h[:, 0], lam, nu, rho)
    c_pred = dt**a / gamma(a + 1.0)
    c_corr = dt**a / gamma(a + 2.0)
    k = np.arange(n + 1, dtype=float)
    ka = k**a
    ka1 = k ** (a + 1.0)
    for m in range(1, n + 1):
        # predictor: b_{j,m} = (m - j)^a - (m - 1 - j)^a, j = 0..m-1
        b = ka[m:0:-1] - ka[m - 1 :: -1][:m]
        hp = c_pred * (f[:, :m] @ b)
        fp = _rhs(uu[:, 0], hp, lam, nu, rho)
        # corrector weights: a_0 = (m-1)^{a+1} - (m-1-a) m^a, a_j for j = 1..m-1
        a0 = ka1[m - 1] - (m - 1.0 - a) * ka[m]
        if m > 1:
            j = np.arange(1, m)
            aj = ka1[m - j + 1] - 2.0 * ka1[m - j] + ka1[m - j - 1]
            mid = f[:, 1:m] @ aj
        else:
            mid = 0.0
        h[:, m] = c_corr * (fp + a0 * f[:, 0] + mid)
        f[:, m] = _rhs(uu[:, 0], h[:, m], lam, nu, rho)
    return np.linspace(0.0, T, n + 1), h


class RoughHeston:
    """Rough Heston with initial variance ``v0``, long-run level ``theta``, mean-reversion
    ``lam``, vol-of-vol ``nu``, correlation ``rho`` and Hurst exponent ``hurst``."""

    def __init__(
        self, v0: float, theta: float, lam: float, nu: float, rho: float, hurst: float
    ) -> None:
        if min(v0, theta) < 0 or lam < 0 or nu <= 0:
            raise ValueError("need v0, theta, lam >= 0 and nu > 0")
        if not -1 < rho < 1:
            raise ValueError("rho must lie in (-1, 1)")
        if not 0 < hurst <= 0.5:
            raise ValueError("hurst must lie in (0, 1/2]")
        self.v0, self.theta, self.lam = float(v0), float(theta), float(lam)
        self.nu, self.rho, self.hurst = float(nu), float(rho), float(hurst)
        self.alpha = self.hurst + 0.5

    def __repr__(self) -> str:
        return (
            f"RoughHeston(v0={self.v0}, theta={self.theta}, lam={self.lam}, nu={self.nu},"
            f" rho={self.rho}, hurst={self.hurst})"
        )

    def log_cf(self, u: Any, T: float, n_steps: int = 200) -> np.ndarray:
        r""":math:`\log E[e^{iuX_T}]` for an array of (complex) ``u``."""
        t, h = fractional_riccati(u, T, self.alpha, self.lam, self.nu, self.rho, n_steps)
        uu = np.atleast_1d(np.asarray(u, complex))[:, None]
        f = _rhs(uu, h, self.lam, self.nu, self.rho)
        return self.theta * self.lam * np.trapezoid(h, t, axis=1) + self.v0 * np.trapezoid(
            f, t, axis=1
        )

    def cf(self, u: Any, T: float, n_steps: int = 200) -> np.ndarray:
        r"""Characteristic function of :math:`X_T = \log(F_T/F_0)`."""
        return np.exp(self.log_cf(u, T, n_steps))

    def expected_variance(self, t: Any) -> Any:
        r""":math:`E[V_t] = V_0 + (\theta - V_0)\,\bigl(1 - E_\alpha(-\lambda t^\alpha)\bigr)`,
        with the Mittag-Leffler function evaluated by its series (accurate for
        :math:`\lambda t^\alpha` up to about 10)."""
        tt = np.asarray(t, float)
        z = -self.lam * tt**self.alpha
        ml = np.zeros_like(z)
        for kk in range(400):
            add = z**kk / gamma(self.alpha * kk + 1.0)
            ml = ml + add
            if np.all(np.abs(add) < 1e-17):
                break
        out = self.v0 + (self.theta - self.v0) * (1.0 - ml)
        return float(out) if np.ndim(t) == 0 else out

    def heston_limit(self) -> Any:
        """The classical :class:`~spxa.vol.heston.Heston` model (requires ``hurst = 1/2``)."""
        from spxa.vol.heston import Heston

        if not math.isclose(self.hurst, 0.5):
            raise ValueError("only H = 1/2 is a classical Heston model")
        return Heston(self.v0, self.lam, self.theta, self.nu, self.rho)
