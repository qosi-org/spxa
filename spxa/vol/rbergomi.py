r"""
Rough Bergomi by the hybrid scheme, with a conditional Monte Carlo pricer.

The rough Bergomi model (Bayer, Friz & Gatheral, 2016) prices off a
forward-variance curve :math:`\xi_0(t)`:

.. math::
    v_t = \xi_0(t)\exp\bigl(\eta Y_t - \tfrac12\eta^2\operatorname{Var}Y_t\bigr), \qquad
    Y_t = \sqrt{2H}\int_0^t (t - s)^{H - 1/2}\,dW_s,

with :math:`dS_t/S_t = \sqrt{v_t}\,(\rho\,dW_t + \sqrt{1 - \rho^2}\,dW^\perp_t)`.

:func:`rbergomi_paths` simulates :math:`Y` with the hybrid scheme of
Bennedsen, Lunde & Pakkanen (2017) with :math:`\kappa = 1`. The singular
first cell is sampled exactly, jointly with the Brownian increment, and the
rest of the kernel is evaluated at the optimal points
:math:`b_k = \bigl((k^{\alpha+1} - (k-1)^{\alpha+1})/(\alpha + 1)\bigr)^{1/\alpha}`
by FFT convolution. The compensator uses the scheme's own variance of
:math:`Y`, so :math:`E[v_t] = \xi_0(t)` holds exactly on the grid.

Conditionally on :math:`W`, :math:`\log S_T` is Gaussian (Romano & Touzi,
1997). :meth:`RBergomiPaths.price` therefore averages Black–Scholes prices
with spot :math:`S_0\exp(\rho\int\sqrt v\,dW - \tfrac12\rho^2\int v\,dt)` and
total variance :math:`(1 - \rho^2)\int v\,dt`. This removes the noise of
:math:`W^\perp` entirely and is the "mixed" estimator of McCrickerd &
Pakkanen (2018).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
from scipy.signal import fftconvolve

from scipy.special import ndtr
from spxa.market.implied import implied_black_volatility

__all__ = ["RBergomiPaths", "hybrid_weights", "rbergomi_paths"]

Curve = Union[float, Callable[[np.ndarray], np.ndarray]]


def hybrid_weights(hurst: float, dt: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Kernel weights :math:`(b_k\Delta)^\alpha` for :math:`k = 2..n` and the scheme variance of :math:`Y_{t_i}/\sqrt{2H}`."""
    a = hurst - 0.5
    k = np.arange(2, n + 1, dtype=float)
    b = ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1.0 / a)
    w = (b * dt) ** a
    first = dt ** (2 * a + 1) / (2 * a + 1)
    var = np.r_[0.0, first + np.r_[0.0, np.cumsum(w * w * dt)]]  # i = 0..n
    return w, var


def _black(F: np.ndarray, K: float, sd: np.ndarray, eta: float) -> np.ndarray:
    # plain Black formula: path-wise values are averaged, so the cancellation
    # repair of spxa.market.black (costly on deep out-of-the-money paths) is not needed
    d1 = np.log(F / K) / sd + 0.5 * sd
    return eta * (F * ndtr(eta * d1) - K * ndtr(eta * (d1 - sd)))


@dataclass(frozen=True)
class RBergomiPaths:
    """Terminal quantities of simulated paths: :math:`\\int v\\,dt`, :math:`\\int\\sqrt v\\,dW`,
    :math:`S_T`, and optionally the variance paths."""

    s0: float
    rho: float
    T: float
    int_v: np.ndarray
    int_sqrt_v_dw: np.ndarray
    spot: np.ndarray
    times: np.ndarray
    variance: Optional[np.ndarray] = None

    def price(
        self, K: Any, option: str = "call", conditional: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prices and standard errors at strikes ``K`` (undiscounted, on the forward)."""
        Ks = np.atleast_1d(np.asarray(K, float))
        if conditional:
            r = self.rho
            s_eff = self.s0 * np.exp(r * self.int_sqrt_v_dw - 0.5 * r * r * self.int_v)
            tv = np.maximum((1 - r * r) * self.int_v, 1e-300)
            sd = np.sqrt(tv)
            eta = 1.0 if option == "call" else -1.0
            vals = np.array([_black(s_eff, k, sd, eta) for k in Ks])
        else:
            sign = 1.0 if option == "call" else -1.0
            vals = np.maximum(sign * (self.spot[None, :] - Ks[:, None]), 0.0)
        n = vals.shape[1]
        return vals.mean(axis=1), vals.std(axis=1, ddof=1) / math.sqrt(n)

    def implied_vol(self, K: Any) -> np.ndarray:
        """Implied vols of the conditional out-of-the-money prices."""
        Ks = np.atleast_1d(np.asarray(K, float))
        calls = self.price(Ks, "call")[0]
        puts = calls - (self.s0 - Ks)
        otm = np.where(Ks >= self.s0, calls, puts)
        kind = np.where(Ks >= self.s0, "call", "put")
        return np.asarray(implied_black_volatility(otm, self.s0, Ks, self.T, 1.0, kind), float)


_BLOCK = 4096


def rbergomi_paths(
    hurst: float,
    eta: float,
    rho: float,
    xi: Curve,
    T: float,
    n_steps: int,
    n_paths: int,
    s0: float = 1.0,
    seed: Optional[int] = None,
    keep_variance: bool = False,
) -> RBergomiPaths:
    """Simulate rough Bergomi on ``n_steps`` equal steps (antithetic in all Brownian drivers)."""
    if not 0 < hurst < 0.5:
        raise ValueError("the hybrid scheme needs 0 < H < 1/2")
    if not -1 < rho < 1:
        raise ValueError("rho must lie in (-1, 1)")
    n, dt = int(n_steps), T / int(n_steps)
    a = hurst - 0.5
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2
    # (dW_j, int_{t_j}^{t_{j+1}} (t_{j+1} - s)^a dW_s) for j = 0..n-1
    cov = np.array(
        [[dt, dt ** (a + 1) / (a + 1)], [dt ** (a + 1) / (a + 1), dt ** (2 * a + 1) / (2 * a + 1)]]
    )
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((half, n, 2)) @ L.T
    zp = rng.standard_normal((half, n)) * math.sqrt(dt)
    w, var = hybrid_weights(hurst, dt, n)
    kernel = np.r_[0.0, 0.0, w][None, :]
    times = np.linspace(0.0, T, n + 1)
    xi_t = np.full(n + 1, float(xi)) if np.isscalar(xi) else np.asarray(xi(times), float)  # type: ignore[operator, arg-type]
    comp = 0.5 * eta * eta * 2 * hurst * var
    int_v, int_sv_dw, log_s = np.empty(n_paths), np.empty(n_paths), np.empty(n_paths)
    v_all = np.empty((n_paths, n + 1)) if keep_variance else None
    # paths are [z, -z][:n_paths]; process in blocks to bound memory (the FFT is the peak)
    for sign, offset in ((1.0, 0), (-1.0, half)):
        for lo in range(0, half, _BLOCK):
            hi = min(lo + _BLOCK, half, n_paths - offset)
            if hi <= lo:
                break
            dW, Wt = sign * z[lo:hi, :, 0], sign * z[lo:hi, :, 1]
            # Y_i / sqrt(2H) = Wt_{i-1} + sum_{k=2}^{i} w_k dW_{i-k}, i = 1..n
            conv = fftconvolve(dW, kernel, axes=1)[:, 1 : n + 1]
            Y = np.zeros((hi - lo, n + 1))
            Y[:, 1:] = math.sqrt(2 * hurst) * (Wt + conv)
            v = xi_t * np.exp(eta * Y - comp)
            sv = np.sqrt(v[:, :-1])
            rows = slice(offset + lo, offset + hi)
            int_v[rows] = v[:, :-1].sum(axis=1) * dt
            int_sv_dw[rows] = (sv * dW).sum(axis=1)
            dB = rho * dW + math.sqrt(1 - rho * rho) * (sign * zp[lo:hi])
            log_s[rows] = math.log(s0) + (sv * dB).sum(axis=1) - 0.5 * int_v[rows]
            if v_all is not None:
                v_all[rows] = v
    return RBergomiPaths(
        float(s0),
        float(rho),
        float(T),
        int_v,
        int_sv_dw,
        np.exp(log_s),
        times,
        v_all,
    )
