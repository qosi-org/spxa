r"""Swaptions in the Cheyette model by a displaced-diffusion approximation.

Andersen & Piterbarg (Vol. II, §13.1) project the swap rate onto a
displaced diffusion. Under the annuity measure

.. math:: dS_t = \frac{\partial S}{\partial x}(t, x_t, y_t)\,\sigma_r(t, x_t)\,dW^A_t.

Freezing :math:`y_t` at :math:`\bar y(t)` gives a local volatility
:math:`\varphi(t, x) = S_x(t, x, \bar y(t))\,\sigma_r(t, x)`. Linearising
it in :math:`S` around :math:`x = 0` gives

.. math::
    \varphi \approx \lambda(t)\bigl(b(t)\,S + (1 - b(t))\,S_0\bigr), \qquad
    \lambda(t) = \frac{\varphi(t, 0)}{S_0}, \qquad
    b(t) = \frac{S_0}{\varphi(t, 0)}\,\frac{\varphi_x(t, 0)}{S_x(t, 0)}.

Piterbarg's averaging then replaces the time-dependent skew by
:math:`\bar b = \int_0^T b(t)\,w(t)\,dt`, with weights
:math:`w(t) \propto \lambda(t)^2 v(t)` and
:math:`v(t) = \int_0^t \lambda^2`. The level becomes
:math:`\bar\lambda^2 T = v(T)`. The swaption is priced in Black's
formula on :math:`S + S_0(1 - \bar b)/\bar b` with volatility
:math:`\bar\lambda\bar b` (Bachelier with :math:`\bar\lambda S_0` as
:math:`\bar b \to 0`). The swap rate is :math:`S = K \pm V_K/A`, from the
model value :math:`V_K` of the swap at strike :math:`K` and its fixed-leg
annuity :math:`A`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from spxa.market.black import bachelier_price, black_price
from spxa.market.curves import YieldCurve
from spxa.rates.cheyette import Cheyette
from spxa.rates.model_cashflows import swap_cashflows

__all__ = ["DisplacedSwapRate", "cheyette_swaption", "swap_rate_dynamics"]


@dataclass(frozen=True)
class DisplacedSwapRate:
    """Averaged displaced-diffusion parameters of a swap rate up to ``expiry``."""

    S0: float
    annuity: float
    expiry: float
    lam: float
    skew: float

    def price(self, K: float, payer: bool = True) -> float:
        kind = "call" if payer else "put"
        if abs(self.skew) < 1e-8:
            return float(
                self.annuity
                * bachelier_price(self.S0, K, self.expiry, self.lam * self.S0, 1.0, kind)
            )
        d = self.S0 * (1 - self.skew) / self.skew
        return float(
            self.annuity
            * black_price(self.S0 + d, K + d, self.expiry, self.lam * self.skew, 1.0, kind)
        )


def swap_rate_dynamics(
    model: Cheyette, swaption: Any, projection: Optional[YieldCurve] = None, n_time: int = 128
) -> DisplacedSwapRate:
    crv = model.curve
    T = float(crv.time(swaption.expiry))
    if T <= 0:
        raise ValueError("swaption has expired")
    cf = swap_cashflows(swaption.swap, crv, projection)
    K = float(swaption.strike)
    sign = 1.0 if swaption.is_payer else -1.0
    ft = np.array([float(crv.time(c.payment_date)) for c in swaption.swap.fixed])
    fw = np.array([c.notional * c.accrual for c in swaption.swap.fixed])

    def S(t: float, x: float) -> float:
        y = model.ybar(t)
        A = float(np.dot(fw, model.discount_bond(t, ft, x, y)))
        V = float(np.dot(cf.amounts, model.discount_bond(t, cf.times, x, y)))
        return K + sign * V / A

    A0 = float(np.dot(fw, crv.discount(ft)))
    S0 = S(0.0, 0.0)
    ts = np.linspace(0.0, T, n_time + 1)
    lam, b = np.empty_like(ts), np.empty_like(ts)
    h = 1e-4
    for i, t in enumerate(ts):
        Sx = (S(t, h) - S(t, -h)) / (2 * h)
        Sxx = (S(t, h) - 2 * S(t, 0.0) + S(t, -h)) / (h * h)
        s0 = float(model.local_vol(t, 0.0))
        sx = model.beta if s0 > 0 else 0.0
        phi, phix = Sx * s0, Sxx * s0 + Sx * sx
        lam[i] = phi / S0
        b[i] = S0 * phix / (phi * Sx) if phi != 0 else 0.0
    l2 = lam**2
    v = np.concatenate([[0.0], np.cumsum(0.5 * (l2[1:] + l2[:-1]) * np.diff(ts))])
    w = l2 * v
    wsum = float(np.trapezoid(w, ts))
    bbar = float(np.trapezoid(w * b, ts) / wsum) if wsum > 0 else float(b[0])
    return DisplacedSwapRate(S0, A0, T, float(np.sqrt(v[-1] / T)), bbar)


def cheyette_swaption(
    model: Cheyette, swaption: Any, projection: Optional[YieldCurve] = None, n_time: int = 128
) -> float:
    """Approximate price of a physically settled swaption."""
    dyn = swap_rate_dynamics(model, swaption, projection, n_time)
    return dyn.price(float(swaption.strike), swaption.is_payer)
