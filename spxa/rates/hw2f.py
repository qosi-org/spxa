r"""The Hull–White two-factor model and its G2++ equivalent.

Hull & White (1994) write

.. math::
    dr = \bigl(\theta(t) + u - \bar a\,r\bigr)\,dt + \sigma_1\,dZ_1, \qquad
    du = -\bar b\,u\,dt + \sigma_2\,dZ_2, \qquad d\langle Z_1, Z_2\rangle = \bar\rho\,dt.

Setting :math:`y = u/(\bar a - \bar b)` and :math:`x = r - y - \varphi(t)`
gives G2++ (Brigo & Mercurio, §4.2.5) with :math:`a = \bar a`,
:math:`b = \bar b` and

.. math::
    \sigma_4 = \frac{\sigma_2}{\bar a - \bar b}, \quad
    \eta = |\sigma_4|, \quad
    \sigma = \sqrt{\sigma_1^2 + \sigma_4^2 - 2\bar\rho\sigma_1\sigma_4}, \quad
    \rho = \operatorname{sign}(\sigma_4)\,\frac{\bar\rho\sigma_1 - \sigma_4}{\sigma}.

The sign accounts for :math:`\bar a < \bar b`, where the driving Brownian
motion of :math:`y` is :math:`-Z_2`. The map is a bijection for
:math:`\bar a \ne \bar b`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spxa.market.curves import YieldCurve
from spxa.rates.g2pp import G2PP

__all__ = ["HullWhite2F"]


@dataclass(frozen=True)
class HullWhite2F:
    """Hull–White two-factor parameters ``(a_bar, sigma1, b_bar, sigma2, rho_bar)``."""

    a_bar: float
    sigma1: float
    b_bar: float
    sigma2: float
    rho_bar: float

    def __post_init__(self) -> None:
        if min(self.a_bar, self.b_bar, self.sigma1, self.sigma2) <= 0 or not -1 < self.rho_bar < 1:
            raise ValueError("need positive speeds and vols and |rho_bar| < 1")
        if abs(self.a_bar - self.b_bar) < 1e-10:
            raise ValueError("a_bar and b_bar must differ")

    def to_g2pp(self, curve: YieldCurve) -> G2PP:
        s4 = self.sigma2 / (self.a_bar - self.b_bar)
        sigma = float(np.sqrt(self.sigma1**2 + s4**2 - 2 * self.rho_bar * self.sigma1 * s4))
        rho = float(np.sign(s4) * (self.rho_bar * self.sigma1 - s4) / sigma)
        return G2PP(curve, self.a_bar, sigma, self.b_bar, abs(s4), rho)

    @classmethod
    def from_g2pp(cls, model: G2PP) -> "HullWhite2F":
        sgn = float(np.sign(model.a - model.b))
        s4 = sgn * model.eta
        # sigma1 dZ1 = sigma dW1 + s4 dZ2, with dZ2 = sign(s4) dW2
        sigma1 = float(
            np.sqrt(model.sigma**2 + model.eta**2 + 2 * model.rho * model.sigma * model.eta)
        )
        rho_bar = (model.sigma * model.rho * sgn + s4) / sigma1
        return cls(model.a, sigma1, model.b, model.eta * abs(model.a - model.b), float(rho_bar))
