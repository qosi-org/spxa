r"""Implied-volatility surfaces in strike and expiry.

A :class:`VolSurface` combines a total-variance model :math:`w(k, T)` in
log-forward moneyness :math:`k = \log(K/F_T)` with a forward curve that
supplies :math:`F_T` and the discount factor :math:`P(0, T)`. From these it
gives implied volatilities and Black prices in strike space:

.. math::
    \sigma(K, T) = \sqrt{w(\log(K/F_T), T)/T}, \qquad
    V(K, T) = P(0, T)\,\mathrm{Black}(F_T, K, T, \sigma(K, T)).

The model is anything with a method ``total_variance(k, T)`` for arrays
``k`` and scalar ``T``. That covers :class:`~spxa.vol.SSVI`,
:class:`~spxa.vol.ESSVI` and :class:`~spxa.vol.PriceInterpolatedSurface`.
The surface does not extrapolate the model in time; it uses whatever the
model does.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

import numpy as np

from spxa.market.black import black_price
from spxa.market.curves import EquityForward
from spxa.vol.interpolation import PriceInterpolatedSurface, VarianceInterpolatedSurface

__all__ = ["TotalVarianceModel", "ForwardLike", "VolSurface"]


class TotalVarianceModel(Protocol):
    """Anything that returns total implied variance ``w(k, T)``."""

    def total_variance(self, k: Any, T: float) -> Any: ...


class ForwardLike(Protocol):
    """Anything that returns forwards and discount factors for year fractions."""

    def forward(self, t: Any) -> Any: ...

    def discount(self, t: Any) -> Any: ...


class _Flat:
    def __init__(self, sigma: float) -> None:
        self.sigma = float(sigma)

    def total_variance(self, k: Any, T: float) -> Any:
        kk = np.asarray(k, float)
        out = np.full(kk.shape, self.sigma**2 * T)
        return float(out) if out.ndim == 0 else out


def _out(x: np.ndarray) -> Any:
    return float(x) if x.ndim == 0 else x


class VolSurface:
    r"""Implied-volatility surface on a forward curve.

    Parameters
    ----------
    model
        Total-variance model with ``total_variance(k, T)``.
    forward
        An :class:`~spxa.market.curves.EquityForward` or any object with
        ``forward(t)`` and ``discount(t)`` in year fractions.

    All strike-space methods broadcast ``K`` against ``T``. At
    :math:`T = 0` the total variance is zero and prices are intrinsic.
    """

    def __init__(self, model: TotalVarianceModel, forward: ForwardLike) -> None:
        if not hasattr(model, "total_variance"):
            raise TypeError("model needs a total_variance(k, T) method")
        self.model = model
        self.forward_curve = forward

    # ------------------------------------------------------------ builders
    @classmethod
    def flat(cls, sigma: float, forward: ForwardLike) -> "VolSurface":
        """Constant volatility ``sigma`` (Black–Scholes)."""
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        return cls(_Flat(sigma), forward)

    @classmethod
    def from_slices(
        cls,
        expiries: Sequence[float],
        slices: Sequence[Any],
        forward: ForwardLike,
        time_interpolation: str = "price",
    ) -> "VolSurface":
        """Slices joined in time.

        ``"price"`` uses Gatheral–Jacquier price interpolation, with no
        extrapolation. ``"fixed_k"`` and ``"normalised"`` use
        :class:`~spxa.vol.VarianceInterpolatedSurface`, which extrapolates.
        """
        if time_interpolation == "price":
            return cls(PriceInterpolatedSurface(expiries, slices), forward)
        return cls(VarianceInterpolatedSurface(expiries, slices, time_interpolation), forward)

    @classmethod
    def from_spot(
        cls,
        model: TotalVarianceModel,
        spot: float,
        rate: Any,
        dividend_yield: Any = None,
        **kwargs: Any,
    ) -> "VolSurface":
        """Surface on an :class:`EquityForward` built from spot, rate and yield."""
        return cls(model, EquityForward(spot, rate, dividend_yield, **kwargs))

    # --------------------------------------------------------------- curves
    def forward(self, T: Any) -> Any:
        r"""Forward :math:`F_T`."""
        return self.forward_curve.forward(T)

    def discount(self, T: Any) -> Any:
        r"""Discount factor :math:`P(0, T)`."""
        return self.forward_curve.discount(T)

    def log_moneyness(self, K: Any, T: Any) -> Any:
        r""":math:`k = \log(K/F_T)`."""
        KK, TT = np.broadcast_arrays(np.asarray(K, float), np.asarray(T, float))
        return _out(np.log(KK / self._forward_array(TT)))

    def strike(self, k: Any, T: Any) -> Any:
        r"""Strike :math:`K = F_T e^k`."""
        kk, TT = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
        return _out(self._forward_array(TT) * np.exp(kk))

    def _forward_array(self, T: np.ndarray) -> np.ndarray:
        if np.any(T < 0):
            raise ValueError("negative expiry")
        out = np.empty(T.shape)
        for t in np.unique(T):
            out[T == t] = float(self.forward_curve.forward(float(t)))
        return out

    def _discount_array(self, T: np.ndarray) -> np.ndarray:
        out = np.empty(T.shape)
        for t in np.unique(T):
            out[T == t] = float(self.forward_curve.discount(float(t)))
        return out

    # ------------------------------------------------------- moneyness space
    def total_variance(self, k: Any, T: Any) -> Any:
        """Total implied variance at log-moneyness ``k`` and expiry ``T``."""
        kk, TT = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
        if np.any(TT < 0):
            raise ValueError("negative expiry")
        out = np.zeros(kk.shape)
        for t in np.unique(TT):
            if t == 0:
                continue
            m = TT == t
            out[m] = np.asarray(self.model.total_variance(kk[m], float(t)), float)
        return _out(out)

    def implied_vol_k(self, k: Any, T: Any) -> Any:
        r"""Implied volatility at log-moneyness ``k`` (``T > 0``)."""
        TT = np.asarray(T, float)
        if np.any(TT <= 0):
            raise ValueError("implied volatility needs T > 0")
        return _out(np.sqrt(np.asarray(self.total_variance(k, T), float) / TT))

    # ---------------------------------------------------------- strike space
    def implied_vol(self, K: Any, T: Any) -> Any:
        r"""Black implied volatility :math:`\sigma(K, T)` (``T > 0``)."""
        return self.implied_vol_k(self.log_moneyness(K, T), T)

    def price(self, K: Any, T: Any, option: Any = "call") -> Any:
        """Discounted Black price of a European option."""
        KK, TT = np.broadcast_arrays(np.asarray(K, float), np.asarray(T, float))
        F = self._forward_array(TT)
        w = np.asarray(self.total_variance(np.log(KK / F), TT), float)
        safe = np.where(TT > 0, TT, 1.0)
        sigma = np.where(TT > 0, np.sqrt(w / safe), 0.0)
        return black_price(F, KK, TT, sigma, self._discount_array(TT), option)

    def smile(self, T: float, strikes: Any) -> Any:
        """Implied volatilities across ``strikes`` at one expiry."""
        return self.implied_vol(np.asarray(strikes, float), float(T))

    # ------------------------------------------------------------------ ATM
    def atm_vol(self, T: Any) -> Any:
        r"""At-the-money-forward volatility :math:`\sigma(F_T, T)`."""
        return self.implied_vol_k(np.zeros(np.shape(T)), T)

    def atm_skew(self, T: float, h: Optional[float] = None) -> float:
        r"""ATM skew :math:`\partial_k \sigma(k, T)|_{k=0}` (central difference)."""
        if T <= 0:
            raise ValueError("T must be positive")
        h = 1e-4 * max(float(np.sqrt(self.total_variance(0.0, T))), 1e-3) if h is None else h
        up, dn = self.implied_vol_k(np.array([h, -h]), T)
        return float((up - dn) / (2 * h))

    # ----------------------------------------------------------------- bumps
    def bumped(self, bump: Any) -> "VolSurface":
        """Surface with an implied-vol bump: a number (parallel) or a function ``(k, T) -> dvol``.

        See :mod:`spxa.vol.bumps` for the shapes.
        """
        from spxa.vol.bumps import BumpedModel, parallel_bump

        f = parallel_bump(float(bump)) if isinstance(bump, (int, float)) else bump
        return VolSurface(BumpedModel(self.model, f), self.forward_curve)

    # ------------------------------------------------------ forward variance
    def forward_variance(self, T1: float, T2: Any, k: Any = 0.0) -> Any:
        r"""Forward implied variance :math:`(w(k, T_2) - w(k, T_1))/(T_2 - T_1)`.

        This is taken at fixed log-forward moneyness ``k`` (ATM forward by
        default). By Gatheral's calendar condition it is non-negative for
        every ``k`` exactly when the surface has no calendar arbitrage.
        ``T1 = 0`` gives the spot implied variance :math:`\sigma^2(k, T_2)`.
        """
        T2a = np.asarray(T2, float)
        if T1 < 0 or np.any(T2a <= T1):
            raise ValueError("need 0 <= T1 < T2")
        ka = np.asarray(k, float)
        w2 = np.asarray(self.total_variance(ka, T2a), float)
        w1 = np.asarray(self.total_variance(ka, T1), float) if T1 > 0 else 0.0
        return _out((w2 - w1) / (T2a - T1))

    def forward_vol(self, T1: float, T2: Any, k: Any = 0.0) -> Any:
        """Square root of :meth:`forward_variance`; raises on calendar arbitrage."""
        v = np.asarray(self.forward_variance(T1, T2, k), float)
        if np.any(v < 0):
            raise ValueError("negative forward variance: calendar arbitrage")
        return _out(np.sqrt(v))

    def forward_variance_grid(self, expiries: Any, k: Any) -> np.ndarray:
        """Forward variances between consecutive ``expiries`` (rows) at each ``k`` (columns).

        A strictly negative entry flags calendar arbitrage between two
        expiries at that moneyness.
        """
        Ts = np.asarray(expiries, float)
        if Ts.ndim != 1 or Ts.size < 2 or np.any(np.diff(Ts) <= 0) or Ts[0] <= 0:
            raise ValueError("expiries must be positive and strictly increasing")
        ka = np.atleast_1d(np.asarray(k, float))
        w = np.array([np.asarray(self.total_variance(ka, T), float) for T in Ts])
        return np.diff(w, axis=0) / np.diff(Ts)[:, None]

    # ---------------------------------------------------------------- greeks
    def greeks(
        self,
        K: Any,
        T: float,
        option: Any = "call",
        dynamics: str = "sticky_strike",
        bump: float = 1e-4,
    ) -> dict[str, Any]:
        r"""Forward delta and gamma under a smile dynamic, plus vega.

        When the forward moves to :math:`F(1 + h)`:

        ``"sticky_strike"``
            :math:`\sigma(K, T)` stays put, so delta and gamma are Black's
            at the surface volatility.
        ``"sticky_moneyness"``
            :math:`\sigma` is a function of :math:`k = \log(K/F)` and moves
            with the forward, which gives
            :math:`\Delta = \Delta_{BS} - \mathcal V_{BS}\,\sigma_k/F`.

        Greeks are taken by central bumps of the forward (relative size
        ``bump``) at a fixed discount factor. ``delta`` and ``gamma`` are
        with respect to :math:`F`; ``vega`` is for a parallel shift of
        implied volatility. Deltas are per unit of forward; multiply by
        :math:`F/S` for spot deltas under proportional carry.
        """
        if dynamics not in ("sticky_strike", "sticky_moneyness"):
            raise ValueError("dynamics must be 'sticky_strike' or 'sticky_moneyness'")
        if T <= 0:
            raise ValueError("T must be positive")
        KK = np.asarray(K, float)
        F = float(self.forward(T))
        D = float(self.discount(T))
        sig0 = np.asarray(self.implied_vol(KK, T), float)

        def value(Fb: float, dsig: float = 0.0) -> np.ndarray:
            if dynamics == "sticky_strike":
                sig = sig0
            else:
                sig = np.asarray(self.implied_vol_k(np.log(KK / Fb), T), float)
            return np.asarray(black_price(Fb, KK, T, sig + dsig, D, option), float)

        h = bump * F
        up, mid, dn = value(F + h), value(F), value(F - h)
        dv = 1e-4
        vega = (value(F, dv) - value(F, -dv)) / (2 * dv)
        return {
            "price": _out(mid),
            "delta": _out((up - dn) / (2 * h)),
            "gamma": _out((up - 2 * mid + dn) / (h * h)),
            "vega": _out(vega),
        }

    # ----------------------------------------------------------------- delta
    def _carry(self, T: float) -> float:
        spot = getattr(self.forward_curve, "spot", None)
        if spot is None:
            raise ValueError("spot deltas need a forward curve with a spot")
        return float(self.forward(T)) * float(self.discount(T)) / float(spot)

    def delta(
        self,
        K: Any,
        T: float,
        option: str = "call",
        kind: str = "forward",
        premium_adjusted: bool = False,
    ) -> Any:
        """Black delta at the surface volatility (see :mod:`spxa.vol.delta`)."""
        from spxa.vol.delta import black_delta

        carry = self._carry(T) if kind == "spot" else 1.0
        F = float(self.forward(T))
        return black_delta(F, K, T, self.implied_vol(K, T), option, kind, premium_adjusted, carry)

    def strike_from_delta(
        self,
        delta: float,
        T: float,
        option: str = "call",
        kind: str = "forward",
        premium_adjusted: bool = False,
    ) -> float:
        """Strike whose delta, at its own smile volatility, equals ``delta``."""
        from spxa.vol.delta import strike_from_delta

        carry = self._carry(T) if kind == "spot" else 1.0
        return strike_from_delta(
            delta,
            float(self.forward(T)),
            T,
            lambda k: float(self.implied_vol_k(k, T)),
            option,
            kind,
            premium_adjusted,
            carry,
        )

    def __repr__(self) -> str:
        return f"VolSurface(model={type(self.model).__name__})"
