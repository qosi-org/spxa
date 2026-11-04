r"""
Computational-graph nodes for implied-volatility surfaces.

* :func:`svi_total_variance_node`: raw SVI
  :math:`w(k) = a + b\bigl(\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\bigr)`
  as a function of its five parameter nodes, with the analytic gradient

  .. math::
      \partial_a w = 1,\quad
      \partial_b w = \rho(k - m) + R,\quad
      \partial_\rho w = b(k - m),\quad
      \partial_m w = -b\Bigl(\rho + \frac{k - m}{R}\Bigr),\quad
      \partial_\sigma w = \frac{b\sigma}{R},

  where :math:`R = \sqrt{(k - m)^2 + \sigma^2}`. Fits by gradient descent
  on a graph use it.
* :func:`surface_price_node`: the value of a European option on a
  :class:`~spxa.vol.surface.VolSurface` as a function of spot and of a
  parallel implied-volatility shift. It uses the conventions of
  :func:`spxa.vol.risk_adapter.surface_price`, with its spot delta and Black
  vega as the vector–Jacobian product.
"""

from __future__ import annotations

import math
from typing import Any, Union

from spxa.graph.node import ConstantNode, Node, OperatorNode
from spxa.graph.types import PortDescriptor
from spxa.risk.instruments import EuropeanOption
from spxa.vol.risk_adapter import surface_price

__all__ = ["surface_price_node", "svi_total_variance_node"]

NodeLike = Union[Node, float]


def _as_node(x: NodeLike, name: str) -> Node:
    return x if isinstance(x, Node) else ConstantNode(float(x), name=name)


def svi_total_variance_node(
    a: NodeLike,
    b: NodeLike,
    rho: NodeLike,
    m: NodeLike,
    sigma: NodeLike,
    *,
    k: float,
    name: str = "svi_total_variance",
) -> OperatorNode:
    """Raw-SVI total variance at log-moneyness ``k`` (see the module docstring)."""
    k = float(k)

    def op(a_: float, b_: float, rho_: float, m_: float, s_: float) -> float:
        x = k - m_
        return a_ + b_ * (rho_ * x + math.sqrt(x * x + s_ * s_))

    def vjp(adj: Any, a_: float, b_: float, rho_: float, m_: float, s_: float) -> list[float]:
        x = k - m_
        R = math.sqrt(x * x + s_ * s_)
        g = [1.0, rho_ * x + R, b_ * x, -b_ * (rho_ + x / R), b_ * s_ / R]
        return [float(adj) * gi for gi in g]

    labels = ("a", "b", "rho", "m", "sigma")
    inputs = [_as_node(v, f"{name}_{n}") for v, n in zip((a, b, rho, m, sigma), labels)]
    return OperatorNode(
        op_fn=op,
        inputs=inputs,
        name=name,
        output_descriptor=PortDescriptor(name="w", description=f"SVI total variance at k={k:g}"),
        vjp_fn=vjp,
    )


def surface_price_node(
    spot: NodeLike,
    vol_shift: NodeLike = 0.0,
    *,
    surface: Any,
    strike: float,
    maturity: float,
    option: str = "call",
    dynamics: str = "sticky_strike",
    name: str = "surface_price",
) -> OperatorNode:
    """European option on ``surface`` as a function of ``(spot, vol_shift)``.

    The price is per unit of notional. ``dynamics`` is ``"sticky_strike"`` or
    ``"sticky_moneyness"``.
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    opt = EuropeanOption(name, "U", strike, maturity, is_call=option == "call")
    base = {"U_SURFACE": surface, "U_VOL": 0.0}

    def quote(s: float, dv: float) -> Any:
        return surface_price(opt, {"U": s, "U_SURFACE": surface, "U_VOL": dv}, base, dynamics)

    def op(s: float, dv: float) -> float:
        return float(quote(s, dv).price)

    def vjp(adj: Any, s: float, dv: float) -> list[float]:
        q = quote(s, dv)
        return [float(adj) * q.delta, float(adj) * q.vega]

    return OperatorNode(
        op_fn=op,
        inputs=[_as_node(spot, f"{name}_spot"), _as_node(vol_shift, f"{name}_dvol")],
        name=name,
        output_descriptor=PortDescriptor(name="price", description=f"{option} on a surface"),
        vjp_fn=vjp,
    )
