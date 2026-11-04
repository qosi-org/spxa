r"""
Computational-graph nodes on a zero curve with exact adjoints.

The curve is given by continuously compounded zero rates :math:`z_i` at
pillars :math:`T_1 < \dots < T_n`, interpolated linearly in :math:`z` and
extrapolated flat, so :math:`z(t) = \sum_i w_i(t) z_i` with hat weights
:math:`w_i` (:func:`zero_curve_weights`) and :math:`P(t) = e^{-z(t)t}`.
Each zero rate is an input node, so
:class:`~spxa.graph.autodiff.tape.GradientTape` returns key-rate
sensitivities in one reverse pass:

* :func:`discounted_cashflows_node`, :math:`V = \sum_j a_j P(t_j)` with
  :math:`\partial V/\partial z_i = -\sum_j a_j t_j P(t_j) w_i(t_j)`;
* :func:`par_swap_rate_node`, :math:`S = (1 - P(T))/A` with the annuity
  :math:`A = \sum_k \tau_k P(t_k)` and

  .. math::
      \frac{\partial S}{\partial z_i} = \frac{T P(T) w_i(T)
      + S \sum_k \tau_k t_k P(t_k) w_i(t_k)}{A}.

The parallel sensitivity is the sum of the key-rate ones, since the weights
sum to one.
"""

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode
from spxa.graph.types import PortDescriptor

__all__ = ["discounted_cashflows_node", "par_swap_rate_node", "zero_curve_weights"]

NodeLike = Union[Node, float]


def _as_node(x: NodeLike, name: str) -> Node:
    return x if isinstance(x, Node) else ConstantNode(float(x), name=name)


def zero_curve_weights(pillars: Sequence[float], times: Any) -> np.ndarray:
    r"""Hat weights ``W[j, i]`` with :math:`z(t_j) = \sum_i W_{ji} z_i` (flat extrapolation)."""
    T = np.asarray(pillars, float)
    if T.ndim != 1 or T.size == 0 or np.any(np.diff(T) <= 0):
        raise ValueError("pillars must be a non-empty increasing sequence")
    t = np.clip(np.atleast_1d(np.asarray(times, float)), T[0], T[-1])
    W = np.zeros((t.size, T.size))
    if T.size == 1:
        W[:, 0] = 1.0
        return W
    j = np.clip(np.searchsorted(T, t, side="right") - 1, 0, T.size - 2)
    u = (t - T[j]) / (T[j + 1] - T[j])
    W[np.arange(t.size), j] = 1.0 - u
    W[np.arange(t.size), j + 1] = u
    return W


def _inputs(zero_rates: Sequence[NodeLike], pillars: Sequence[float], name: str) -> list[Node]:
    if len(zero_rates) != len(pillars):
        raise ValueError("one zero rate per pillar")
    return [_as_node(z, f"{name}_z{T:g}") for z, T in zip(zero_rates, pillars)]


def discounted_cashflows_node(
    zero_rates: Sequence[NodeLike],
    pillars: Sequence[float],
    times: Sequence[float],
    amounts: Sequence[float],
    name: str = "discounted_cashflows",
) -> OperatorNode:
    """Present value of fixed cash flows ``amounts`` at ``times`` (see the module docstring)."""
    t = np.asarray(times, float)
    a = np.asarray(amounts, float)
    if t.shape != a.shape or t.ndim != 1 or np.any(t < 0):
        raise ValueError("times and amounts must be matching 1-d arrays with times >= 0")
    W = zero_curve_weights(pillars, t)

    def discount(z: Sequence[float]) -> np.ndarray:
        return np.exp(-(W @ np.asarray(z, float)) * t)

    def op(*z: float) -> float:
        return float(a @ discount(z))

    def vjp(adj: Any, *z: float) -> list[float]:
        g = -(a * t * discount(z)) @ W
        return [float(adj) * float(gi) for gi in g]

    return OperatorNode(
        op_fn=op,
        inputs=_inputs(zero_rates, pillars, name),
        name=name,
        output_descriptor=PortDescriptor(name="pv", description="discounted cash flows"),
        vjp_fn=vjp,
    )


def par_swap_rate_node(
    zero_rates: Sequence[NodeLike],
    pillars: Sequence[float],
    maturity: float,
    frequency: float = 0.5,
    name: str = "par_swap_rate",
) -> OperatorNode:
    """Single-curve par rate of a swap paying every ``frequency`` years to ``maturity``."""
    if maturity <= 0 or frequency <= 0:
        raise ValueError("maturity and frequency must be positive")
    n = int(np.ceil(maturity / frequency - 1e-9))
    t = np.minimum(frequency * np.arange(1, n + 1), maturity)
    tau = np.diff(np.r_[0.0, t])
    W = zero_curve_weights(pillars, t)

    def parts(z: Sequence[float]) -> tuple[np.ndarray, float, float]:
        p = np.exp(-(W @ np.asarray(z, float)) * t)
        annuity = float(tau @ p)
        return p, annuity, (1.0 - float(p[-1])) / annuity

    def op(*z: float) -> float:
        return parts(z)[2]

    def vjp(adj: Any, *z: float) -> list[float]:
        p, annuity, s = parts(z)
        g = (maturity * p[-1] * W[-1] + s * (tau * t * p) @ W) / annuity
        return [float(adj) * float(gi) for gi in g]

    return OperatorNode(
        op_fn=op,
        inputs=_inputs(zero_rates, pillars, name),
        name=name,
        output_descriptor=PortDescriptor(name="rate", description="par swap rate"),
        vjp_fn=vjp,
    )
