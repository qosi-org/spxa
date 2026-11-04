"""Finite-state semi-Markov process with state- or transition-dependent holding times.

In state-dependent mode, each visit draws a holding time from the current
state's distribution independently of the next-state choice. In
transition-dependent mode, the next state is chosen first, then the holding
time is drawn from the distribution associated with that transition.

This generalizes a discrete-time Markov chain in continuous time: holding
times need not be exponential, so the process is generally not a
continuous-time Markov chain and has no Lévy–Khintchine triplet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


@dataclass(frozen=True)
class SemiMarkovEvents:
    """Event-driven path: transition times and states occupied after each event.

    ``transition_times`` contains transitions strictly after time zero and at
    or before the requested horizon. ``states`` starts with the initial state;
    therefore ``len(states) == len(transition_times) + 1``.
    """

    transition_times: np.ndarray
    states: np.ndarray
    horizon: float


def _positive_integer(value: object) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value >= 1
    )


class SemiMarkovProcess(Process):
    """Finite-state semi-Markov process with state-dependent holding times.

    Parameters
    ----------
    transition_matrix:
        Square row-stochastic matrix P. ``P[i, j]`` is the probability that
        the next visited state is j after a visit to i.
    holding_time_distributions:
        One distribution name per state: ``"exponential"``, ``"weibull"``,
        or ``"gamma"`` (case-insensitive).
    holding_time_parameters:
        One parameter dictionary per state. Exponential uses ``rate``;
        Weibull uses ``shape`` and ``scale``; Gamma uses ``shape`` and
        ``scale``. All parameters must be finite and strictly positive.
    initial_state:
        Index of the initial state.

    Notes
    -----
    Supply either state-dependent distributions (the compatibility mode) or
    ``transition_holding_time_distributions`` mapping each possible transition
    ``(i, j)`` to ``(distribution_name, parameter_dict)``. In transition-
    dependent mode, every positive-probability transition must have a spec;
    the next state is sampled first, then the holding time conditional on i→j.
    This class is simulation-only and does not claim analytical moments or a
    Lévy triplet.
    """

    _SUPPORTED_DISTRIBUTIONS = {"exponential", "weibull", "gamma"}

    def __init__(
        self,
        transition_matrix: ArrayLike,
        holding_time_distributions: list[str] | tuple[str, ...] | None = None,
        holding_time_parameters: list[dict[str, float]]
        | tuple[dict[str, float], ...]
        | None = None,
        *,
        transition_holding_time_distributions: dict[tuple[int, int], tuple[str, dict[str, float]]]
        | None = None,
        initial_state: int = 0,
    ) -> None:
        raw_matrix = np.asarray(transition_matrix)
        if np.iscomplexobj(raw_matrix):
            raise ValueError("transition_matrix must contain real-valued probabilities")
        try:
            p = np.asarray(transition_matrix, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("transition_matrix must be a rectangular numeric matrix") from exc
        if p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[0] < 1:
            raise ValueError("transition_matrix must be a non-empty square matrix")
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("transition_matrix entries must be finite and nonnegative")
        if not np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("each transition_matrix row must sum to 1")
        n_states = p.shape[0]
        if (
            isinstance(initial_state, (bool, np.bool_))
            or not isinstance(initial_state, (int, np.integer))
            or not 0 <= initial_state < n_states
        ):
            raise ValueError(f"initial_state must be an integer in [0, {n_states - 1}]")
        # The two configuration modes are mutually exclusive.  In particular,
        # reject partially supplied state-mode arguments even when transition
        # mode is also selected; silently ignoring them is almost certainly a bug.
        if transition_holding_time_distributions is None:
            if holding_time_distributions is None or holding_time_parameters is None:
                raise ValueError(
                    "provide both holding_time_distributions and "
                    "holding_time_parameters, or provide "
                    "transition_holding_time_distributions"
                )
        elif holding_time_distributions is not None or holding_time_parameters is not None:
            raise ValueError(
                "provide transition_holding_time_distributions alone; do not combine "
                "it with state-dependent holding-time arguments"
            )

        required = {
            "exponential": {"rate"},
            "weibull": {"shape", "scale"},
            "gamma": {"shape", "scale"},
        }

        def validate_spec(
            name: str, params: dict[str, float], label: str
        ) -> tuple[str, dict[str, float]]:
            if not isinstance(name, str):
                raise TypeError(f"distribution for {label} must be a string")
            canonical = name.lower()
            if canonical not in self._SUPPORTED_DISTRIBUTIONS:
                raise ValueError(
                    f"unsupported holding-time distribution {name!r}; "
                    f"choose from {sorted(self._SUPPORTED_DISTRIBUTIONS)}"
                )
            if not isinstance(params, dict):
                raise TypeError(f"parameters for {label} must be a dictionary")
            missing = required[canonical] - params.keys()
            extra = params.keys() - required[canonical]
            if missing or extra:
                raise ValueError(
                    f"{label} {canonical} parameters require "
                    f"{sorted(required[canonical])}; "
                    f"missing={sorted(missing)}, unexpected={sorted(extra)}"
                )
            clean: dict[str, float] = {}
            for key, value in params.items():
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise ValueError(f"{key} for {label} must be a positive finite number")
                number = float(value)
                if not np.isfinite(number) or number <= 0:
                    raise ValueError(f"{key} for {label} must be a positive finite number")
                clean[key] = number
            return canonical, clean

        self.transition_holding_time_distributions = None
        if transition_holding_time_distributions is not None:
            if not isinstance(transition_holding_time_distributions, dict):
                raise TypeError("transition_holding_time_distributions must be a dictionary")
            transition_specs: dict[tuple[int, int], tuple[str, dict[str, float]]] = {}
            for edge, spec in transition_holding_time_distributions.items():
                if (
                    not isinstance(edge, tuple)
                    or len(edge) != 2
                    or any(
                        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                        for v in edge
                    )
                ):
                    raise ValueError(
                        "transition keys must be integer (from_state, to_state) tuples"
                    )
                i, j = map(int, edge)
                if not (0 <= i < n_states and 0 <= j < n_states) or p[i, j] <= 0:
                    raise ValueError(
                        f"transition {edge} must be a possible transition in transition_matrix"
                    )
                if not isinstance(spec, (tuple, list)) or len(spec) != 2:
                    raise ValueError(
                        f"transition {edge} spec must be (distribution_name, parameter_dict)"
                    )
                transition_specs[(i, j)] = validate_spec(spec[0], spec[1], f"transition {edge}")
            required_edges = {
                (i, j) for i in range(n_states) for j in range(n_states) if p[i, j] > 0
            }
            missing_edges = required_edges - transition_specs.keys()
            if missing_edges:
                raise ValueError(
                    f"missing holding-time specs for possible transitions: {sorted(missing_edges)}"
                )
            self.transition_holding_time_distributions = transition_specs
            distributions, parameters = [], []
        else:
            assert holding_time_distributions is not None and holding_time_parameters is not None
            if not isinstance(holding_time_distributions, (list, tuple)):
                raise TypeError("holding_time_distributions must be a list or tuple of names")
            if not isinstance(holding_time_parameters, (list, tuple)):
                raise TypeError("holding_time_parameters must be a list or tuple of dictionaries")
            if len(holding_time_distributions) != n_states:
                raise ValueError("one holding-time distribution is required per state")
            if len(holding_time_parameters) != n_states:
                raise ValueError("one holding-time parameter dictionary is required per state")
            validated = [
                validate_spec(name, params, f"state {i}")
                for i, (name, params) in enumerate(
                    zip(holding_time_distributions, holding_time_parameters, strict=True)
                )
            ]
            distributions = [item[0] for item in validated]
            parameters = [item[1] for item in validated]

        self.transition_matrix = p.copy()
        self.transition_matrix.setflags(write=False)
        self.holding_time_distributions = tuple(distributions)
        self.holding_time_parameters = tuple(parameters)
        self.initial_state = int(initial_state)
        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"SemiMarkovProcess(n_states={n_states})",
            math_note=(
                "Finite-state semi-Markov process: state-dependent or transition-dependent "
                "holding times with next states sampled from a row-stochastic matrix"
            ),
            reference=(
                "Pyke (1961), Markov Renewal Processes: Definitions and Preliminary Properties"
            ),
        )
        super().__init__(exactness=ExactnessLevel.SIMULATION_ONLY, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="SIMULATION_ONLY",
            reason=(
                "A general semi-Markov process is not a Lévy process and has "
                "no Lévy–Khintchine triplet. Use simulate_events() or simulate()."
            ),
        )

    def _properties(self) -> ProcessProperties:
        # Find states reachable from the initial state.
        reachable = {self.initial_state}
        pending = [self.initial_state]

        while pending:
            state = pending.pop()
            for next_state in np.flatnonzero(self.transition_matrix[state] > 0):
                next_state = int(next_state)
                if next_state not in reachable:
                    reachable.add(next_state)
                    pending.append(next_state)

        is_constant = len(reachable) == 1

        return ProcessProperties(
            has_stationary_increments=is_constant,
            has_independent_increments=is_constant,
            is_martingale=is_constant,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=is_constant and self.initial_state == 0,
            hurst_index=None,
            notes=[
                "Finite-state semi-Markov process with state- or "
                "transition-dependent holding times.",
                "Holding times are not restricted to the exponential family.",
                "Finite mean and variance follow from the finite state space.",
                "Process properties are derived from states reachable from the initial state.",
                "Analytical moments and a Levy triplet are not provided.",
            ],
        )

    def _sample_holding_time(
        self, state: int, rng: np.random.Generator, next_state: int | None = None
    ) -> float:
        if self.transition_holding_time_distributions is not None:
            if next_state is None:
                raise ValueError("next_state is required for transition-dependent holding times")
            name, params = self.transition_holding_time_distributions[(state, next_state)]
        else:
            name = self.holding_time_distributions[state]
            params = self.holding_time_parameters[state]
        if name == "exponential":
            return float(rng.exponential(scale=1.0 / params["rate"]))
        if name == "weibull":
            return float(params["scale"] * rng.weibull(params["shape"]))
        return float(rng.gamma(shape=params["shape"], scale=params["scale"]))

    def simulate_events(
        self,
        T: float = 1.0,  # noqa: N803
        *,
        rng: np.random.Generator | None = None,
        max_transitions: int = 1_000_000,
    ) -> SemiMarkovEvents:
        """Simulate transition timestamps and states up to horizon ``T``."""
        if (
            isinstance(T, (bool, np.bool_))
            or not isinstance(T, (int, float, np.integer, np.floating))
            or not np.isfinite(T)
            or T < 0
        ):
            raise ValueError("T must be a finite, nonnegative real number")
        if not _positive_integer(max_transitions):
            raise ValueError("max_transitions must be a positive integer")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        generator = np.random.default_rng() if rng is None else rng
        current_state = self.initial_state
        current_time = 0.0
        times: list[float] = []
        states = [current_state]
        for _ in range(int(max_transitions)):
            if self.transition_holding_time_distributions is not None:
                next_state = int(
                    generator.choice(
                        self.transition_matrix.shape[0],
                        p=self.transition_matrix[current_state],
                    )
                )
                dwell = self._sample_holding_time(current_state, generator, next_state)
            else:
                dwell = self._sample_holding_time(current_state, generator)
                next_state = int(
                    generator.choice(
                        self.transition_matrix.shape[0],
                        p=self.transition_matrix[current_state],
                    )
                )
            if not np.isfinite(dwell) or dwell <= 0.0:
                raise RuntimeError("sampled holding time was not finite and strictly positive")
            next_time = current_time + dwell
            # A positive dwell can still be too small to change a large float
            # timestamp. Recording that as an event would violate strict time
            # ordering and could create a misleading path.
            if next_time <= current_time:
                raise RuntimeError(
                    "sampled holding time is too small to advance simulation time "
                    "at floating-point precision"
                )
            if next_time > T:
                break
            current_state = next_state
            current_time = next_time
            times.append(current_time)
            states.append(current_state)
        else:
            # The cap is a guard, not an unconditional failure: if the next
            # event would lie beyond T, the requested path is already complete.
            if self.transition_holding_time_distributions is not None:
                next_state = int(
                    generator.choice(
                        self.transition_matrix.shape[0],
                        p=self.transition_matrix[current_state],
                    )
                )
                dwell = self._sample_holding_time(current_state, generator, next_state)
            else:
                dwell = self._sample_holding_time(current_state, generator)
            if not np.isfinite(dwell) or dwell <= 0.0:
                raise RuntimeError("sampled holding time was not finite and strictly positive")
            if current_time + dwell <= T:
                raise RuntimeError(
                    f"max_transitions={max_transitions} reached before horizon T={T}; "
                    "increase max_transitions"
                )
        time_array = np.asarray(times, dtype=float)
        state_array = np.asarray(states, dtype=int)
        time_array.setflags(write=False)
        state_array.setflags(write=False)
        return SemiMarkovEvents(time_array, state_array, float(T))

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,  # noqa: N803
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate state paths on an equally spaced grid; shape is (n_paths, n_steps + 1)."""
        if not _positive_integer(n_steps):
            raise ValueError("n_steps must be a positive integer")
        if not _positive_integer(n_paths):
            raise ValueError("n_paths must be a positive integer")
        if (
            isinstance(T, (bool, np.bool_))
            or not isinstance(T, (int, float, np.integer, np.floating))
            or not np.isfinite(T)
            or T < 0
        ):
            raise ValueError("T must be a finite, nonnegative real number")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        generator = np.random.default_rng() if rng is None else rng
        grid = np.linspace(0.0, T, int(n_steps) + 1)
        paths = np.empty((int(n_paths), int(n_steps) + 1), dtype=int)
        for i in range(int(n_paths)):
            events = self.simulate_events(T, rng=generator)
            indices = np.searchsorted(events.transition_times, grid, side="right")
            paths[i] = events.states[indices]
        return paths

    def __repr__(self) -> str:
        mode = (
            "transition-dependent"
            if self.transition_holding_time_distributions is not None
            else "state-dependent"
        )
        return f"SemiMarkovProcess(n_states={self.transition_matrix.shape[0]}, mode={mode!r})"
