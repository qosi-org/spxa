import numpy as np
import pytest

from spxa import ExactnessLevel, SemiMarkovProcess
from spxa.core.exceptions import ExactnessError


def make_process():
    return SemiMarkovProcess(
        transition_matrix=[[0.0, 1.0], [1.0, 0.0]],
        holding_time_distributions=["weibull", "gamma"],
        holding_time_parameters=[
            {"shape": 1.5, "scale": 1.0},
            {"shape": 2.0, "scale": 0.5},
        ],
    )


def test_constructor_and_public_import():
    process = make_process()
    assert process.transition_matrix.shape == (2, 2)
    assert process.exactness is ExactnessLevel.SIMULATION_ONLY
    assert "SemiMarkovProcess(n_states=2" in repr(process)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.2], [0.0, 1.0]],
        [[1.0, -1.0], [0.0, 1.0]],
        [[1.0, float("nan")], [0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ],
)
def test_rejects_invalid_transition_matrix(matrix):
    with pytest.raises(ValueError):
        SemiMarkovProcess(matrix, ["exponential"] * 2, [{"rate": 1.0}] * 2)


def test_rejects_invalid_distribution_or_parameters():
    with pytest.raises(ValueError, match="unsupported"):
        SemiMarkovProcess([[1.0]], ["normal"], [{}])
    with pytest.raises(ValueError, match="positive finite"):
        SemiMarkovProcess([[1.0]], ["exponential"], [{"rate": 0.0}])
    with pytest.raises(ValueError, match="require"):
        SemiMarkovProcess([[1.0]], ["gamma"], [{"shape": 1.0}])


def test_event_simulation_shapes_and_order():
    result = make_process().simulate_events(T=10, rng=np.random.default_rng(10))
    assert result.states[0] == 0
    assert len(result.states) == len(result.transition_times) + 1
    assert np.all(np.diff(result.transition_times) > 0)
    assert np.all(result.transition_times <= 10)
    assert result.horizon == 10
    assert set(result.states).issubset({0, 1})
    assert np.all(result.states[1:] != result.states[:-1])


def test_grid_simulation_shape_and_state_values():
    paths = make_process().simulate(n_steps=100, n_paths=4, T=5, rng=np.random.default_rng(42))
    assert paths.shape == (4, 101)
    assert paths.dtype.kind in "iu"
    assert set(np.unique(paths)).issubset({0, 1})
    assert np.all(paths[:, 0] == 0)


def test_seeded_reproducibility():
    a = make_process().simulate_events(T=5, rng=np.random.default_rng(123))
    b = make_process().simulate_events(T=5, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(a.transition_times, b.transition_times)
    np.testing.assert_array_equal(a.states, b.states)


def test_zero_horizon_and_absorbing_state():
    process = SemiMarkovProcess([[1.0]], ["exponential"], [{"rate": 1.0}])
    events = process.simulate_events(T=0, rng=np.random.default_rng(1))
    assert events.transition_times.size == 0
    np.testing.assert_array_equal(events.states, [0])
    paths = process.simulate(10, T=0, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(paths, np.zeros((1, 11), dtype=int))


def test_properties_for_absorbing_initial_state():
    process = SemiMarkovProcess(
        transition_matrix=np.eye(2),
        holding_time_distributions=["exponential", "gamma"],
        holding_time_parameters=[
            {"rate": 1.0},
            {"shape": 2.0, "scale": 1.0},
        ],
        initial_state=0,
    )

    properties = process.properties

    assert properties.has_stationary_increments
    assert properties.has_independent_increments
    assert properties.is_martingale
    assert properties.is_subordinator

    paths = process.simulate(
        n_steps=10,
        n_paths=3,
        T=5,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(paths, np.zeros((3, 11), dtype=int))


def test_validation_and_triplet():
    process = make_process()
    with pytest.raises(ValueError, match="n_steps"):
        process.simulate(0)
    with pytest.raises(ValueError, match="n_paths"):
        process.simulate(10, n_paths=0)
    with pytest.raises(ValueError, match="T"):
        process.simulate_events(T=-1)
    with pytest.raises(ValueError, match="max_transitions"):
        process.simulate_events(max_transitions=0)
    with pytest.raises(ExactnessError):
        _ = process.triplet


def test_transition_dependent_holding_times():
    process = SemiMarkovProcess(
        transition_matrix=[[0.0, 1.0], [1.0, 0.0]],
        transition_holding_time_distributions={
            (0, 1): ("exponential", {"rate": 100.0}),
            (1, 0): ("gamma", {"shape": 2.0, "scale": 0.01}),
        },
    )
    result = process.simulate_events(T=2.0, rng=np.random.default_rng(17))
    assert len(result.states) == len(result.transition_times) + 1
    assert np.all(np.diff(result.transition_times) > 0)
    assert np.all(result.states[1:] != result.states[:-1])
    assert process.transition_holding_time_distributions[(0, 1)][0] == "exponential"


def test_transition_dependent_requires_each_possible_edge():
    with pytest.raises(ValueError, match="missing holding-time specs"):
        SemiMarkovProcess(
            transition_matrix=[[0.0, 1.0], [1.0, 0.0]],
            transition_holding_time_distributions={(0, 1): ("exponential", {"rate": 1.0})},
        )


def test_transition_dependent_rejects_impossible_edge():
    with pytest.raises(ValueError, match="possible transition"):
        SemiMarkovProcess(
            transition_matrix=[[0.0, 1.0], [1.0, 0.0]],
            transition_holding_time_distributions={
                (0, 1): ("exponential", {"rate": 1.0}),
                (1, 0): ("gamma", {"shape": 1.0, "scale": 1.0}),
                (0, 0): ("gamma", {"shape": 1.0, "scale": 1.0}),
            },
        )


def test_transition_specific_holding_time_laws_are_sampled_for_selected_edge():
    process = SemiMarkovProcess(
        transition_matrix=[[0.0, 1.0], [1.0, 0.0]],
        transition_holding_time_distributions={
            (0, 1): ("exponential", {"rate": 100.0}),
            (1, 0): ("gamma", {"shape": 2.0, "scale": 0.01}),
        },
    )
    rng = np.random.default_rng(2026)
    samples_01 = np.array([process._sample_holding_time(0, rng, 1) for _ in range(20_000)])
    samples_10 = np.array([process._sample_holding_time(1, rng, 0) for _ in range(20_000)])
    # Exponential mean = 1/rate; Gamma mean = shape * scale.
    assert abs(samples_01.mean() - 0.01) < 0.0005
    assert abs(samples_10.mean() - 0.02) < 0.0008


def test_transition_cap_does_not_fail_when_next_event_is_after_horizon():
    process = SemiMarkovProcess(
        transition_matrix=[[0.0, 1.0], [1.0, 0.0]],
        transition_holding_time_distributions={
            (0, 1): ("exponential", {"rate": 0.01}),
            (1, 0): ("exponential", {"rate": 0.01}),
        },
    )
    # With this seeded stream, the first dwell exceeds T; no transition is needed.
    events = process.simulate_events(T=0.1, max_transitions=1, rng=np.random.default_rng(4))
    assert events.transition_times.size == 0
    np.testing.assert_array_equal(events.states, [0])


def test_transition_cap_raises_if_more_events_are_required():
    process = SemiMarkovProcess(
        transition_matrix=[[0.0, 1.0], [1.0, 0.0]],
        transition_holding_time_distributions={
            (0, 1): ("exponential", {"rate": 100.0}),
            (1, 0): ("exponential", {"rate": 100.0}),
        },
    )
    with pytest.raises(RuntimeError, match="max_transitions=1"):
        process.simulate_events(T=10.0, max_transitions=1, rng=np.random.default_rng(2))


@pytest.mark.parametrize("bad_t", [float("nan"), float("inf"), -1.0, True, "1", [1.0]])
def test_rejects_invalid_horizons(bad_t):
    process = make_process()
    with pytest.raises(ValueError, match="T must be"):
        process.simulate_events(T=bad_t)
    with pytest.raises(ValueError, match="T must be"):
        process.simulate(n_steps=10, T=bad_t)


def test_rejects_mixed_or_incomplete_holding_time_configuration():
    with pytest.raises(ValueError, match="provide both"):
        SemiMarkovProcess([[1.0]], holding_time_distributions=["exponential"])

    with pytest.raises(ValueError, match="do not combine"):
        SemiMarkovProcess(
            [[1.0]],
            holding_time_distributions=["exponential"],
            transition_holding_time_distributions={(0, 0): ("exponential", {"rate": 1.0})},
        )

    with pytest.raises(ValueError, match="do not combine"):
        SemiMarkovProcess(
            [[1.0]],
            holding_time_parameters=[{"rate": 1.0}],
            transition_holding_time_distributions={(0, 0): ("exponential", {"rate": 1.0})},
        )


def test_rejects_non_sequence_state_configuration_and_string_parameters():
    with pytest.raises(TypeError, match="must be a list or tuple"):
        SemiMarkovProcess([[1.0]], "exponential", [{"rate": 1.0}])

    with pytest.raises(ValueError, match="positive finite number"):
        SemiMarkovProcess([[1.0]], ["exponential"], [{"rate": "1.0"}])


def test_rejects_complex_transition_probabilities_without_discarding_imaginary_part():
    with pytest.raises(ValueError, match="real-valued"):
        SemiMarkovProcess(np.array([[1.0 + 0.5j]]), ["exponential"], [{"rate": 1.0}])
