"""Tests for spxa.pde.grid.TimeGrid."""

import numpy as np
import pytest

from spxa.pde.grid import TimeGrid


class TestTimeGrid:
    def test_uniform_without_events(self):
        tg = TimeGrid.uniform(1.0, 4)
        np.testing.assert_allclose(tg.times, [0, 0.25, 0.5, 0.75, 1.0])
        assert tg.n_steps == 4 and tg.maturity == 1.0
        assert tg.event_indices == ()

    def test_events_hit_exactly(self):
        tg = TimeGrid.uniform(1.0, 10, event_taus=[0.333, 0.71])
        for e in (0.333, 0.71):
            assert np.any(tg.times == e)
        np.testing.assert_allclose(tg.event_taus, [0.333, 0.71])
        assert 9 <= tg.n_steps <= 12

    def test_event_at_maturity(self):
        tg = TimeGrid.uniform(1.0, 5, event_taus=[1.0])
        assert tg.event_indices == (5,)

    def test_tiny_segment_gets_one_step(self):
        tg = TimeGrid.uniform(1.0, 10, event_taus=[0.001])
        assert tg.times[1] == pytest.approx(0.001)

    def test_calendar_events(self):
        tg = TimeGrid.from_calendar_events(2.0, 20, event_times=[0.5, 1.5])
        np.testing.assert_allclose(sorted(tg.event_taus), [0.5, 1.5])
        tg0 = TimeGrid.from_calendar_events(1.0, 4, event_times=[0.0])
        assert tg0.event_indices == (4,)

    def test_validation(self):
        with pytest.raises(ValueError):
            TimeGrid([0.1, 0.2])
        with pytest.raises(ValueError):
            TimeGrid([0.0, 0.5, 0.5])
        with pytest.raises(ValueError):
            TimeGrid([0.0, 0.5, 1.0], event_taus=[0.7])
        with pytest.raises(ValueError):
            TimeGrid([0.0, 0.5, 1.0], event_taus=[0.0])
        with pytest.raises(ValueError):
            TimeGrid.uniform(1.0, 0)
        with pytest.raises(ValueError):
            TimeGrid.uniform(1.0, 5, event_taus=[1.5])
        with pytest.raises(ValueError):
            TimeGrid.from_calendar_events(1.0, 5, event_times=[1.0])

    def test_refine_keeps_events(self):
        tg = TimeGrid.uniform(1.0, 4, event_taus=[0.5]).refine()
        assert tg.n_steps == 8
        np.testing.assert_allclose(tg.event_taus, [0.5])
        np.testing.assert_allclose(tg.steps, 0.125)

    def test_repr(self):
        assert "n_steps=3" in repr(TimeGrid.uniform(1.0, 3))
