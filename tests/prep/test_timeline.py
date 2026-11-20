"""Tests for time stepping: playhead movement, step sizes, time formatting."""

from __future__ import annotations

import numpy as np
import pytest

from dasviz.prep import (
    DEFAULT_KEY_STEPS,
    advance_playhead,
    coerce_step,
    format_duration,
    format_time,
    nearest_index,
    nice_step,
    time_to_seconds,
)


class TestTimeToSeconds:
    def test_datetime(self):
        t0 = np.datetime64("2026-05-29T00:00:00")
        times = t0 + np.array([0, 500, 1500]) * np.timedelta64(1, "ms")
        np.testing.assert_allclose(time_to_seconds(times), [0.0, 0.5, 1.5])

    def test_timedelta(self):
        times = np.array([10, 20, 40]).astype("timedelta64[s]")
        np.testing.assert_allclose(time_to_seconds(times), [0.0, 10.0, 30.0])

    def test_numbers_are_seconds_relative_to_first(self):
        np.testing.assert_allclose(time_to_seconds(np.array([5.0, 6.5])), [0.0, 1.5])


class TestCoerceStep:
    def test_timedelta64(self):
        assert coerce_step(np.timedelta64(2, "m")) == 120.0

    def test_python_timedelta(self):
        from datetime import timedelta

        assert coerce_step(timedelta(minutes=1, seconds=30)) == 90.0

    def test_number_is_seconds(self):
        assert coerce_step(0.25) == 0.25

    @pytest.mark.parametrize("bad", [0, -1, np.timedelta64(0, "s")])
    def test_non_positive_raises(self, bad):
        with pytest.raises(ValueError, match="positive"):
            coerce_step(bad)


class TestNiceStep:
    def test_day_of_data_gets_fifteen_minutes(self):
        assert nice_step(86400.0) == 900.0

    def test_one_hour_gets_thirty_seconds(self):
        assert nice_step(3600.0) == 30.0

    def test_subsecond_uses_1_2_5(self):
        assert nice_step(0.5) == pytest.approx(0.005)

    def test_never_below_min_step(self):
        assert nice_step(10.0, min_step=1.0) == 1.0

    def test_zero_duration_returns_min_step(self):
        assert nice_step(0.0, min_step=0.25) == 0.25

    def test_very_long_is_whole_weeks(self):
        step = nice_step(10 * 365 * 86400.0)
        assert step % 604800 == 0

    def test_crosses_patch_in_roughly_target_steps(self):
        # Snapping to round numbers shouldn't stray far from the target count.
        for duration in (0.5, 7.0, 90.0, 3600.0, 86400.0, 5 * 86400.0):
            n_presses = duration / nice_step(duration)
            assert DEFAULT_KEY_STEPS / 3 < n_presses < DEFAULT_KEY_STEPS * 3


class TestNearestIndex:
    rel = np.array([0.0, 1.0, 2.0, 5.0])

    def test_exact(self):
        assert nearest_index(self.rel, 2.0) == 2

    def test_between_picks_closer(self):
        assert nearest_index(self.rel, 1.4) == 1
        assert nearest_index(self.rel, 1.6) == 2
        assert nearest_index(self.rel, 4.0) == 3  # irregular spacing

    def test_clamps_at_ends(self):
        assert nearest_index(self.rel, -3.0) == 0
        assert nearest_index(self.rel, 99.0) == 3


class TestAdvancePlayhead:
    rel = np.arange(10, dtype=float)  # 1 s sampling

    def _press(self, n, step, direction=1, t=0.0):
        indices = []
        for _ in range(n):
            t, j = advance_playhead(self.rel, t, direction, step)
            indices.append(j)
        return t, indices

    def test_multi_sample_step(self):
        _, indices = self._press(3, step=2.0)
        assert indices == [2, 4, 6]

    def test_sub_sample_steps_accumulate(self):
        # 0.4 s per press on 1 s sampling: 10 presses must land on sample 4,
        # not get stuck at 0 because each press alone rounds to the same sample.
        t, indices = self._press(10, step=0.4)
        assert t == pytest.approx(4.0)
        assert indices[-1] == 4
        assert indices == sorted(indices)

    def test_clamps_at_end_and_start(self):
        t, indices = self._press(50, step=1.0)
        assert t == 9.0 and indices[-1] == 9
        t, indices = self._press(50, step=1.0, direction=-1, t=t)
        assert t == 0.0 and indices[-1] == 0

    def test_no_drift_after_hitting_the_end(self):
        # Pressing past the end then back should respond immediately.
        t, _ = self._press(50, step=1.0)
        t, j = advance_playhead(self.rel, t, -1, 1.0)
        assert j == 8

    def test_irregular_sampling_is_time_based(self):
        rel = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
        t, j = advance_playhead(rel, 0.0, 1, 9.0)
        assert j == 3  # nearest to t=9 is the sample at 10 s


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (900, "15 min"),
            (45, "45 s"),
            (5400, "1.5 h"),
            (172800, "2 d"),
            (0.005, "5 ms"),
            (2e-6, "2 us"),
        ],
    )
    def test_units(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestFormatTime:
    def test_datetime_drops_T(self):
        t = np.datetime64("2026-05-29T00:10:05.123")
        assert format_time(t, "s") == "2026-05-29 00:10:05"

    def test_unit_controls_resolution(self):
        t = np.datetime64("2026-05-29T00:10:05.123")
        assert format_time(t, "m") == "2026-05-29 00:10"
        assert format_time(t, "ms") == "2026-05-29 00:10:05.123"

    def test_non_datetime_falls_back_to_str(self):
        assert format_time(12.5) == "12.5"
