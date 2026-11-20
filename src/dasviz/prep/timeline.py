"""Time stepping: playhead movement, step sizes and time formatting.

Note: This was developed with the intent of being able to "pseudo-animate" timestepping
through a patch in the pyvista patchviewer by holding down an arrow key, but could
extend to other animation logic.
"""

from __future__ import annotations

import numpy as np

# Target number of key presses to traverse the whole patch when no step is
# given. Key repeat is typically ~30 Hz, so this is a few seconds of holding.
DEFAULT_KEY_STEPS = 100

# "Human" step sizes in seconds, used to snap the automatic key step to
# something that reads naturally (15 min, not 14.4 min).
_HUMAN_STEPS = np.array(
    [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800]
    + [21600, 43200, 86400, 172800, 604800],
    dtype=float,
)


def format_time(value, unit: str = "s") -> str:
    """Format a time coordinate value for display (datetimes get a space, not 'T')."""
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.datetime64):
        return np.datetime_as_string(arr, unit=unit).replace("T", " ")
    return str(value)


def time_to_seconds(times: np.ndarray) -> np.ndarray:
    """
    Convert a time coordinate to float seconds relative to its first value.

    Datetimes and timedeltas are converted properly; plain numbers are assumed
    to already be in seconds.
    """
    times = np.asarray(times)
    if np.issubdtype(times.dtype, np.datetime64) or np.issubdtype(
        times.dtype, np.timedelta64
    ):
        return (times - times[0]) / np.timedelta64(1, "s")
    return times.astype(float) - float(times[0])


def coerce_step(step) -> float:
    """
    Convert a user-supplied step to float seconds.

    Accepts ``np.timedelta64``, ``datetime.timedelta``, or a number of seconds.
    """
    if isinstance(step, np.timedelta64):
        seconds = float(step / np.timedelta64(1, "s"))
    elif hasattr(step, "total_seconds"):  # datetime.timedelta, pandas.Timedelta
        seconds = float(step.total_seconds())
    else:
        seconds = float(step)
    if not seconds > 0:
        msg = f"key_step must be positive, got {step!r}"
        raise ValueError(msg)
    return seconds


def nice_step(duration: float, target_steps: int = DEFAULT_KEY_STEPS, min_step=0.0):
    """
    Return a human-friendly step (seconds) that crosses `duration` in ~`target_steps`.

    Below one second, snaps to 1-2-5 x 10^k. Above, snaps to the nearest
    "clock" value (1, 2, 5, 10, 15, 30 s, 1, 2, 5, 10, 15, 30 min, 1, 2, 3, 6,
    12 h, 1 d, ...). Never returns less than `min_step` (use the sample
    spacing so that a key press always moves at least one sample).
    """
    if not duration > 0:
        return float(min_step)
    raw = duration / target_steps
    if raw < 1:
        decade = 10.0 ** np.floor(np.log10(raw))
        options = decade * np.array([1.0, 2.0, 5.0, 10.0])
    elif raw <= _HUMAN_STEPS[-1]:
        options = _HUMAN_STEPS
    else:  # very long: whole weeks
        options = np.array([604800.0 * np.ceil(raw / 604800.0)])
    snapped = float(options[np.argmin(np.abs(np.log(options / raw)))])
    return max(snapped, float(min_step))


def nearest_index(rel_times: np.ndarray, t: float) -> int:
    """Index of the sample whose time is closest to `t` (`rel_times` sorted)."""
    j = int(np.searchsorted(rel_times, t))
    if j <= 0:
        return 0
    if j >= len(rel_times):
        return len(rel_times) - 1
    return j if rel_times[j] - t < t - rel_times[j - 1] else j - 1


def advance_playhead(
    rel_times: np.ndarray, t: float, direction: int, step: float
) -> tuple[float, int]:
    """
    Move a continuous playhead by `direction * step` seconds, clamped to the data.

    Returns ``(new_t, sample_index)``. The playhead is kept as a float so steps
    smaller than the sample spacing accumulate instead of being rounded away.
    """
    new_t = float(np.clip(t + direction * step, 0.0, rel_times[-1]))
    return new_t, nearest_index(rel_times, new_t)


def format_duration(seconds: float) -> str:
    """Compact human-readable duration, e.g. 900 -> '15 min', 0.005 -> '5 ms'."""
    for scale, unit in ((86400, "d"), (3600, "h"), (60, "min"), (1, "s")):
        if seconds >= scale:
            return f"{seconds / scale:g} {unit}"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:g} ms"
    return f"{seconds * 1e6:g} us"
