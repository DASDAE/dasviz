"""Image preparation: color limits, colormaps and display decimation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from dascore.constants import DEFAULT_COLORMAPS
from dascore.exceptions import ParameterError

ScaleType = Literal["relative", "absolute"]


def default_colormap(patch) -> str:
    """
    Return the name of the colormap DASCore uses for a patch's data type.

    The names are matplotlib's, which PyVista also accepts.
    """
    data_type = str(patch.attrs.get("data_type", "") or "").lower()
    return DEFAULT_COLORMAPS.get(data_type, "bwr")


def _tukey_fence(data: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """Return the data's range with outliers beyond Tukey's fences left out."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q1, q3 = np.nanpercentile(data, [25, 75])
        dmin, dmax = np.nanmin(data), np.nanmax(data)
    iqr = q3 - q1
    return np.asarray(
        [max(q1 - multiplier * iqr, dmin), min(q3 + multiplier * iqr, dmax)]
    )


def color_limits(
    data,
    scale: float | Sequence[float] | None = None,
    scale_type: ScaleType = "relative",
) -> np.ndarray:
    """
    Return the low and high color limits for drawing data.

    This follows ``dascore.viz.waterfall``'s ``scale`` and ``scale_type``.

    Parameters
    ----------
    data
        The values to be drawn. NaNs are ignored.
    scale
        With ``scale_type="relative"``:

        - None: the data's range, leaving out outliers beyond Tukey's
          fences (1.5 interquartile ranges past the quartiles).
        - A number in (0, 1]: that fraction of half the data's range either
          side of its mean.
        - A pair in [0, 1]: fractions of the range from its minimum.

        With ``scale_type="absolute"``: a number ``s`` gives ``(-s, s)``, and
        a pair is used as the limits.
    scale_type
        ``"relative"`` or ``"absolute"``.

    Raises
    ------
    ParameterError
        If scale or scale_type is invalid.
    """
    if scale_type not in ("relative", "absolute"):
        msg = f"scale_type must be 'relative' or 'absolute', got {scale_type!r}."
        raise ParameterError(msg)
    data = np.asarray(data)
    if scale is None:
        if scale_type == "absolute":
            msg = "An absolute scale needs a value: a number or a pair of limits."
            raise ParameterError(msg)
        return _tukey_fence(data)
    values = np.atleast_1d(np.asarray(scale, dtype=float))
    if values.ndim != 1 or len(values) not in (1, 2):
        msg = f"scale must be a number or a pair of numbers, got {scale!r}."
        raise ParameterError(msg)
    if scale_type == "absolute":
        if len(values) == 1:
            return np.asarray([-abs(values[0]), abs(values[0])])
        return values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
    if len(values) == 1:
        if values[0] <= 0:
            msg = f"A relative scale must be greater than 0, got {values[0]}."
            raise ParameterError(msg)
        half_range = 0.5 * (dmax - dmin) or 1e-10  # constant data
        mean = float(np.nanmean(data))
        return np.asarray(
            [mean - values[0] * half_range, mean + values[0] * half_range]
        )
    if np.any(values < 0) or values[0] > values[1]:
        msg = (
            "A relative scale pair must be non-negative fractions, low then "
            f"high, got {scale!r}."
        )
        raise ParameterError(msg)
    return dmin + values * (dmax - dmin)


def colorbar_extend(data, limits) -> Literal["neither", "min", "max", "both"]:
    """
    Say which ends of a colorbar should show that data were clipped.

    Returns matplotlib's ``extend`` vocabulary: ``"min"`` when data fall
    below the low limit, ``"max"`` above the high one, ``"both"`` or
    ``"neither"``.
    """
    low, high = limits
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dmin, dmax = np.nanmin(data), np.nanmax(data)
    below = dmin < low and not np.isclose(dmin, low)
    above = dmax > high and not np.isclose(dmax, high)
    return {(True, True): "both", (True, False): "min", (False, True): "max"}.get(
        (bool(below), bool(above)), "neither"
    )


def _max_abs_bins(data: np.ndarray, axis: int, factor: int) -> np.ndarray:
    """Reduce an axis by factor, keeping the value of largest magnitude per bin."""
    size = data.shape[axis]
    bins = -(-size // factor)
    pad = bins * factor - size
    if pad:
        widths = [(0, 0)] * data.ndim
        widths[axis] = (0, pad)
        data = np.pad(data, widths, constant_values=np.nan)
    shape = data.shape[:axis] + (bins, factor) + data.shape[axis + 1 :]
    binned = data.reshape(shape)
    magnitude = np.abs(binned)
    magnitude[np.isnan(magnitude)] = -np.inf  # an all-NaN bin stays NaN
    index = np.argmax(magnitude, axis=axis + 1, keepdims=True)
    return np.take_along_axis(binned, index, axis=axis + 1).squeeze(axis + 1)


def decimate_for_display(data, max_samples: int | Sequence[int | None] | None = 2048):
    """
    Shrink an array to at most max_samples along each axis, for drawing.

    Each axis longer than its limit is cut into equal bins (the last padded),
    and each bin keeps its value of largest magnitude with its sign. Unlike
    striding, this keeps every spike visible, which is what matters on a
    screen with fewer pixels than samples. The result is for display only.

    Parameters
    ----------
    data
        The array to shrink.
    max_samples
        The most samples to keep along each axis: one limit for all, one per
        axis, or None for no limit (on all, or on one axis).
    """
    data = np.asarray(data)
    if max_samples is None:
        return data
    limits = (
        [max_samples] * data.ndim
        if isinstance(max_samples, int | np.integer)
        else list(max_samples)
    )
    if len(limits) != data.ndim:
        msg = f"max_samples needs one limit per axis ({data.ndim}), got {max_samples}."
        raise ParameterError(msg)
    for axis, limit in enumerate(limits):
        if limit is None or data.shape[axis] <= limit:
            continue
        if limit < 1:
            msg = f"max_samples must be at least 1, got {limit}."
            raise ParameterError(msg)
        if not np.issubdtype(data.dtype, np.inexact):  # padding needs NaN
            data = data.astype(np.float64)
        factor = -(-data.shape[axis] // limit)
        data = _max_abs_bins(data, axis, factor)
    return data
