"""Axis preparation: cell edges and labels for a patch's coordinates."""

from __future__ import annotations

import string
from typing import Any, NamedTuple

import numpy as np
from dascore.exceptions import ParameterError
from dascore.units import get_quantity, get_quantity_str


class CellEdges(NamedTuple):
    """
    The outer edges of a coordinate's cells, low to high.

    ``low`` and ``high`` keep the coordinate's type: datetime64 for absolute
    times, timedelta64 for relative ones, float otherwise. ``descending``
    says the coordinate's values run high to low, so an image drawn between
    the edges must be flipped along that axis.
    """

    low: Any
    high: Any
    descending: bool


def is_time_like(values_or_dtype) -> bool:
    """Return True for datetime64 or timedelta64 values, or such a dtype."""
    dtype = getattr(values_or_dtype, "dtype", values_or_dtype)
    dtype = np.dtype(dtype)
    return np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)


def cell_edges(coord) -> CellEdges:
    """
    Return the edges of the cells a coordinate's samples sit at the centers of.

    Each end is moved out by half a step, so an image drawn between the
    edges puts every sample at the center of its pixel. The step is the
    coordinate's own if it has one, else the mean spacing of its values.

    Parameters
    ----------
    coord
        A DASCore coordinate, or any 1D array of values.

    Raises
    ------
    ParameterError
        If the coordinate is empty.
    """
    values = np.asarray(getattr(coord, "values", coord))
    if values.ndim != 1 or len(values) == 0:
        msg = f"A coordinate needs at least one value to draw, got {values.shape}."
        raise ParameterError(msg)
    first, last = values[0], values[-1]
    descending = bool(last < first)
    low, high = (last, first) if descending else (first, last)
    step = getattr(coord, "step", None)
    if step is None or _is_nan(step):
        step = (high - low) / (len(values) - 1) if len(values) > 1 else None
    if step is None:
        return CellEdges(low, high, descending)
    half = abs(step) / 2
    if not is_time_like(values):
        low, high, half = float(low), float(high), float(half)
    return CellEdges(low - half, high + half, descending)


def _is_nan(value) -> bool:
    """Return True if value is a NaN (or NaT)."""
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _is_per_second(units) -> bool:
    """Return True if units are 1/s (or Hz) however they are written."""
    try:
        return bool(get_quantity(units) == get_quantity("1/s"))
    except Exception:  # pint raises many kinds for an unparseable unit
        return False


def _label(name, units, hertz: bool = False) -> str:
    """
    Label a quantity "name [units]", leaving out whichever is unset.

    With hertz, units of 1/s are written Hz, as suits a frequency axis.
    """
    name = str(name) if name else ""
    if hertz and units is not None and _is_per_second(units):
        units = "Hz"
    units = get_quantity_str(units) if units is not None else ""
    if name and units:
        return f"{name} [{units}]"
    return name or units or ""


def dim_label(patch, dim: str) -> str:
    """
    Return the axis label for a patch dimension.

    Absolute times are labeled by name alone, since their tick labels are
    dates. Relative times are drawn in seconds and labeled so.
    """
    coord = patch.get_coord(dim)
    name = string.capwords(str(dim))
    dtype = np.asarray(coord.values[:1]).dtype
    if np.issubdtype(dtype, np.datetime64):
        return name
    if np.issubdtype(dtype, np.timedelta64):
        return f"{name} [s]"
    return _label(name, coord.units, hertz=True)


def data_label(patch) -> str:
    """Return the label for a patch's data: its type and units, if set."""
    data_type = str(patch.attrs.get("data_type", "") or "").replace("_", " ")
    return _label(data_type.capitalize(), patch.attrs.get("data_units"))
