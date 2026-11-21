"""Fiber geometry and color sources: what is drawn where, and what colors it."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# String values that mean "undefined" rather than being a real category.
MISSING_STRINGS = ("", "nan", "None")

# Categorical palettes, as hex. These are matplotlib's tab10 and tab20, written
# out so that choosing category colors needs no plotting library;
# tests/prep/test_fiber.py checks them against matplotlib.
TAB10 = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

TAB20 = (
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
)


def change_points(*arrays: np.ndarray) -> np.ndarray:
    """
    Mark where any of several per-channel arrays changes value.

    Returns a boolean array as long as the inputs, True at index ``i`` when
    the value at ``i`` differs from the one at ``i - 1`` in any array (never
    at 0). Pass the result as `get_valid_runs`'s `breaks` to start a new piece
    of fiber there. Non-numeric values are compared as strings, and NaN equals
    NaN, so a run of missing values is not a change.

    Raises
    ------
    ValueError
        If no arrays are given, or they differ in length.
    """
    if not arrays:
        msg = "change_points needs at least one array."
        raise ValueError(msg)
    arrays = tuple(np.asarray(a) for a in arrays)
    if len({a.shape for a in arrays}) != 1 or arrays[0].ndim != 1:
        shapes = [a.shape for a in arrays]
        msg = f"change_points needs 1D arrays of one length, got {shapes}."
        raise ValueError(msg)
    out = np.zeros(len(arrays[0]), dtype=bool)
    for arr in arrays:
        if arr.dtype.kind in "fiu":
            arr = arr.astype(float)
            both_nan = np.isnan(arr[1:]) & np.isnan(arr[:-1])
            differs = (arr[1:] != arr[:-1]) & ~both_nan
        else:
            strings = arr.astype(str)
            differs = strings[1:] != strings[:-1]
        out[1:] |= differs
    return out


def get_valid_runs(
    points: np.ndarray, breaks: np.ndarray | None = None
) -> list[tuple[int, int]]:
    """
    Find contiguous runs of rows in `points` that contain no NaNs.

    Returns a list of half-open (start, end) index pairs, so
    ``points[start:end]`` is one drawable piece of fiber. Rows with NaN
    coordinates (fiber that has no spatial reference) split the fiber into
    separate pieces.

    Parameters
    ----------
    points
        An (n, 3) array of positions along the fiber.
    breaks
        Optional boolean array of length n. Where ``breaks[i]`` is True a new
        piece starts at row ``i`` even though row ``i - 1`` is valid, for
        fiber that is not continuous (see `change_points`).
    """
    points = np.asarray(points, dtype=float)
    mask = ~np.isnan(points).any(axis=1)
    # Pad with False on both sides so runs touching either edge are detected.
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    runs = [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]
    if breaks is None:
        return runs
    breaks = np.asarray(breaks, dtype=bool)
    if breaks.shape != (len(points),):
        msg = f"breaks must have one value per row ({len(points)}), got {breaks.shape}."
        raise ValueError(msg)
    out = []
    for start, end in runs:
        cuts = np.flatnonzero(breaks[start + 1 : end]) + start + 1
        edges = [start, *map(int, cuts), end]
        out.extend(zip(edges[:-1], edges[1:], strict=True))
    return out


def is_colocated(points: np.ndarray, atol: float = 1e-6) -> bool:
    """
    Return True if all points are (numerically) at one location.

    This happens on spooled/coiled cable, where many fiber samples share a
    single spatial position.
    """
    return bool(np.allclose(points, points[0], atol=atol))


def encode_categories(
    values: np.ndarray, missing: Sequence[str] = MISSING_STRINGS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode string-like values as float codes suitable for color mapping.

    Returns ``(codes, categories)``. `categories` is the sorted array of
    unique non-missing strings; `codes[i]` is the index of ``values[i]`` in
    `categories`, or NaN if the value is in `missing`. Missing values are NaN
    (not a category) so they don't claim a color or a legend entry.
    """
    strings = np.asarray(values).astype(str)
    valid = ~np.isin(strings, list(missing))
    categories, inverse = np.unique(strings[valid], return_inverse=True)
    codes = np.full(len(strings), np.nan)
    codes[valid] = inverse
    return codes, categories


def make_color_source(values: np.ndarray) -> dict:
    """
    Build a color-source description from a static (per-fiber-point) array.

    Numeric arrays become ``{"kind": "num", "values", "clim"}``; anything else
    is treated as categorical: ``{"kind": "cat", "values", "uniq",
    "has_missing"}``, where ``has_missing`` says whether any value was missing
    (its code is NaN).
    """
    values = np.asarray(values)
    if values.dtype.kind in "fiu":
        floats = values.astype(float)
        return {
            "kind": "num",
            "values": floats,
            "clim": (float(np.nanmin(floats)), float(np.nanmax(floats))),
        }
    codes, categories = encode_categories(values)
    return {
        "kind": "cat",
        "values": codes,
        "uniq": categories,
        "has_missing": bool(np.isnan(codes).any()),
    }


def find_static_coords(patch, exclude: Sequence[str], fiber_dim: str) -> list[str]:
    """Names of coords that depend only on the fiber dimension (not time)."""
    return [
        name
        for name in patch.coords.coord_map
        if name not in exclude and patch.coords.dim_map[name] == (fiber_dim,)
    ]


def category_color_list(categories: Sequence, given: dict | None = None) -> list:
    """
    One color per category: explicit choices from `given`, else a palette.

    `given` maps category string -> a color, which is passed through as is
    for the backend to interpret. Categories not in `given` take the palette
    color at their index (tab10, or tab20 when there are more than 10
    categories), so a partial `given` still works.
    """
    given = given or {}
    palette = TAB10 if len(categories) <= 10 else TAB20
    return [
        given.get(str(cat), palette[k % len(palette)])
        for k, cat in enumerate(categories)
    ]
