"""Tests for fiber geometry runs and color sources."""

from __future__ import annotations

import numpy as np
import pytest

from dasviz.prep import (
    MISSING_STRINGS,
    category_color_list,
    encode_categories,
    find_static_coords,
    get_valid_runs,
    is_colocated,
    make_color_source,
)
from dasviz.prep.fiber import TAB10, TAB20


class TestGetValidRuns:
    def test_all_valid(self):
        assert get_valid_runs(np.zeros((5, 3))) == [(0, 5)]

    def test_gap_in_middle_splits(self):
        pts = np.zeros((7, 3))
        pts[3] = np.nan
        assert get_valid_runs(pts) == [(0, 3), (4, 7)]

    def test_nan_at_edges(self):
        pts = np.zeros((6, 3))
        pts[0] = np.nan
        pts[-1] = np.nan
        assert get_valid_runs(pts) == [(1, 5)]

    def test_single_nan_component_invalidates_row(self):
        # NaN in just one of x/y/z is enough to drop the sample.
        pts = np.zeros((4, 3))
        pts[2, 1] = np.nan
        assert get_valid_runs(pts) == [(0, 2), (3, 4)]

    def test_all_nan(self):
        assert get_valid_runs(np.full((4, 3), np.nan)) == []

    def test_slices_are_usable(self):
        pts = np.arange(18, dtype=float).reshape(6, 3)
        pts[2] = np.nan
        lengths = [len(pts[s:e]) for s, e in get_valid_runs(pts)]
        assert lengths == [2, 3]

    def test_returns_plain_ints(self):
        ((start, end),) = get_valid_runs(np.zeros((3, 3)))
        assert type(start) is int
        assert type(end) is int


class TestIsColocated:
    def test_identical_points(self):
        assert is_colocated(np.tile([1.0, 2.0, 3.0], (10, 1)))

    def test_within_tolerance(self):
        pts = np.tile([1.0, 2.0, 3.0], (10, 1))
        pts[5] += 1e-9
        assert is_colocated(pts)

    def test_moving_fiber_is_not_colocated(self):
        pts = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
        assert not is_colocated(pts)


class TestEncodeCategories:
    def test_codes_follow_sorted_categories(self):
        codes, cats = encode_categories(np.array(["borehole", "trenched", "loose", "trenched"]))
        assert list(cats) == ["borehole", "loose", "trenched"]
        assert list(codes) == [0, 2, 1, 2]

    @pytest.mark.parametrize("missing", MISSING_STRINGS)
    def test_missing_strings_are_nan_not_categories(self, missing):
        codes, cats = encode_categories(np.array(["a", missing, "b"], dtype=object))
        assert list(cats) == ["a", "b"]
        assert np.isnan(codes[1])
        assert list(codes[[0, 2]]) == [0, 1]

    def test_all_missing(self):
        codes, cats = encode_categories(np.array(["", ""]))
        assert len(cats) == 0
        assert np.isnan(codes).all()

    def test_custom_missing(self):
        values = np.array(["a", "unknown"])
        codes, cats = encode_categories(values, missing=("unknown",))
        assert list(cats) == ["a"]
        assert np.isnan(codes[1])


class TestMakeColorSource:
    def test_numeric(self):
        src = make_color_source(np.array([1, 5, 3]))
        assert src["kind"] == "num"
        assert src["clim"] == (1.0, 5.0)
        assert src["values"].dtype == float

    def test_numeric_clim_ignores_nan(self):
        src = make_color_source(np.array([1.0, np.nan, 4.0]))
        assert src["clim"] == (1.0, 4.0)

    def test_strings_are_categorical(self):
        src = make_color_source(np.array(["a", "b", "a"]))
        assert src["kind"] == "cat"
        assert list(src["uniq"]) == ["a", "b"]
        assert list(src["values"]) == [0, 1, 0]
        assert src["has_missing"] is False

    def test_missing_strings_are_flagged(self):
        src = make_color_source(np.array(["a", "", "b"], dtype=object))
        assert src["has_missing"] is True
        assert list(src["uniq"]) == ["a", "b"]


# class _FakeCoords:
#     """Stand-in for DASCore's coord manager (only the two attributes we use)."""
#
#     def __init__(self, dim_map):
#         self.dim_map = dim_map
#         self.coord_map = dict.fromkeys(dim_map)
#
#
# class _FakePatch:
#     def __init__(self, dim_map):
#         self.coords = _FakeCoords(dim_map)


def test_find_static_coords(random_patch_with_xyz):
    patch = random_patch_with_xyz.update_coords(
        section=("distance", np.array(["stuff"]*len(random_patch_with_xyz.get_array("distance")), dtype=object)),
        temperature_offset=("time", np.array([1]*len(random_patch_with_xyz.get_array("time")), dtype=object)),
    )
    found = find_static_coords(
        patch, exclude=("x", "y", "z", "time", "distance"), fiber_dim="distance"
    )
    assert found == ["section"]


class TestCategoryColorList:
    def test_default_palette_gives_distinct_colors(self):
        colors = category_color_list(["a", "b", "c"])
        assert len({tuple(c) for c in colors}) == 3

    def test_given_colors_override(self):
        colors = category_color_list(["a", "b"], {"b": "red"})
        assert colors[1] == "red"
        assert colors[0] != "red"

    def test_partial_given_falls_back_by_index(self):
        default = category_color_list(["a", "b", "c"])
        mixed = category_color_list(["a", "b", "c"], {"a": "red"})
        assert mixed[0] == "red"
        assert mixed[1:] == default[1:]

    def test_unknown_given_keys_ignored(self):
        assert category_color_list(["a"], {"zzz": "red"}) == category_color_list(["a"])

    def test_more_than_ten_categories_uses_larger_palette(self):
        colors = category_color_list([str(i) for i in range(15)])
        assert len({tuple(c) for c in colors}) == 15


class TestPalettes:
    @pytest.mark.parametrize(("name", "palette"), [("tab10", TAB10), ("tab20", TAB20)])
    def test_match_matplotlib(self, name, palette):
        mpl = pytest.importorskip("matplotlib")
        from matplotlib.colors import to_hex

        assert list(palette) == [to_hex(c) for c in mpl.colormaps[name].colors]
