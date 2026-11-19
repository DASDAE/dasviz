"""Tests for color limits, colorbar extension and display decimation."""

import numpy as np
import pytest
from dascore.exceptions import ParameterError

from dasviz.prep import (
    color_limits,
    colorbar_extend,
    decimate_for_display,
    default_colormap,
)


class TestColorLimits:
    """color_limits follows dascore.viz.waterfall's scale and scale_type."""

    def test_default_leaves_out_outliers(self):
        data = np.concatenate([np.linspace(-1, 1, 1000), [1000.0]])
        low, high = color_limits(data)
        assert low == pytest.approx(-1)
        assert high < 10

    def test_relative_fraction_about_mean(self):
        data = np.array([0.0, 10.0])
        np.testing.assert_allclose(color_limits(data, 0.5), [2.5, 7.5])

    def test_relative_pair_of_fractions(self):
        data = np.array([0.0, 10.0])
        np.testing.assert_allclose(color_limits(data, [0.1, 0.9]), [1, 9])

    def test_absolute_number_is_symmetric(self):
        np.testing.assert_allclose(color_limits([0, 1], -3, "absolute"), [-3, 3])

    def test_absolute_pair_used_as_is(self):
        np.testing.assert_allclose(color_limits([0, 1], (1, 4), "absolute"), [1, 4])

    def test_constant_data(self):
        low, high = color_limits(np.ones(5), 1.0)
        assert low < high

    def test_ignores_nan(self):
        np.testing.assert_allclose(color_limits([0.0, np.nan, 10.0], [0, 1]), [0, 10])

    @pytest.mark.parametrize(
        ("scale", "scale_type"),
        [
            (None, "absolute"),
            (1.0, "bad"),
            (0.0, "relative"),
            (-1.0, "relative"),
            ((0.9, 0.1), "relative"),
            ((-0.1, 0.5), "relative"),
            ((1, 2, 3), "relative"),
            (np.ones((2, 2)), "absolute"),
        ],
    )
    def test_invalid(self, scale, scale_type):
        with pytest.raises(ParameterError):
            color_limits(np.arange(10.0), scale, scale_type)


@pytest.mark.parametrize(
    ("limits", "expected"),
    [((0, 10), "neither"), ((1, 10), "min"), ((0, 9), "max"), ((1, 9), "both")],
)
def test_colorbar_extend(limits, expected):
    assert colorbar_extend(np.arange(11.0), limits) == expected


def test_default_colormap(patch):
    assert default_colormap(patch) == "bwr"
    strain = patch.update_attrs(data_type="strain")
    assert default_colormap(strain) == "seismic"


class TestDecimateForDisplay:
    """decimate_for_display keeps each bin's largest-magnitude sample."""

    def test_short_axes_untouched(self):
        data = np.arange(12.0).reshape(3, 4)
        assert decimate_for_display(data, 4) is data

    def test_none_is_no_limit(self):
        data = np.zeros((5000, 5000), dtype=np.float32)
        assert decimate_for_display(data, None).shape == data.shape

    def test_keeps_spike_and_its_sign(self):
        data = np.zeros(1000)
        data[517] = -7.0
        out = decimate_for_display(data, 10)
        assert out.shape == (10,)
        assert out.min() == -7.0
        assert np.count_nonzero(out) == 1

    def test_uneven_last_bin(self):
        out = decimate_for_display(np.arange(10.0), 4)
        # Bins of 3: [0 1 2] [3 4 5] [6 7 8] [9 pad pad].
        np.testing.assert_array_equal(out, [2, 5, 8, 9])

    def test_per_axis_limits(self):
        data = np.ones((100, 50))
        assert decimate_for_display(data, (10, None)).shape == (10, 50)

    def test_all_nan_bin_stays_nan(self):
        data = np.array([np.nan, np.nan, 1.0, -2.0])
        out = decimate_for_display(data, 2)
        assert np.isnan(out[0])
        assert out[1] == -2.0

    def test_integer_data(self):
        out = decimate_for_display(np.arange(9), 3)
        np.testing.assert_array_equal(out, [2, 5, 8])

    def test_complex_kept_complex(self):
        out = decimate_for_display(np.array([1, 3j, 0, 1]), 2)
        np.testing.assert_array_equal(out, [3j, 1])

    @pytest.mark.parametrize("limits", [(1, 2, 3), 0])
    def test_invalid(self, limits):
        with pytest.raises(ParameterError):
            decimate_for_display(np.ones((4, 4)), limits)
