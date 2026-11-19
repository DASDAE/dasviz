"""Tests for the matplotlib waterfall, patch.fig.waterfall."""

import dascore as dc
import numpy as np
import pytest
from dascore.exceptions import ParameterError

plt = pytest.importorskip("matplotlib.pyplot")
mdates = pytest.importorskip("matplotlib.dates")

from dasviz.mpl import waterfall  # noqa: E402


def _image(ax):
    """Return the array an axes draws."""
    (image,) = ax.get_images()
    return np.asarray(image.get_array())


class TestWaterfall:
    """The waterfall draws a patch, decimated, with labeled axes."""

    def test_through_namespace(self, patch):
        _, ax = plt.subplots()
        assert patch.fig.waterfall(ax=ax) is ax
        np.testing.assert_array_equal(_image(ax), patch.data)

    def test_new_axes_by_default(self, patch):
        ax = waterfall(patch)
        assert ax.get_xlabel() == "Time"
        assert ax.get_ylabel() == "Distance [m]"

    def test_extent_is_cell_edges(self, patch):
        ax = waterfall(patch)
        assert ax.get_ylim() == pytest.approx((-0.5, 299.5))
        time = patch.get_coord("time")
        start = mdates.date2num(time.min() - time.step / 2)
        assert ax.get_xlim()[0] == pytest.approx(start)

    def test_absolute_time_axis_is_dates(self, patch):
        ax = waterfall(patch)
        formatter = ax.xaxis.get_major_formatter()
        assert isinstance(formatter, mdates.ConciseDateFormatter)

    def test_time_runs_down(self, patch):
        ax = waterfall(patch.transpose("time", "distance"))
        assert ax.yaxis_inverted()
        assert not ax.xaxis_inverted()

    def test_decimates(self, patch):
        ax = waterfall(patch, max_samples=100)
        assert _image(ax).shape == (100, 100)
        assert waterfall(patch, max_samples=None).get_images()[0].get_array().shape == (
            patch.shape
        )

    def test_limits_from_full_data(self, patch):
        ax = waterfall(patch, scale=(0, 1), max_samples=10)
        clim = ax.get_images()[0].get_clim()
        assert clim == pytest.approx((patch.data.min(), patch.data.max()))

    def test_descending_coordinate_drawn_ascending(self, patch):
        coord = patch.get_coord("distance")
        flipped = dc.Patch(
            data=patch.data[::-1],
            coords={
                "distance": coord.values[::-1],
                "time": patch.get_coord("time").values,
            },
            dims=patch.dims,
            attrs=patch.attrs,
        )
        ax = waterfall(flipped)
        np.testing.assert_array_equal(_image(ax), patch.data)
        assert not ax.yaxis_inverted()

    def test_colorbar_label_and_extend(self, patch):
        labeled = patch.update_attrs(data_type="strain_rate", data_units="1/s")
        ax = waterfall(labeled, scale=0.1)
        (colorbar_ax,) = [a for a in ax.get_figure().axes if a is not ax]
        assert colorbar_ax.get_ylabel() == "Strain rate [1 / s]"
        assert ax.get_images()[0].colorbar.extend == "both"

    def test_no_colorbar(self, patch):
        ax = waterfall(patch, colorbar=False)
        assert ax.get_figure().axes == [ax]

    def test_complex_refused(self, patch):
        with pytest.raises(ParameterError, match="complex"):
            waterfall(patch.new(data=patch.data.astype(np.complex128)))

    def test_one_dimension_refused(self, patch):
        line = dc.Patch(
            data=np.arange(5.0), coords={"distance": np.arange(5.0)}, dims=("distance",)
        )
        with pytest.raises(ParameterError, match="2 dimensions"):
            waterfall(line)
