"""Tests for cell edges and labels."""

from types import SimpleNamespace

import numpy as np
import pytest
from dascore.exceptions import ParameterError

from dasviz.prep import cell_edges, data_label, dim_label, is_time_like


def _stub_patch(values, units=None):
    """A stand-in with one coordinate, "time"."""
    coord = SimpleNamespace(values=np.asarray(values), units=units)
    return SimpleNamespace(get_coord=lambda dim: coord)


class TestCellEdges:
    """cell_edges puts each sample at the center of its cell."""

    def test_ascending_array(self):
        assert cell_edges(np.array([0.0, 1.0, 2.0])) == (-0.5, 2.5, False)

    def test_descending_array(self):
        assert cell_edges(np.array([2.0, 1.0, 0.0])) == (-0.5, 2.5, True)

    def test_single_value(self):
        assert cell_edges(np.array([4])) == (4, 4, False)

    def test_uses_coordinate_step(self, patch):
        coord = patch.get_coord("time")
        low, high, descending = cell_edges(coord)
        assert not descending
        assert low == coord.min() - coord.step / 2
        assert high == coord.max() + coord.step / 2

    def test_reverse_coordinate(self, patch):
        coord = patch.get_coord("distance")
        flipped = coord.update(data=coord.values[::-1])
        assert cell_edges(flipped) == (-0.5, 299.5, True)

    def test_empty(self):
        with pytest.raises(ParameterError):
            cell_edges(np.array([]))


def test_is_time_like():
    assert is_time_like(np.array(["2020-01-01"], dtype="datetime64[ns]"))
    assert is_time_like(np.dtype("timedelta64[s]"))
    assert not is_time_like(np.arange(3.0))


class TestLabels:
    """Axis and data labels."""

    def test_dim_with_units(self, patch):
        assert dim_label(patch, "distance") == "Distance [m]"

    def test_absolute_time_has_no_units(self, patch):
        assert dim_label(patch, "time") == "Time"

    def test_relative_time_in_seconds(self):
        stub = _stub_patch(np.array([0, 1], dtype="timedelta64[s]"), "s")
        assert dim_label(stub, "time") == "Time [s]"

    @pytest.mark.parametrize("units", ["1/s", "Hz"])
    def test_frequency_in_hertz(self, units):
        assert (
            dim_label(_stub_patch([0.0, 1.0], units), "frequency") == "Frequency [Hz]"
        )

    def test_data_label(self, patch):
        assert data_label(patch) == ""
        rate = patch.update_attrs(data_type="strain_rate", data_units="1/s")
        assert data_label(rate) == "Strain rate [1 / s]"
