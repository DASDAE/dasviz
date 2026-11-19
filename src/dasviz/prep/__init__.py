"""
Backend-free preparation shared by every backend.

What goes on screen is worked out here: color limits, display decimation,
cell edges, labels. A backend module only draws the result. Nothing in this
package imports a plotting library, so all of it can be tested with plain
arrays and patches.
"""

from dasviz.prep.axes import CellEdges, cell_edges, data_label, dim_label, is_time_like
from dasviz.prep.image import (
    color_limits,
    colorbar_extend,
    decimate_for_display,
    default_colormap,
)

__all__ = [
    "CellEdges",
    "cell_edges",
    "color_limits",
    "colorbar_extend",
    "data_label",
    "decimate_for_display",
    "default_colormap",
    "dim_label",
    "is_time_like",
]
