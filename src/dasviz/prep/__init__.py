"""
Backend-free preparation shared by every backend.

What goes on screen is worked out here: color limits, display decimation,
cell edges, labels. A backend module only draws the result. Nothing in this
package imports a plotting library, so all of it can be tested with plain
arrays and patches.
"""

from dasviz.prep.axes import CellEdges, cell_edges, data_label, dim_label, is_time_like
from dasviz.prep.fiber import (
    MISSING_STRINGS,
    category_color_list,
    encode_categories,
    find_static_coords,
    get_valid_runs,
    is_colocated,
    make_color_source,
)
from dasviz.prep.image import (
    color_limits,
    colorbar_extend,
    decimate_for_display,
    default_colormap,
)
from dasviz.prep.timeline import (
    DEFAULT_KEY_STEPS,
    advance_playhead,
    coerce_step,
    format_duration,
    format_time,
    nearest_index,
    nice_step,
    time_to_seconds,
)

__all__ = [
    "DEFAULT_KEY_STEPS",
    "MISSING_STRINGS",
    "CellEdges",
    "advance_playhead",
    "category_color_list",
    "cell_edges",
    "coerce_step",
    "color_limits",
    "colorbar_extend",
    "data_label",
    "decimate_for_display",
    "default_colormap",
    "dim_label",
    "encode_categories",
    "find_static_coords",
    "format_duration",
    "format_time",
    "get_valid_runs",
    "is_colocated",
    "is_time_like",
    "make_color_source",
    "nearest_index",
    "nice_step",
    "time_to_seconds",
]
