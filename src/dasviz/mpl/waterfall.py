"""A waterfall image of a 2D patch, decimated to what a screen can show."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from dascore.exceptions import ParameterError

from dasviz._deps import requires
from dasviz.prep import (
    cell_edges,
    color_limits,
    colorbar_extend,
    data_label,
    decimate_for_display,
    default_colormap,
    dim_label,
)
from dasviz.prep.image import ScaleType

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _plot_number(value) -> float:
    """Return a coordinate value in matplotlib's plotting units."""
    import matplotlib.dates as mdates

    if isinstance(value, np.datetime64):
        return float(mdates.date2num(value))
    if isinstance(value, np.timedelta64):
        return float(value / np.timedelta64(1, "s"))
    return float(value)


@requires("matplotlib")
def waterfall(
    patch,
    ax: Axes | None = None,
    *,
    scale: float | Sequence[float] | None = None,
    scale_type: ScaleType = "relative",
    cmap=None,
    max_samples: int | Sequence[int | None] | None = 2048,
    colorbar: bool = True,
    show: bool = False,
) -> Axes:
    """
    Draw a 2D patch as an image, its first dimension on the y axis.

    Incubating a successor to ``dascore.viz.waterfall``. It takes the same
    ``scale`` and ``scale_type``, but decimates large patches before drawing:
    each axis longer than ``max_samples`` is reduced to bins which keep their
    largest-magnitude sample, so spikes stay visible and drawing stays fast.
    The color limits are always worked out from the full data. The image
    assumes evenly sampled coordinates.

    Parameters
    ----------
    patch
        A patch with two dimensions.
    ax
        The axes to draw on; new ones are made if None.
    scale
        The color limits; see `dasviz.prep.color_limits`. By default, the
        data's range without outliers.
    scale_type
        ``"relative"`` or ``"absolute"``; see `dasviz.prep.color_limits`.
    cmap
        A colormap or its name. By default, DASCore's for the patch's data
        type.
    max_samples
        The most samples to draw along each axis: one limit for both, a
        pair, or None to draw every sample.
    colorbar
        Whether to add a colorbar.
    show
        Whether to show the figure before returning.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Examples
    --------
    >>> import dascore as dc
    >>> import dasviz  # noqa: F401 (optional; dascore loads the namespace)
    >>> patch = dc.get_example_patch()
    >>> ax = patch.fig.waterfall(scale=0.5)
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if len(patch.dims) != 2:
        msg = f"waterfall needs a patch with 2 dimensions, got {patch.dims}."
        raise ParameterError(msg)
    data = np.asarray(patch.data)
    if np.iscomplexobj(data):
        msg = "waterfall cannot draw complex data; draw its abs, real or imag."
        raise ParameterError(msg)
    limits = color_limits(data, scale, scale_type)
    shown = decimate_for_display(data, max_samples)

    y_dim, x_dim = patch.dims
    edges = {dim: cell_edges(patch.get_coord(dim)) for dim in patch.dims}
    # Draw low to high on both axes, flipping any axis stored high to low,
    # so which way an axis runs depends only on what it shows.
    for axis, dim in enumerate(patch.dims):
        if edges[dim].descending:
            shown = np.flip(shown, axis=axis)
    extent = [
        *(_plot_number(v) for v in edges[x_dim][:2]),
        *(_plot_number(v) for v in edges[y_dim][:2]),
    ]

    if ax is None:
        _, ax = plt.subplots()
    if cmap is None:
        cmap = default_colormap(patch)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap).copy()
    image = ax.imshow(
        shown,
        extent=extent,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
    )
    ax.set_xlabel(dim_label(patch, x_dim))
    ax.set_ylabel(dim_label(patch, y_dim))
    for dim, axis in ((x_dim, ax.xaxis), (y_dim, ax.yaxis)):
        if isinstance(edges[dim].low, np.datetime64):
            axis.axis_date()
            locator = axis.get_major_locator()
            axis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    # As in dascore.viz: time runs down the y axis, as on a seismic section.
    if isinstance(edges[y_dim].low, np.datetime64 | np.timedelta64):
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
    if colorbar:
        bar = ax.get_figure().colorbar(
            image,
            ax=ax,
            extend=colorbar_extend(data, limits),
            fraction=0.05,
            pad=0.025,
        )
        bar.set_label(data_label(patch))
    if show:
        plt.show()
    return ax
