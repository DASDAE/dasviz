"""
3D viewer of a fiber-optic patch drawn along the fiber's physical path.

Moved here from a standalone prototype (``plot_patch3d``). Meshes are built
once; time stepping and color-source switching only swap scalar arrays. The
backend-free pieces (geometry runs, color sources, time stepping) live in
`dasviz.prep`; this module only draws.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from dascore.exceptions import ParameterError

from dasviz._deps import requires
from dasviz.prep import (
    DEFAULT_KEY_STEPS,
    advance_playhead,
    category_color_list,
    change_points,
    coerce_step,
    color_limits,
    colorbar_extend,
    data_label,
    default_colormap,
    dim_label,
    find_static_coords,
    format_duration,
    format_time,
    get_valid_runs,
    is_colocated,
    make_color_source,
    nice_step,
    time_to_seconds,
)

# How missing values in a categorical coordinate appear in the legend.
MISSING_LABEL = "(missing)"
MISSING_COLOR = "lightgray"


class PatchViewer(NamedTuple):
    """Handles returned by `patchviewer`."""

    plotter: object  # pyvista.Plotter
    set_step: Callable[[float], int]
    set_color_by: Callable[[str], None]
    color_sources: dict  # name -> color-source dict; first key is the patch data
    meshes: list  # one mesh per drawn piece of fiber (tube or coil sphere)
    nudge: Callable[[int], None]  # nudge(+1 / -1): advance by `key_step`
    key_step: float  # seconds of data time advanced per nudge
    step_sample: Callable[[int], None]  # step_sample(+1 / -1): move one sample

    def show(self, *args, **kwargs):
        """Show the viewer's window; arguments go to ``pyvista.Plotter.show``."""
        return self.plotter.show(*args, **kwargs)


@requires("pyvista")
def patchviewer(
    patch,
    x="x",
    y="y",
    z="z",
    cmap=None,
    scale=None,
    scale_type="relative",
    clim=None,
    radius=0.1,
    sphere_scale_factor=2,
    time_dim="time",
    time_unit="s",
    plotter=None,
    color_by=None,
    category_colors=None,
    split_by=None,
    key_step=None,
    controls=True,
    show=False,
    **kwargs,
) -> PatchViewer:
    """
    Draw a 2D (time, fiber) patch on the fiber's 3D path, with time stepping.

    Meshes are built once; stepping through time or switching color source only
    swaps scalar arrays, which is far cheaper than rebuilding tubes each frame.

    Parameters
    ----------
    patch
        A 2D patch with a time dimension and one fiber dimension. The coords
        `x`, `y`, `z` must be defined along the fiber dimension. NaN
        coordinates split the fiber into separate pieces.
    x, y, z
        Names of the spatial coordinates.
    cmap
        Colormap for continuous values (the patch data and numeric coords): a
        name or colormap object PyVista accepts. By default, the one DASCore uses
        for the patch's data type (`dasviz.prep.default_colormap`).
    scale, scale_type
        Color limits for the patch data, as in `patch.fig.waterfall` (see
        `dasviz.prep.color_limits`). By default, the data's range without
        outliers (Tukey's fences). The limits are worked out from *all* time
        steps at once, so colors don't rescale from frame to frame.
    clim
        Explicit color limits ``(low, high)`` for the patch data, instead of
        `scale`. Give one or the other.
    radius
        Tube radius, in coordinate units.
    sphere_scale_factor
        Coiled sections (all samples at one location) are drawn as a sphere of
        radius ``radius * sphere_scale_factor``.
    time_dim
        Name of the time dimension.
    time_unit
        Resolution for displayed timestamps (a numpy datetime unit).
    plotter
        An existing ``pyvista.Plotter`` to draw into (e.g. one that already has
        a building drawn). A new one is created if None.
    color_by
        Names of static (fiber-dimension-only) coords to offer as alternative
        color sources. None auto-detects every such coord except x/y/z. String
        coords are shown categorically, numeric ones continuously.
    category_colors
        ``{coord_name: {category: color}}``. Categories not listed fall back to
        a default palette.
    split_by
        Name (or names) of static coords marking where the cable is not
        continuous, such as a borehole or leg. The fiber is cut wherever any of
        them changes between adjacent channels, rather than joining the end of
        one piece to the start of the next. Needed after selecting channels
        (e.g. only the downgoing legs), since channels that were far apart then
        sit next to each other. Channels with NaN positions always split it.
    key_step
        How much data time one left/right arrow press advances (holding the key
        repeats, giving a pseudo-animation). A ``np.timedelta64``,
        ``datetime.timedelta``, or a number of seconds. Default: a round value
        (1-2-5 below a second, "clock" values like 15 s / 10 min / 1 h above)
        that crosses the whole patch in roughly 100 presses, never less than
        one sample spacing. The slider is unaffected and still visits every
        sample, and Page Up/Page Down always move exactly one sample. Steps
        smaller than the sample spacing accumulate, so they advance a sample
        every few presses.
    controls
        If True, add the time slider, left/right arrow keys, and (if there is
        more than one color source) radio buttons. Set False for scripted use or
        when the plotter has no interactor (e.g. off-screen).
    show
        If True, show the scene immediately on creation.

    Other Parameters
    ----------------
    See pyvista.Plotter.show.

    Returns
    -------
    A `PatchViewer` namedtuple of (plotter, set_step, set_color_by,
    color_sources, meshes, nudge, key_step, step_sample). ``nudge(+1/-1)`` does
    what the arrow keys do and ``step_sample(+1/-1)`` what Page Up/Down do; both
    work even with ``controls=False``. ``key_step`` is the step actually used,
    in seconds.
    """
    import pyvista as pv

    if time_dim not in patch.dims or len(patch.dims) != 2:
        msg = f"patch must be 2D with a '{time_dim}' dimension; got dims {patch.dims}"
        raise ValueError(msg)

    fiber_dim = next(d for d in patch.dims if d != time_dim)
    patch = patch.transpose(time_dim, fiber_dim)
    data = patch.data  # (n_time, n_fiber)
    n_time = data.shape[0]
    times = patch.coords.get_array(time_dim)
    # The label is also the key of the patch data among the color sources, so it
    # must not be empty for a patch with no data type or units.
    label = data_label(patch) or "Data"
    points = np.column_stack([patch.coords.get_array(c) for c in (x, y, z)])
    breaks = None
    if split_by is not None:
        names = [split_by] if isinstance(split_by, str) else list(split_by)
        for name in names:
            if patch.coords.dim_map.get(name) != (fiber_dim,):
                msg = (
                    f"split_by coord '{name}' must depend only on '{fiber_dim}'; "
                    f"options are {find_static_coords(patch, (), fiber_dim)}"
                )
                raise ParameterError(msg)
        breaks = change_points(*(patch.coords.get_array(n) for n in names))

    # Arrow keys step through *time*, not samples, so playback speed is comparable
    # between patches with different sample rates or lengths.
    rel_times = time_to_seconds(times)  # float seconds since the first sample
    duration = float(rel_times[-1])
    if key_step is None:
        spacing = float(np.median(np.diff(rel_times))) if n_time > 1 else 0.0
        step_s = nice_step(duration, DEFAULT_KEY_STEPS, min_step=spacing)
    else:
        step_s = coerce_step(key_step)
    if clim is not None and scale is not None:
        msg = "Give clim or scale, not both."
        raise ParameterError(msg)
    if clim is None:
        clim = tuple(float(v) for v in color_limits(data, scale, scale_type))
    if len(clim) != 2:
        msg = f"Color limits need a low and a high value, got {clim!r}."
        raise ParameterError(msg)
    if cmap is None:
        cmap = default_colormap(patch)
    # Which ends of the limits leave data out. PyVista's scalar bar has no extend
    # triangles, so the data bar gets above/below swatches instead.
    extend = colorbar_extend(data, clim)
    clipped_below = extend in ("min", "both")
    clipped_above = extend in ("max", "both")
    if plotter is None:
        plotter = pv.Plotter()
    pl = plotter

    # ---- build one mesh per valid run of fiber -------------------------------
    pieces = []  # (mesh, start, end, idx); idx maps mesh points -> fiber samples
    meshes = []
    actors = []
    for start, end in get_valid_runs(points, breaks):
        seg_pts = points[start:end]
        if is_colocated(seg_pts):
            mesh = pv.Sphere(radius=radius * sphere_scale_factor, center=seg_pts[0])
            idx = None
        else:
            line = pv.MultipleLines(seg_pts)
            # tube() makes several points per input point; tag each input point
            # with its index so we can map fiber samples onto tube points later.
            line["_idx"] = np.arange(len(seg_pts))
            mesh = line.tube(radius=radius)
            idx = np.rint(mesh["_idx"]).astype(int)
        mesh[label] = np.zeros(mesh.n_points)
        actors.append(
            pl.add_mesh(
                mesh,
                scalars=label,
                cmap=cmap,
                clim=clim,
                nan_color="lightgray",
                show_scalar_bar=False,  # one shared bar, managed by set_color_by
            )
        )
        pieces.append((mesh, start, end, idx))
        meshes.append(mesh)

    # ---- color sources: the time-varying data, plus static fiber-dim coords ---
    if color_by is None:
        color_by = find_static_coords(patch, (x, y, z, time_dim, fiber_dim), fiber_dim)
    sources = {label: {"kind": "data"}}
    # Scalar-bar titles: the patch data is already labeled; coords add their units.
    titles = {label: label}
    for name in color_by:
        sources[name] = make_color_source(patch.coords.get_array(name))
        titles[name] = dim_label(patch, name)

    # "t" is a continuous playhead (seconds). It's kept separately from the sample
    # index so that steps smaller than the sample spacing still accumulate.
    state = {"i": 0, "t": 0.0, "color": label, "bar": None}

    def _mark_clipped(lut):
        """Give out-of-range data their own color, as the end of the colormap."""
        # The table's own ends: whatever colormap was given, it is already applied.
        if clipped_below:
            lut.SetBelowRangeColor(*lut.GetTableValue(0))
            lut.SetUseBelowRangeColor(True)
        if clipped_above:
            lut.SetAboveRangeColor(*lut.GetTableValue(lut.GetNumberOfTableValues() - 1))
            lut.SetUseAboveRangeColor(True)

    def _make_lut(name):
        src = sources[name]
        if src["kind"] == "cat":
            names = [str(s) for s in src["uniq"]]
            colors = category_color_list(src["uniq"], (category_colors or {}).get(name))
            if src["has_missing"]:
                # Missing values get their own gray entry, after the real categories
                # so those keep their colors, and so the legend explains the gray.
                names.append(MISSING_LABEL)
                colors = [*colors, MISSING_COLOR]
            n = len(names)
            lut = pv.LookupTable(
                n_values=n,
                scalar_range=(-0.5, n - 0.5),  # band k is centered on integer k
            )
            lut.SetNumberOfTableValues(n)
            for k, color in enumerate(colors):
                lut.SetTableValue(k, *pv.Color(color).float_rgba)
            lut.annotations = {float(k): s for k, s in enumerate(names)}
        else:
            rng = clim if src["kind"] == "data" else src["clim"]
            lut = pv.LookupTable(cmap=cmap, scalar_range=rng)
            if src["kind"] == "data":
                _mark_clipped(lut)
        lut.nan_color = "lightgray"
        return lut

    def _paint(i):
        src = sources[state["color"]]
        for mesh, start, end, idx in pieces:
            full = data[i] if src["kind"] == "data" else src["values"]
            if src["kind"] == "cat" and src["has_missing"]:
                # Missing codes are NaN; point them at the extra "(missing)" entry.
                full = np.where(np.isnan(full), len(src["uniq"]), full)
            seg = full[start:end]
            if idx is None:  # coiled section -> one value for the whole sphere
                value = seg[0] if src["kind"] == "cat" else np.nanmean(seg)
                mesh[label] = np.full(mesh.n_points, value)
            else:
                mesh[label] = seg[idx]

    buttons = {}  # color-source name -> checkbox widget; filled when controls=True

    def set_color_by(name):
        """Switch what the fiber color represents (a key of `color_sources`)."""
        if name not in sources:
            msg = f"Unknown color source '{name}'; options are {list(sources)}"
            raise KeyError(msg)
        state["color"] = name
        # Keep the radio buttons in step when this is called from code.
        for other, widget in buttons.items():
            widget.GetRepresentation().SetState(int(other == name))
        lut = _make_lut(name)
        is_cat = sources[name]["kind"] == "cat"
        for actor in actors:
            actor.mapper.lookup_table = lut
            actor.mapper.scalar_range = lut.scalar_range
            # Categorical codes must not be interpolated *before* color mapping,
            # or a 0 -> 2 boundary passes through category 1's color.
            actor.mapper.SetInterpolateScalarsBeforeMapping(not is_cat)
        if state["bar"] is not None:
            pl.remove_scalar_bar(state["bar"])
        state["bar"] = titles[name]  # PyVista keys scalar bars by title
        if actors:
            # Categorical: hide numeric ticks so only the annotation strings show.
            bar = pl.add_scalar_bar(
                title=titles[name],
                mapper=actors[0].mapper,
                **({"n_labels": 0} if is_cat else {}),
            )
            if sources[name]["kind"] == "data":
                # Swatches beyond the ends of the bar say the data were clipped.
                bar.SetDrawBelowRangeSwatch(clipped_below)
                bar.SetDrawAboveRangeSwatch(clipped_above)
        _paint(state["i"])
        pl.render()

    slider = None  # assigned below; PyVista fires the callback during creation
    rep = None

    def set_step(i):
        """Show time step `i` (clipped to the valid range); returns the index used."""
        i = int(np.clip(round(i), 0, n_time - 1))
        state["i"] = i
        state["t"] = float(rel_times[i])  # slider/direct jumps re-sync the playhead
        _paint(i)
        stamp = format_time(times[i], time_unit)
        pl.add_text(stamp, position="upper_left", font_size=10, name="time_label")
        if rep is not None:
            # VTK builds the slider's value label with sprintf(LabelFormat, value).
            # A format with no % specifier prints literally, so the label becomes
            # the timestamp.
            rep.SetLabelFormat(stamp.replace("%", "%%"))
            rep.SetLabelHeight(0.02)
        pl.render()
        return i

    def nudge(direction):
        """Move the playhead by +/- `key_step` seconds and show the nearest sample."""
        t, j = advance_playhead(rel_times, state["t"], direction, step_s)
        if rep is not None:
            rep.SetValue(j)  # keep the slider handle in sync
        set_step(j)
        state["t"] = t  # set_step snapped it to the sample; restore the fraction

    def step_sample(direction):
        """Move exactly one sample (no matter the sample rate)."""
        j = int(np.clip(state["i"] + direction, 0, n_time - 1))
        if rep is not None:
            rep.SetValue(j)
        set_step(j)  # also re-syncs the playhead to this sample

    # With a single time step there is nothing to scrub or step through, so the
    # slider, keys and key hint are left out (the timestamp is still shown).
    if controls and n_time > 1:
        slider = pl.add_slider_widget(
            lambda v: state.update(i=set_step(v)),
            rng=[0, n_time - 1],
            value=0,
            title=(
                f"{format_time(times[0], time_unit)}  to  "
                f"{format_time(times[-1], time_unit)}"
            ),
            fmt="%.0f",
            pointa=(0.5, 0.9),
            pointb=(0.9, 0.9),
            interaction_event="always",  # live scrubbing; drop if it's laggy
            style="modern",
            tube_width=0.025,
            title_height=0.02,
        )
        rep = slider.GetRepresentation()
        rep.GetSelectedProperty().SetColor(0.2, 0.4, 0.8)

        # The slider still visits every sample; the arrow keys step by time, and
        # Page Up/Down step by exactly one sample. (Up/Down arrows are taken by the
        # terrain interactor's zoom; VTK names Page Up/Down "Prior"/"Next".)
        pl.add_key_event("Right", lambda: nudge(1))
        pl.add_key_event("Left", lambda: nudge(-1))
        pl.add_key_event("Next", lambda: step_sample(1))
        pl.add_key_event("Prior", lambda: step_sample(-1))
        pl.add_text(
            f"Left/Right: {format_duration(step_s)} per press\n"
            "PgUp/PgDn: one time step",
            position="lower_left",
            font_size=8,
            name="key_hint",
        )

    set_color_by(label)  # also draws the scalar bar

    if controls and len(sources) > 1:
        # PyVista has no radio-button widget: one checkbox per source, with
        # callbacks enforcing mutual exclusivity. Positions are in pixels
        # (origin bottom-left), so they don't track window resizing.
        btn_size, row_h = 24, 36
        top = pl.window_size[1] - 100  # below the time label

        def _make_cb(name):
            def _cb(is_on):
                if not is_on:
                    # Clicking the active button: keep it on (radio behavior).
                    buttons[name].GetRepresentation().SetState(1)
                    return
                for other, widget in buttons.items():
                    if other != name:
                        widget.GetRepresentation().SetState(0)
                set_color_by(name)

            return _cb

        for k, name in enumerate(sources):
            y_pix = top - k * row_h
            buttons[name] = pl.add_checkbox_button_widget(
                _make_cb(name),
                value=(k == 0),
                position=(10, y_pix),
                size=btn_size,
                border_size=3,
                color_on="#3366cc",
                color_off="lightgray",
            )
            pl.add_text(
                name,
                position=(10 + btn_size + 10, y_pix + 2),
                font_size=10,
                name=f"opt_label_{k}",
            )

    set_step(0)
    viewer = PatchViewer(
        pl, set_step, set_color_by, sources, meshes, nudge, step_s, step_sample
    )
    if show:
        viewer.show(**kwargs)

    return viewer
