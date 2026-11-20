"""
Tests for the PyVista patch viewer, patch.scene.patchviewer.

Builds real PyVista meshes off-screen; skipped if PyVista is missing.
The backend-free helpers are tested in tests/prep.
"""

from __future__ import annotations

from datetime import timedelta

import dascore as dc
import numpy as np
import pytest
from dascore.exceptions import ParameterError

pv = pytest.importorskip("pyvista")

from dasviz.prep import category_color_list, data_label  # noqa: E402
from dasviz.pv import PatchViewer, patchviewer  # noqa: E402

pv.OFF_SCREEN = True

N_DIST, N_TIME = 12, 6


def _make_patch(x, y, z, **extra_coords):
    """A (distance, time) patch with spatial coords attached along distance."""
    patch = dc.get_example_patch("random_das", shape=(len(x), N_TIME))
    coords = {"x": x, "y": y, "z": z, **extra_coords}
    return patch.update_coords(
        **{k: ("distance", np.asarray(v)) for k, v in coords.items()}
    )


@pytest.fixture
def line_patch(patch):
    """Straight fiber along x, with data[k, :] == k, so values match positions."""
    n = N_DIST
    patch = _make_patch(np.arange(n, dtype=float), np.zeros(n), np.zeros(n))
    return patch.new(data=np.tile(np.arange(n, dtype=float)[:, None], (1, N_TIME)))


@pytest.fixture
def viewers():
    """Collect viewers so their plotters get closed after each test."""
    made = []
    yield made
    for viewer in made:
        viewer.plotter.close()


def _plot(viewers, patch, **kwargs):
    kwargs.setdefault("controls", False)
    kwargs.setdefault("plotter", pv.Plotter(off_screen=True))
    viewer = patchviewer(patch, **kwargs)
    viewers.append(viewer)
    return viewer


def _label(viewer):
    """Name of the patch-data color source (always the first one)."""
    return next(iter(viewer.color_sources))


class TestPlotPatch3D:
    def test_basic_returns_viewer(self, viewers, line_patch):
        viewer = _plot(viewers, line_patch)
        assert isinstance(viewer.plotter, pv.Plotter)
        assert len(viewer.meshes) == 1
        assert len(viewer.color_sources) == 1  # just the patch data

    def test_creates_plotter_if_none_given(self, viewers, line_patch):
        viewer = patchviewer(line_patch, controls=False)
        viewers.append(viewer)
        assert isinstance(viewer.plotter, pv.Plotter)

    def test_1d_patch_raises(self, line_patch):
        patch = line_patch.mean("time").squeeze()
        with pytest.raises(ValueError, match="2D"):
            patchviewer(patch, controls=False)

    def test_missing_time_dim_raises(self, line_patch):
        with pytest.raises(ValueError, match="2D"):
            patchviewer(line_patch, time_dim="not_a_dim", controls=False)

    def test_tube_data_lines_up_with_geometry(self, viewers, line_patch):
        """
        Regression test for the tube-point -> fiber-sample mapping.

        The fiber runs along +x and data[k, :] == k == x[k], so the scalar at
        every tube point must equal that point's x position. If the `_idx` tag
        didn't survive `.tube()`, these would disagree.
        """
        viewer = _plot(viewers, line_patch, radius=0.05)
        mesh = viewer.meshes[0]
        np.testing.assert_allclose(mesh[_label(viewer)], mesh.points[:, 0], atol=1e-6)

    def test_set_step_changes_scalars(self, viewers, line_patch):
        # Offset each time step by 100 so they're distinguishable.
        data = line_patch.data + np.arange(N_TIME)[None, :] * 100
        viewer = _plot(viewers, line_patch.new(data=data))
        label = _label(viewer)
        viewer.set_step(0)
        first = np.array(viewer.meshes[0][label])
        viewer.set_step(3)
        later = np.array(viewer.meshes[0][label])
        np.testing.assert_allclose(later - first, 300)

    def test_set_step_clips_and_returns_index(self, viewers, line_patch):
        viewer = _plot(viewers, line_patch)
        assert viewer.set_step(-5) == 0
        assert viewer.set_step(10_000) == N_TIME - 1
        assert viewer.set_step(2.4) == 2

    def test_nan_coords_split_fiber(self, viewers):
        n = N_DIST
        x = np.arange(n, dtype=float)
        x[5] = np.nan
        patch = _make_patch(x, np.zeros(n), np.zeros(n))
        viewer = _plot(viewers, patch)
        assert len(viewer.meshes) == 2

    def test_coil_between_gaps_becomes_sphere(self, viewers):
        # line (0-3), gap, coil of 4 samples at one spot, gap, line (10-11)
        n = N_DIST
        x = np.arange(n, dtype=float)
        x[4] = np.nan
        x[5:9] = 100.0
        x[9] = np.nan
        patch = _make_patch(x, np.zeros(n), np.zeros(n))
        radius, scale = 0.1, 3
        viewer = _plot(viewers, patch, radius=radius, sphere_scale_factor=scale)
        assert len(viewer.meshes) == 3
        sphere = viewer.meshes[1]
        np.testing.assert_allclose(sphere.center, [100.0, 0.0, 0.0], atol=1e-6)
        xmin, xmax, *_ = sphere.bounds
        assert (xmax - xmin) == pytest.approx(2 * radius * scale, rel=0.05)

    def test_default_clim_spans_all_time_steps(self, viewers, line_patch):
        viewer = _plot(viewers, line_patch)
        lo, hi = float(np.nanmin(line_patch.data)), float(np.nanmax(line_patch.data))
        assert _data_range(viewer) == pytest.approx((lo, hi))


def _data_range(viewer):
    """Color limits of the fiber mesh (the first actor added to the renderer)."""
    actor = next(iter(viewer.plotter.renderer.actors.values()))
    return actor.mapper.scalar_range


def _lut_endpoints(viewer):
    """RGB of the low and high ends of the fiber's lookup table, as 0-1 floats."""
    actor = next(iter(viewer.plotter.renderer.actors.values()))
    values = np.asarray(actor.mapper.lookup_table.values)
    return values[0, :3] / 255, values[-1, :3] / 255


class TestColoring:
    """Labels, color limits and colormaps follow dasviz.prep, as the waterfall does."""

    @pytest.fixture
    def spiky(self, line_patch):
        """Data with one huge outlier, in the last time step."""
        data = np.array(line_patch.data)
        data[0, -1] = 1e6
        return line_patch.new(data=data)

    def test_default_limits_leave_out_outliers(self, viewers, spiky):
        lo, hi = _data_range(_plot(viewers, spiky))
        assert hi < 1e6  # Tukey's fences, not the max
        assert lo == pytest.approx(float(spiky.data.min()))

    def test_limits_use_all_time_steps(self, viewers, line_patch):
        # Make only the last time step large; the limits must still see it.
        data = np.array(line_patch.data)
        data[:, -1] = 50.0
        viewer = _plot(viewers, line_patch.new(data=data), scale=(0, 1))
        assert _data_range(viewer) == pytest.approx((data.min(), data.max()))

    def test_relative_scale_pair(self, viewers, line_patch):
        lo, hi = line_patch.data.min(), line_patch.data.max()
        viewer = _plot(viewers, line_patch, scale=(0.25, 0.75))
        expected = (lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo))
        assert _data_range(viewer) == pytest.approx(expected)

    def test_absolute_scale(self, viewers, line_patch):
        viewer = _plot(viewers, line_patch, scale=2, scale_type="absolute")
        assert _data_range(viewer) == pytest.approx((-2, 2))

    def test_explicit_clim(self, viewers, line_patch):
        viewer = _plot(viewers, line_patch, clim=(1, 3))
        assert _data_range(viewer) == pytest.approx((1, 3))

    def test_clim_and_scale_conflict(self, line_patch):
        with pytest.raises(ParameterError, match="clim or scale"):
            patchviewer(line_patch, clim=(1, 3), scale=0.5, controls=False)

    def test_bad_scale_type_raises(self, line_patch):
        with pytest.raises(ParameterError, match="scale_type"):
            patchviewer(line_patch, scale_type="nope", controls=False)

    @staticmethod
    def _clipped(viewer):
        """(below, above) flags on the fiber's lookup table."""
        actor = next(iter(viewer.plotter.renderer.actors.values()))
        lut = actor.mapper.lookup_table
        return bool(lut.GetUseBelowRangeColor()), bool(lut.GetUseAboveRangeColor())

    def test_clipping_marked_on_the_side_it_happens(self, viewers, spiky):
        assert self._clipped(_plot(viewers, spiky)) == (False, True)

    def test_clipping_marked_on_both_sides(self, viewers, line_patch):
        assert self._clipped(_plot(viewers, line_patch, clim=(1, 3))) == (True, True)

    def test_no_clipping_no_marks(self, viewers, line_patch):
        assert self._clipped(_plot(viewers, line_patch, scale=(0, 1))) == (False, False)

    def test_clip_swatches_on_scalar_bar(self, viewers, spiky):
        viewer = _plot(viewers, spiky)
        bar = viewer.plotter.scalar_bars[_label(viewer)]
        assert not bar.GetDrawBelowRangeSwatch()
        assert bar.GetDrawAboveRangeSwatch()

    def test_default_colormap_follows_data_type(self, viewers, line_patch):
        plt = pytest.importorskip("matplotlib.pyplot")

        patch = line_patch.update_attrs(data_type="envelope")  # viridis
        low, high = _lut_endpoints(_plot(viewers, patch))
        expected = plt.get_cmap("viridis")
        np.testing.assert_allclose(low, expected(0.0)[:3], atol=2 / 255)
        np.testing.assert_allclose(high, expected(1.0)[:3], atol=2 / 255)

    def test_explicit_cmap_wins(self, viewers, line_patch):
        plt = pytest.importorskip("matplotlib.pyplot")

        patch = line_patch.update_attrs(data_type="envelope")
        low, _ = _lut_endpoints(_plot(viewers, patch, cmap="magma"))
        np.testing.assert_allclose(low, plt.get_cmap("magma")(0.0)[:3], atol=2 / 255)

    def test_data_label_has_type_and_units(self, viewers, line_patch):
        patch = line_patch.update_attrs(data_type="strain_rate", data_units="1/s")
        assert _label(_plot(viewers, patch)) == data_label(patch)

    def test_label_not_empty_without_type_or_units(self, viewers):
        n = N_DIST
        bare = dc.Patch(
            data=np.random.default_rng(0).random((n, N_TIME)),
            coords={
                "distance": np.arange(n, dtype=float),
                "time": np.datetime64("2026-01-01")
                + np.arange(N_TIME) * np.timedelta64(1, "s"),
            },
            dims=("distance", "time"),
        )
        patch = bare.update_coords(
            x=("distance", np.arange(n, dtype=float)),
            y=("distance", np.zeros(n)),
            z=("distance", np.zeros(n)),
        )
        assert _label(_plot(viewers, patch))


class TestColorSources:
    @pytest.fixture
    def sectioned(self):
        n = N_DIST
        sections = np.array(
            ["loose"] * 4 + ["trenched"] * 4 + ["borehole"] * 4, dtype=object
        )
        return _make_patch(
            np.arange(n, dtype=float),
            np.zeros(n),
            np.zeros(n),
            section=sections,
            offset=np.linspace(10.0, 20.0, n),
        )

    def test_static_coords_autodetected(self, viewers, sectioned):
        names = list(_plot(viewers, sectioned).color_sources)
        assert "section" in names
        assert "offset" in names
        assert not {"x", "y", "z", "time", "distance"} & set(names)

    def test_color_by_restricts_options(self, viewers, sectioned):
        viewer = _plot(viewers, sectioned, color_by=["section"])
        assert list(viewer.color_sources)[1:] == ["section"]

    def test_categorical_codes_on_mesh(self, viewers, sectioned):
        viewer = _plot(viewers, sectioned, color_by=["section"])
        viewer.set_color_by("section")
        codes = np.unique(np.rint(viewer.meshes[0][_label(viewer)]).astype(int))
        assert list(codes) == [0, 1, 2]  # sorted: borehole, loose, trench

    def test_numeric_static_coord_on_mesh(self, viewers, sectioned):
        viewer = _plot(viewers, sectioned, color_by=["offset"])
        viewer.set_color_by("offset")
        vals = np.asarray(viewer.meshes[0][_label(viewer)])
        assert vals.min() == pytest.approx(10.0)
        assert vals.max() == pytest.approx(20.0)

    def test_switching_back_to_data(self, viewers, sectioned):
        viewer = _plot(viewers, sectioned, color_by=["section"])
        label = _label(viewer)
        viewer.set_color_by("section")
        viewer.set_color_by(label)
        expected = sectioned.transpose("time", "distance").data[0].max()
        assert viewer.meshes[0][label].max() == pytest.approx(expected)

    def test_switching_sources_leaves_one_scalar_bar(self, viewers, sectioned):
        # Bars are keyed by title, which differs from the source name.
        viewer = _plot(viewers, sectioned, color_by=["offset"])
        viewer.set_color_by("offset")
        viewer.set_color_by(_label(viewer))
        assert len(viewer.plotter.scalar_bars) == 1

    def test_categorical_lut_uses_category_colors(self, viewers, sectioned):
        viewer = _plot(
            viewers,
            sectioned,
            color_by=["section"],
            category_colors={"section": {"loose": "red"}},
        )
        viewer.set_color_by("section")
        actor = next(iter(viewer.plotter.renderer.actors.values()))
        lut = actor.mapper.lookup_table
        # Sorted categories: borehole, loose, trenched.
        expected = category_color_list(
            ["borehole", "loose", "trenched"], {"loose": "red"}
        )
        for k, color in enumerate(expected):
            np.testing.assert_allclose(
                lut.GetTableValue(k)[:3], pv.Color(color).float_rgb, atol=1 / 255
            )

    def test_unknown_color_source_raises(self, viewers, sectioned):
        viewer = _plot(viewers, sectioned)
        with pytest.raises(KeyError, match="Unknown color source"):
            viewer.set_color_by("nope")

    def test_missing_strings_do_not_become_categories(self, viewers):
        n = N_DIST
        sections = np.array([""] * 2 + ["loose"] * 5 + ["trenched"] * 5, dtype=object)
        patch = _make_patch(
            np.arange(n, dtype=float), np.zeros(n), np.zeros(n), section=sections
        )
        viewer = _plot(viewers, patch, color_by=["section"])
        assert list(viewer.color_sources["section"]["uniq"]) == ["loose", "trenched"]

    def test_missing_gets_its_own_gray_legend_entry(self, viewers):
        n = N_DIST
        sections = np.array([""] * 2 + ["loose"] * 5 + ["trenched"] * 5, dtype=object)
        patch = _make_patch(
            np.arange(n, dtype=float), np.zeros(n), np.zeros(n), section=sections
        )
        viewer = _plot(viewers, patch, color_by=["section"])
        viewer.set_color_by("section")
        actor = next(iter(viewer.plotter.renderer.actors.values()))
        lut = actor.mapper.lookup_table
        assert lut.GetNumberOfTableValues() == 3  # lab, roof, (missing)
        np.testing.assert_allclose(
            lut.GetTableValue(2)[:3], pv.Color("lightgray").float_rgb, atol=1 / 255
        )
        # The first two samples are missing; they must land on the extra entry.
        codes = np.rint(viewer.meshes[0][_label(viewer)]).astype(int)
        assert codes.min() == 0
        assert codes.max() == 2
        assert (codes == 2).any()

    def test_no_missing_entry_when_nothing_is_missing(self, viewers):
        n = N_DIST
        sections = np.array(["loose"] * 6 + ["trenched"] * 6, dtype=object)
        patch = _make_patch(
            np.arange(n, dtype=float), np.zeros(n), np.zeros(n), section=sections
        )
        viewer = _plot(viewers, patch, color_by=["section"])
        viewer.set_color_by("section")
        actor = next(iter(viewer.plotter.renderer.actors.values()))
        assert actor.mapper.lookup_table.GetNumberOfTableValues() == 2


class TestKeyStepping:
    """Arrow keys step by data *time*; `viewer.nudge` is what the keys call."""

    @pytest.fixture
    def time_patch(self, line_patch):
        """Same geometry, but every value equals its time index (0..N_TIME-1)."""
        data = np.tile(np.arange(N_TIME, dtype=float)[None, :], (N_DIST, 1))
        return line_patch.new(data=data)

    @pytest.fixture
    def dt(self, time_patch):
        """Sample spacing in seconds."""
        t = time_patch.get_array("time")
        return float((t[1] - t[0]) / np.timedelta64(1, "s"))

    @staticmethod
    def _shown_index(viewer):
        return int(round(float(viewer.meshes[0][_label(viewer)][0])))

    def test_default_step_is_at_least_one_sample(self, viewers, time_patch, dt):
        viewer = _plot(viewers, time_patch)
        assert viewer.key_step >= dt - 1e-12

    @pytest.mark.parametrize(
        "step", [np.timedelta64(2, "s"), 2.0, timedelta(seconds=2)]
    )
    def test_explicit_step_converted_to_seconds(self, viewers, time_patch, step):
        assert _plot(viewers, time_patch, key_step=step).key_step == 2.0

    def test_bad_step_raises(self, time_patch):
        with pytest.raises(ValueError, match="positive"):
            patchviewer(time_patch, key_step=-1, controls=False)

    def test_nudge_advances_by_time_not_samples(self, viewers, time_patch, dt):
        viewer = _plot(viewers, time_patch, key_step=3 * dt)
        viewer.set_step(0)
        viewer.nudge(1)
        assert self._shown_index(viewer) == 3
        viewer.nudge(-1)
        assert self._shown_index(viewer) == 0

    def test_sub_sample_steps_accumulate(self, viewers, time_patch, dt):
        viewer = _plot(viewers, time_patch, key_step=dt / 2)
        viewer.set_step(0)
        viewer.nudge(1)
        assert self._shown_index(viewer) in (0, 1)  # half a sample: either is fine
        viewer.nudge(1)
        assert self._shown_index(viewer) == 1  # but two presses must reach sample 1

    def test_nudge_clamps_at_ends(self, viewers, time_patch, dt):
        viewer = _plot(viewers, time_patch, key_step=dt)
        for _ in range(N_TIME * 3):
            viewer.nudge(1)
        assert self._shown_index(viewer) == N_TIME - 1
        for _ in range(N_TIME * 3):
            viewer.nudge(-1)
        assert self._shown_index(viewer) == 0

    def test_set_step_resyncs_playhead(self, viewers, time_patch, dt):
        # Jumping with set_step (what the slider does) moves where nudge starts.
        viewer = _plot(viewers, time_patch, key_step=dt)
        viewer.set_step(2)
        viewer.nudge(1)
        assert self._shown_index(viewer) == 3

    def test_slider_still_reaches_every_sample(self, viewers, time_patch, dt):
        viewer = _plot(viewers, time_patch, key_step=3 * dt)
        shown = []
        for i in range(N_TIME):
            viewer.set_step(i)
            shown.append(self._shown_index(viewer))
        assert shown == list(range(N_TIME))

    def test_step_sample_moves_one_sample_regardless_of_key_step(
        self, viewers, time_patch, dt
    ):
        viewer = _plot(viewers, time_patch, key_step=3 * dt)
        viewer.set_step(1)
        viewer.step_sample(1)
        assert self._shown_index(viewer) == 2
        viewer.step_sample(-1)
        viewer.step_sample(-1)
        assert self._shown_index(viewer) == 0
        viewer.step_sample(-1)  # clamps at the start
        assert self._shown_index(viewer) == 0

    def test_step_sample_resyncs_playhead_for_nudge(self, viewers, time_patch, dt):
        viewer = _plot(viewers, time_patch, key_step=dt / 2)
        viewer.nudge(1)  # playhead now half a sample in
        viewer.step_sample(1)  # jump to sample 1, dropping the half-sample offset
        viewer.nudge(1)
        assert self._shown_index(viewer) in (1, 2)
        viewer.nudge(1)
        assert self._shown_index(viewer) == 2


class TestControls:
    def test_controls_do_not_break_creation(self, viewers, line_patch):
        """
        Widgets may not be available on every off-screen plotter. Skip rather
        than fail if PyVista can't create them without a real interactor.
        """
        try:
            viewer = _plot(viewers, line_patch, controls=True)
        except (AttributeError, RuntimeError) as exc:
            pytest.skip(f"widgets unavailable off-screen: {exc!r}")
        assert viewer.set_step(1) == 1

    @pytest.fixture
    def one_sample(self, line_patch):
        patch = line_patch.select(time=(0, 1), samples=True)
        assert patch.get_array("time").size == 1  # the fixture did what it says
        return patch

    def test_single_time_step_works_without_controls(self, viewers, one_sample):
        viewer = _plot(viewers, one_sample)
        assert viewer.set_step(5) == 0
        viewer.nudge(1)  # nothing to move to, but must not fail
        viewer.step_sample(-1)

    def test_single_time_step_has_no_slider(self, viewers, one_sample):
        try:
            viewer = _plot(viewers, one_sample, controls=True)
        except (AttributeError, RuntimeError) as exc:
            pytest.skip(f"widgets unavailable off-screen: {exc!r}")
        assert not viewer.plotter.slider_widgets


class TestNamespace:
    def test_bound_on_scene_namespace(self, viewers, line_patch):
        viewer = line_patch.scene.patchviewer(
            controls=False, plotter=pv.Plotter(off_screen=True)
        )
        viewers.append(viewer)
        assert isinstance(viewer, PatchViewer)

    def test_spool_points_to_patch(self, spool):
        with pytest.raises(AttributeError, match=r"chunk\(time=None\)\[0\]\.scene"):
            spool.scene.patchviewer  # noqa: B018
