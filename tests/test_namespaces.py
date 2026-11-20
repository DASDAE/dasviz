"""Tests that DASCore finds dasviz's namespaces through their entry points."""

from importlib.metadata import entry_points

import dascore as dc
import pytest

from dasviz.namespaces._base import DasvizPatchNamespace, DasvizSpoolNamespace

NAMES = ("fig", "nb", "scene", "anim")


def _dasviz_entry_points(group):
    """Return dasviz's entry points in a group, by name."""
    return {
        ep.name: ep
        for ep in entry_points(group=group)
        if ep.value.startswith("dasviz.")
    }


class TestEntryPoints:
    """Each namespace is registered for both Patch and Spool."""

    @pytest.mark.parametrize(
        ("group", "base"),
        [
            ("dascore.patch_namespace", DasvizPatchNamespace),
            ("dascore.spool_namespace", DasvizSpoolNamespace),
        ],
    )
    def test_registered(self, group, base):
        eps = _dasviz_entry_points(group)
        assert set(eps) == set(NAMES)
        for name, ep in eps.items():
            cls = ep.load()
            assert issubclass(cls, base)
            assert cls.name == name

    def test_spool_classes_point_to_patch_classes(self):
        patch_eps = _dasviz_entry_points("dascore.patch_namespace")
        for name, ep in _dasviz_entry_points("dascore.spool_namespace").items():
            assert ep.load()._patch_namespace is patch_eps[name].load()


class TestAccess:
    """The namespaces resolve on DASCore objects without importing dasviz."""

    @pytest.mark.parametrize("name", NAMES)
    def test_on_patch(self, patch, name):
        namespace = getattr(patch, name)
        assert namespace._obj is patch
        assert type(namespace) is dc.Patch.get_registered_namespaces()[name]

    @pytest.mark.parametrize("name", NAMES)
    def test_on_spool(self, spool, name):
        namespace = getattr(spool, name)
        assert namespace._obj is spool
        assert isinstance(namespace, DasvizSpoolNamespace)

    def test_repr_lists_methods(self, patch, spool):
        assert repr(patch.fig) == "<dasviz 'fig' namespace on Patch: waterfall>"
        assert repr(patch.scene).endswith("scene' namespace on Patch: patchviewer>")
        assert repr(patch.anim).endswith("no methods yet>")
        assert "'fig' namespace on" in repr(spool.fig)


class TestSpoolPointsToPatch:
    """A patch method called on a spool says how to get a patch."""

    def test_patch_method_on_spool(self, spool):
        with pytest.raises(AttributeError, match=r"chunk\(time=None\)\[0\]\.fig"):
            spool.fig.waterfall  # noqa: B018

    def test_unknown_name(self, spool):
        with pytest.raises(AttributeError, match="has no attribute 'nope'"):
            spool.fig.nope  # noqa: B018

    def test_hasattr_is_false(self, spool):
        assert not hasattr(spool.fig, "waterfall")
