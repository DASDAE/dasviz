"""
The ``scene`` namespace: 3D scenes.

Empty for now; PyVista renderers (`dasviz.pv`) go here.
"""

from dasviz.namespaces._base import DasvizPatchNamespace, DasvizSpoolNamespace


class ScenePatch(DasvizPatchNamespace):
    """3D scenes. Empty for now; PyVista renderers (`dasviz.pv`) go here."""

    name = "scene"


class SceneSpool(DasvizSpoolNamespace):
    """3D scenes. Empty for now; PyVista renderers (`dasviz.pv`) go here."""

    name = "scene"
    _patch_namespace = ScenePatch
