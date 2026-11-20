"""The ``scene`` namespace: 3D scenes, drawn with PyVista (`dasviz.pv`)."""

from dasviz.namespaces._base import DasvizPatchNamespace, DasvizSpoolNamespace
from dasviz.pv import patchviewer


class ScenePatch(DasvizPatchNamespace):
    """3D scenes."""

    name = "scene"
    patchviewer = patchviewer


class SceneSpool(DasvizSpoolNamespace):
    """3D scenes."""

    name = "scene"
    _patch_namespace = ScenePatch
